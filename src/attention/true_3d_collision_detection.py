"""
Gerçek 3D Çakışma Analizi Modülü
"""

import numpy as np
import math
import time
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .gaze_3d_types import GazePoint3D
from .enhanced_head_pose import HeadPose3D, GazeVector3D
from detection.object_3d_positioning import Object3DPosition
from spatial.spatial_environment_analyzer import SpatialObject, SpatialPerson


@dataclass
class Collision3DEvent:
    """3D çakışma event verisi"""
    collision_id: str
    
    # Gaze bilgileri
    gaze_origin: Tuple[float, float, float]
    gaze_direction: Tuple[float, float, float]
    gaze_point: Optional[Tuple[float, float, float]]
    
    # Obje bilgileri
    object_id: str
    object_label: str
    object_position: Tuple[float, float, float]
    object_dimensions: Tuple[float, float, float]
    object_bounding_box: List[Tuple[float, float, float]]
    
    # Çakışma geometrisi
    collision_point_3d: Tuple[float, float, float]
    collision_distance: float
    collision_angle: float  # Gaze ile obje normali arasındaki açı
    penetration_depth: float  # Obje içine ne kadar girdiği
    
    # Güvenilirlik ve kalite
    collision_confidence: float
    geometric_precision: float
    
    # Temporal bilgiler
    start_timestamp: float
    end_timestamp: Optional[float] = None
    duration: float = 0.0
    
    # Çakışma durumu
    is_active: bool = True
    collision_type: str = "gaze_object"  # "gaze_object", "predicted_gaze", "peripheral"
    
    # Ek bilgiler
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_duration(self, current_time: float):
        """Çakışma süresini güncelle"""
        if self.is_active:
            self.duration = current_time - self.start_timestamp
    
    def end_collision(self, end_time: float):
        """Çakışmayı sonlandır"""
        self.end_timestamp = end_time
        self.duration = end_time - self.start_timestamp
        self.is_active = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary formatına çevir"""
        return {
            'collision_id': self.collision_id,
            'gaze_origin': self.gaze_origin,
            'gaze_direction': self.gaze_direction,
            'gaze_point': self.gaze_point,
            'object_id': self.object_id,
            'object_label': self.object_label,
            'object_position': self.object_position,
            'object_dimensions': self.object_dimensions,
            'collision_point_3d': self.collision_point_3d,
            'collision_distance': self.collision_distance,
            'collision_angle': self.collision_angle,
            'penetration_depth': self.penetration_depth,
            'collision_confidence': self.collision_confidence,
            'geometric_precision': self.geometric_precision,
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
            'duration': self.duration,
            'is_active': self.is_active,
            'collision_type': self.collision_type,
            'metadata': self.metadata
        }


class RayBoxIntersection:
    """Ray-Box çakışma tespiti"""
    
    @staticmethod
    def ray_aabb_intersection(ray_origin: Tuple[float, float, float],
                            ray_direction: Tuple[float, float, float],
                            aabb_min: Tuple[float, float, float],
                            aabb_max: Tuple[float, float, float]) -> Tuple[bool, float, Tuple[float, float, float]]:
        """Ray-AABB çakışma testi"""
        try:
            # Ray direction normalize et
            ray_dir = np.array(ray_direction)
            ray_dir_norm = np.linalg.norm(ray_dir)
            if ray_dir_norm == 0:
                return False, float('inf'), (0, 0, 0)
            ray_dir = ray_dir / ray_dir_norm
            
            # AABB sınırları
            min_bound = np.array(aabb_min)
            max_bound = np.array(aabb_max)
            ray_orig = np.array(ray_origin)
            
            # Slab algoritması
            t_min = float('-inf')
            t_max = float('inf')
            
            for i in range(3):
                if abs(ray_dir[i]) < 1e-10:
                    # Ray düzleme paralel
                    if ray_orig[i] < min_bound[i] or ray_orig[i] > max_bound[i]:
                        return False, float('inf'), (0, 0, 0)
                else:
                    # t değerlerini hesapla
                    t1 = (min_bound[i] - ray_orig[i]) / ray_dir[i]
                    t2 = (max_bound[i] - ray_orig[i]) / ray_dir[i]
                    
                    if t1 > t2:
                        t1, t2 = t2, t1
                    
                    t_min = max(t_min, t1)
                    t_max = min(t_max, t2)
                    
                    if t_min > t_max:
                        return False, float('inf'), (0, 0, 0)
            
            # Çakışma noktası
            if t_min >= 0:
                intersection_point = ray_orig + t_min * ray_dir
                return True, t_min, tuple(intersection_point)
            elif t_max >= 0:
                intersection_point = ray_orig + t_max * ray_dir
                return True, t_max, tuple(intersection_point)
            else:
                return False, float('inf'), (0, 0, 0)
                
        except Exception as e:
            logging.error(f"Ray-AABB çakışma testi hatası: {e}")
            return False, float('inf'), (0, 0, 0)
    
    @staticmethod
    def ray_oriented_box_intersection(ray_origin: Tuple[float, float, float],
                                    ray_direction: Tuple[float, float, float],
                                    box_center: Tuple[float, float, float],
                                    box_dimensions: Tuple[float, float, float],
                                    box_orientation: Tuple[float, float, float]) -> Tuple[bool, float, Tuple[float, float, float]]:
        """Ray-Oriented Bounding Box çakışma testi"""
        try:
            # Şimdilik basit AABB olarak işle (gelecekte rotation eklenecek)
            aabb_min = (
                box_center[0] - box_dimensions[0] / 2,
                box_center[1] - box_dimensions[1] / 2,
                box_center[2] - box_dimensions[2] / 2
            )
            aabb_max = (
                box_center[0] + box_dimensions[0] / 2,
                box_center[1] + box_dimensions[1] / 2,
                box_center[2] + box_dimensions[2] / 2
            )
            
            return RayBoxIntersection.ray_aabb_intersection(
                ray_origin, ray_direction, aabb_min, aabb_max
            )
            
        except Exception as e:
            logging.error(f"Ray-OBB çakışma testi hatası: {e}")
            return False, float('inf'), (0, 0, 0)


class GazeGeometryAnalyzer:
    """Gaze geometrisi analiz sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Parametreler
        self.max_gaze_distance = config.get('max_gaze_distance', 3000.0)  # mm
        self.angle_threshold = config.get('angle_threshold', 45.0)  # derece
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        
    def analyze_gaze_object_geometry(self, gaze_vector: GazeVector3D,
                                   object_3d: Object3DPosition) -> Dict[str, Any]:
        """Gaze-obje geometrisi analiz et"""
        try:
            # Ray-box çakışması
            intersection_result = RayBoxIntersection.ray_oriented_box_intersection(
                gaze_vector.origin_3d,
                gaze_vector.direction_3d,
                object_3d.position_3d,
                object_3d.dimensions_3d,
                object_3d.orientation_3d
            )
            
            has_intersection, distance, intersection_point = intersection_result
            
            if not has_intersection:
                return {
                    'has_intersection': False,
                    'distance': float('inf'),
                    'intersection_point': None,
                    'collision_angle': 0.0,
                    'penetration_depth': 0.0,
                    'confidence': 0.0
                }
            
            # Çakışma açısı hesapla
            collision_angle = self._calculate_collision_angle(
                gaze_vector.direction_3d, object_3d.position_3d, intersection_point
            )
            
            # Penetration depth hesapla
            penetration_depth = self._calculate_penetration_depth(
                gaze_vector.origin_3d, gaze_vector.direction_3d,
                object_3d.position_3d, object_3d.dimensions_3d
            )
            
            # Güvenilirlik hesapla
            confidence = self._calculate_geometry_confidence(
                gaze_vector, object_3d, distance, collision_angle
            )
            
            return {
                'has_intersection': True,
                'distance': distance,
                'intersection_point': intersection_point,
                'collision_angle': collision_angle,
                'penetration_depth': penetration_depth,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Gaze-obje geometrisi analiz hatası: {e}")
            return {
                'has_intersection': False,
                'distance': float('inf'),
                'intersection_point': None,
                'collision_angle': 0.0,
                'penetration_depth': 0.0,
                'confidence': 0.0
            }
    
    def _calculate_collision_angle(self, gaze_direction: Tuple[float, float, float],
                                 object_center: Tuple[float, float, float],
                                 intersection_point: Tuple[float, float, float]) -> float:
        """Çakışma açısını hesapla"""
        try:
            # Obje yüzeyinin normali (basit yaklaşım)
            # Gerçekte obje yüzeyinin hangi tarafına çarptığını belirlememiz gerekir
            
            # Intersection point'ten obje merkezine vektör
            to_center = np.array(object_center) - np.array(intersection_point)
            to_center_norm = np.linalg.norm(to_center)
            
            if to_center_norm == 0:
                return 0.0
            
            to_center = to_center / to_center_norm
            
            # Gaze direction ile angle
            gaze_dir = np.array(gaze_direction)
            gaze_dir_norm = np.linalg.norm(gaze_dir)
            
            if gaze_dir_norm == 0:
                return 0.0
            
            gaze_dir = gaze_dir / gaze_dir_norm
            
            # Cosine açısı
            cos_angle = np.dot(gaze_dir, to_center)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            angle_rad = math.acos(abs(cos_angle))
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            self.logger.error(f"Çakışma açısı hesaplama hatası: {e}")
            return 0.0
    
    def _calculate_penetration_depth(self, gaze_origin: Tuple[float, float, float],
                                   gaze_direction: Tuple[float, float, float],
                                   object_center: Tuple[float, float, float],
                                   object_dimensions: Tuple[float, float, float]) -> float:
        """Obje içine penetrasyon derinliği"""
        try:
            # Ray'in obje içindeki mesafesi
            # Entry ve exit point'leri bul
            
            aabb_min = (
                object_center[0] - object_dimensions[0] / 2,
                object_center[1] - object_dimensions[1] / 2,
                object_center[2] - object_dimensions[2] / 2
            )
            aabb_max = (
                object_center[0] + object_dimensions[0] / 2,
                object_center[1] + object_dimensions[1] / 2,
                object_center[2] + object_dimensions[2] / 2
            )
            
            # Entry point
            entry_result = RayBoxIntersection.ray_aabb_intersection(
                gaze_origin, gaze_direction, aabb_min, aabb_max
            )
            
            if not entry_result[0]:
                return 0.0
            
            entry_distance = entry_result[1]
            
            # Exit point için ray'i tersten çıkar
            # Bu basit bir yaklaşım - daha sophisticated hesaplama yapılabilir
            max_dimension = max(object_dimensions)
            penetration = min(max_dimension / 2, 50.0)  # Maksimum 50mm penetrasyon
            
            return penetration
            
        except Exception as e:
            self.logger.error(f"Penetrasyon derinliği hesaplama hatası: {e}")
            return 0.0
    
    def _calculate_geometry_confidence(self, gaze_vector: GazeVector3D,
                                     object_3d: Object3DPosition,
                                     distance: float,
                                     collision_angle: float) -> float:
        """Geometri güvenilirliği hesapla"""
        try:
            confidence = 0.0
            
            # Gaze confidence
            confidence += gaze_vector.confidence * 0.4
            
            # Obje confidence
            confidence += object_3d.confidence * 0.3
            
            # Mesafe faktörü
            if distance < self.max_gaze_distance:
                distance_factor = 1.0 - (distance / self.max_gaze_distance)
                confidence += distance_factor * 0.2
            
            # Açı faktörü
            if collision_angle < self.angle_threshold:
                angle_factor = 1.0 - (collision_angle / self.angle_threshold)
                confidence += angle_factor * 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Geometri confidence hesaplama hatası: {e}")
            return 0.0


class True3DCollisionDetector:
    """Ana 3D çakışma tespit sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alt modüller
        self.geometry_analyzer = GazeGeometryAnalyzer(config.get('geometry_analysis', {}))
        
        # Çakışma yönetimi
        self.active_collisions = {}  # collision_id -> Collision3DEvent
        self.collision_history = deque(maxlen=config.get('max_collision_history', 500))
        
        # Parametreler
        self.collision_timeout = config.get('collision_timeout', 10.0)  # saniye
        self.min_collision_duration = config.get('min_collision_duration', 0.1)  # saniye
        self.collision_confidence_threshold = config.get('collision_confidence_threshold', 0.4)
        
        # Statistics
        self.collision_stats = {
            'total_collisions': 0,
            'active_collision_count': 0,
            'collision_types': defaultdict(int),
            'object_collision_counts': defaultdict(int)
        }
        
        self.logger.info("True3DCollisionDetector başlatıldı")
    
    def detect_3d_collisions(self, gaze_vector: GazeVector3D,
                           objects_3d: List[Object3DPosition],
                           spatial_people: List[SpatialPerson] = None) -> List[Collision3DEvent]:
        """3D çakışmaları tespit et"""
        try:
            current_time = time.time()
            collision_events = []
            
            # Timeout olan çakışmaları temizle
            self._cleanup_expired_collisions(current_time)
            
            if gaze_vector is None or gaze_vector.confidence < self.collision_confidence_threshold:
                return collision_events
            
            # Mevcut çakışmaları kontrol et
            current_collision_ids = set()
            
            # Obje çakışmaları
            for obj_3d in objects_3d:
                collision_result = self._check_gaze_object_collision(
                    gaze_vector, obj_3d, current_time
                )
                
                if collision_result:
                    collision_events.append(collision_result)
                    current_collision_ids.add(collision_result.collision_id)
            
            # Bitmiş çakışmaları tespit et
            self._finalize_ended_collisions(current_collision_ids, current_time)
            
            # Statistics güncelle
            self._update_collision_statistics(collision_events)
            
            return collision_events
            
        except Exception as e:
            self.logger.error(f"3D çakışma tespiti hatası: {e}")
            return []
    
    def _check_gaze_object_collision(self, gaze_vector: GazeVector3D,
                                   object_3d: Object3DPosition,
                                   current_time: float) -> Optional[Collision3DEvent]:
        """Gaze-obje çakışması kontrol et"""
        try:
            # Geometri analizi
            geometry_result = self.geometry_analyzer.analyze_gaze_object_geometry(
                gaze_vector, object_3d
            )
            
            if not geometry_result['has_intersection']:
                return None
            
            # Güvenilirlik kontrolü
            if geometry_result['confidence'] < self.collision_confidence_threshold:
                return None
            
            # Collision ID oluştur
            collision_id = f"gaze_{object_3d.object_id}_{int(current_time * 1000) % 100000}"
            
            # Mevcut çakışma var mı kontrol et
            existing_collision = self._find_existing_collision(gaze_vector, object_3d)
            
            if existing_collision:
                # Mevcut çakışmayı güncelle
                existing_collision.update_duration(current_time)
                existing_collision.collision_confidence = geometry_result['confidence']
                existing_collision.collision_point_3d = geometry_result['intersection_point']
                existing_collision.collision_distance = geometry_result['distance']
                existing_collision.collision_angle = geometry_result['collision_angle']
                existing_collision.penetration_depth = geometry_result['penetration_depth']
                
                return existing_collision
            else:
                # Yeni çakışma oluştur
                collision_event = Collision3DEvent(
                    collision_id=collision_id,
                    gaze_origin=gaze_vector.origin_3d,
                    gaze_direction=gaze_vector.direction_3d,
                    gaze_point=gaze_vector.target_point_3d,
                    object_id=object_3d.object_id,
                    object_label=object_3d.label,
                    object_position=object_3d.position_3d,
                    object_dimensions=object_3d.dimensions_3d,
                    object_bounding_box=object_3d.bounding_box_3d,
                    collision_point_3d=geometry_result['intersection_point'],
                    collision_distance=geometry_result['distance'],
                    collision_angle=geometry_result['collision_angle'],
                    penetration_depth=geometry_result['penetration_depth'],
                    collision_confidence=geometry_result['confidence'],
                    geometric_precision=self._calculate_geometric_precision(geometry_result),
                    start_timestamp=current_time,
                    collision_type="gaze_object",
                    metadata={
                        'object_class_id': object_3d.class_id,
                        'object_tracking_id': object_3d.tracking_id,
                        'gaze_confidence': gaze_vector.confidence,
                        'object_confidence': object_3d.confidence
                    }
                )
                
                self.active_collisions[collision_id] = collision_event
                self.collision_stats['total_collisions'] += 1
                
                return collision_event
                
        except Exception as e:
            self.logger.error(f"Gaze-obje çakışma kontrolü hatası: {e}")
            return None
    
    def _find_existing_collision(self, gaze_vector: GazeVector3D,
                               object_3d: Object3DPosition) -> Optional[Collision3DEvent]:
        """Mevcut çakışma bul"""
        try:
            for collision in self.active_collisions.values():
                if (collision.object_id == object_3d.object_id and
                    collision.is_active and
                    collision.collision_type == "gaze_object"):
                    return collision
            return None
            
        except Exception as e:
            self.logger.error(f"Mevcut çakışma bulma hatası: {e}")
            return None
    
    def _calculate_geometric_precision(self, geometry_result: Dict[str, Any]) -> float:
        """Geometrik hassasiyet hesapla"""
        try:
            precision = 0.5  # Başlangıç
            
            # Mesafe hassasiyeti
            distance = geometry_result.get('distance', float('inf'))
            if distance < 1000:  # 1m içinde
                precision += 0.2
            
            # Açı hassasiyeti
            angle = geometry_result.get('collision_angle', 90)
            if angle < 30:  # 30 derece altında
                precision += 0.2
            
            # Penetrasyon hassasiyeti
            penetration = geometry_result.get('penetration_depth', 0)
            if penetration > 0:
                precision += 0.1
            
            return min(1.0, precision)
            
        except Exception as e:
            self.logger.error(f"Geometrik hassasiyet hesaplama hatası: {e}")
            return 0.5
    
    def _cleanup_expired_collisions(self, current_time: float):
        """Süresi dolmuş çakışmaları temizle"""
        try:
            expired_ids = []
            
            for collision_id, collision in self.active_collisions.items():
                if (current_time - collision.start_timestamp) > self.collision_timeout:
                    collision.end_collision(current_time)
                    self.collision_history.append(collision)
                    expired_ids.append(collision_id)
            
            for collision_id in expired_ids:
                del self.active_collisions[collision_id]
                
        except Exception as e:
            self.logger.error(f"Süresi dolmuş çakışma temizleme hatası: {e}")
    
    def _finalize_ended_collisions(self, current_collision_ids: set, current_time: float):
        """Bitmiş çakışmaları sonlandır"""
        try:
            ended_collision_ids = set(self.active_collisions.keys()) - current_collision_ids
            
            for collision_id in ended_collision_ids:
                collision = self.active_collisions[collision_id]
                
                # Minimum süre kontrolü
                if collision.duration >= self.min_collision_duration:
                    collision.end_collision(current_time)
                    self.collision_history.append(collision)
                
                del self.active_collisions[collision_id]
                
        except Exception as e:
            self.logger.error(f"Bitmiş çakışma sonlandırma hatası: {e}")
    
    def _update_collision_statistics(self, collision_events: List[Collision3DEvent]):
        """Çakışma istatistiklerini güncelle"""
        try:
            self.collision_stats['active_collision_count'] = len(self.active_collisions)
            
            for collision in collision_events:
                self.collision_stats['collision_types'][collision.collision_type] += 1
                self.collision_stats['object_collision_counts'][collision.object_label] += 1
                
        except Exception as e:
            self.logger.error(f"İstatistik güncelleme hatası: {e}")
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """Çakışma istatistiklerini al"""
        try:
            return {
                'total_collisions': self.collision_stats['total_collisions'],
                'active_collision_count': self.collision_stats['active_collision_count'],
                'collision_types': dict(self.collision_stats['collision_types']),
                'object_collision_counts': dict(self.collision_stats['object_collision_counts']),
                'collision_history_size': len(self.collision_history),
                'avg_collision_duration': self._calculate_avg_collision_duration()
            }
            
        except Exception as e:
            self.logger.error(f"İstatistik alma hatası: {e}")
            return {}
    
    def _calculate_avg_collision_duration(self) -> float:
        """Ortalama çakışma süresi hesapla"""
        try:
            if not self.collision_history:
                return 0.0
            
            total_duration = sum(collision.duration for collision in self.collision_history)
            return total_duration / len(self.collision_history)
            
        except Exception as e:
            self.logger.error(f"Ortalama çakışma süresi hesaplama hatası: {e}")
            return 0.0
    
    def get_active_collisions(self) -> List[Collision3DEvent]:
        """Aktif çakışmaları al"""
        return list(self.active_collisions.values())
    
    def get_collision_history(self, limit: int = 50) -> List[Collision3DEvent]:
        """Çakışma geçmişini al"""
        return list(self.collision_history)[-limit:]
    
    def clear_collision_history(self):
        """Çakışma geçmişini temizle"""
        self.collision_history.clear()
        self.collision_stats['total_collisions'] = 0
        self.collision_stats['collision_types'].clear()
        self.collision_stats['object_collision_counts'].clear()


class CollisionAnalytics3D:
    """3D çakışma analitik sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analytics parametreleri
        self.analytics_window = config.get('analytics_window', 300)  # 5 dakika
        self.min_meaningful_duration = config.get('min_meaningful_duration', 1.0)  # saniye
        
    def analyze_collision_patterns(self, collision_history: List[Collision3DEvent]) -> Dict[str, Any]:
        """Çakışma desenlerini analiz et"""
        try:
            if not collision_history:
                return {}
            
            # Nesne türlerine göre çakışma analizi
            object_analysis = self._analyze_object_collisions(collision_history)
            
            # Temporal analizi
            temporal_analysis = self._analyze_temporal_patterns(collision_history)
            
            # Spatial analizi
            spatial_analysis = self._analyze_spatial_patterns(collision_history)
            
            # Attention analizi
            attention_analysis = self._analyze_attention_patterns(collision_history)
            
            return {
                'object_analysis': object_analysis,
                'temporal_analysis': temporal_analysis,
                'spatial_analysis': spatial_analysis,
                'attention_analysis': attention_analysis,
                'summary': self._generate_collision_summary(collision_history)
            }
            
        except Exception as e:
            self.logger.error(f"Çakışma deseni analizi hatası: {e}")
            return {}
    
    def _analyze_object_collisions(self, collisions: List[Collision3DEvent]) -> Dict[str, Any]:
        """Nesne çakışma analizi"""
        try:
            object_stats = defaultdict(lambda: {
                'count': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
                'avg_confidence': 0.0,
                'avg_distance': 0.0
            })
            
            for collision in collisions:
                if collision.duration >= self.min_meaningful_duration:
                    stats = object_stats[collision.object_label]
                    stats['count'] += 1
                    stats['total_duration'] += collision.duration
                    stats['avg_confidence'] += collision.collision_confidence
                    stats['avg_distance'] += collision.collision_distance
            
            # Ortalamaları hesapla
            for label, stats in object_stats.items():
                if stats['count'] > 0:
                    stats['avg_duration'] = stats['total_duration'] / stats['count']
                    stats['avg_confidence'] /= stats['count']
                    stats['avg_distance'] /= stats['count']
            
            return dict(object_stats)
            
        except Exception as e:
            self.logger.error(f"Nesne çakışma analizi hatası: {e}")
            return {}
    
    def _analyze_temporal_patterns(self, collisions: List[Collision3DEvent]) -> Dict[str, Any]:
        """Temporal desen analizi"""
        try:
            if not collisions:
                return {}
            
            # Süre dağılımı
            durations = [c.duration for c in collisions if c.duration >= self.min_meaningful_duration]
            
            if not durations:
                return {}
            
            return {
                'total_meaningful_collisions': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_attention_time': sum(durations),
                'collision_frequency': len(durations) / (self.analytics_window / 60)  # per minute
            }
            
        except Exception as e:
            self.logger.error(f"Temporal analiz hatası: {e}")
            return {}
    
    def _analyze_spatial_patterns(self, collisions: List[Collision3DEvent]) -> Dict[str, Any]:
        """Spatial desen analizi"""
        try:
            if not collisions:
                return {}
            
            # Mesafe dağılımı
            distances = [c.collision_distance for c in collisions if c.collision_distance < float('inf')]
            
            if not distances:
                return {}
            
            # Çakışma noktaları
            collision_points = [c.collision_point_3d for c in collisions if c.collision_point_3d]
            
            return {
                'avg_collision_distance': sum(distances) / len(distances),
                'min_collision_distance': min(distances),
                'max_collision_distance': max(distances),
                'collision_points_count': len(collision_points),
                'spatial_distribution': self._calculate_spatial_distribution(collision_points)
            }
            
        except Exception as e:
            self.logger.error(f"Spatial analiz hatası: {e}")
            return {}
    
    def _analyze_attention_patterns(self, collisions: List[Collision3DEvent]) -> Dict[str, Any]:
        """Dikkat deseni analizi"""
        try:
            if not collisions:
                return {}
            
            # En çok dikkat çeken objeler
            attention_by_object = defaultdict(float)
            
            for collision in collisions:
                if collision.duration >= self.min_meaningful_duration:
                    attention_by_object[collision.object_label] += collision.duration
            
            # Sırala
            sorted_attention = sorted(attention_by_object.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'most_attended_objects': sorted_attention[:5],
                'attention_distribution': dict(attention_by_object),
                'unique_objects_attended': len(attention_by_object)
            }
            
        except Exception as e:
            self.logger.error(f"Dikkat analizi hatası: {e}")
            return {}
    
    def _calculate_spatial_distribution(self, points: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Spatial dağılım hesapla"""
        try:
            if not points:
                return {}
            
            # Merkez hesapla
            center_x = sum(p[0] for p in points) / len(points)
            center_y = sum(p[1] for p in points) / len(points)
            center_z = sum(p[2] for p in points) / len(points)
            
            # Variance hesapla
            var_x = sum((p[0] - center_x)**2 for p in points) / len(points)
            var_y = sum((p[1] - center_y)**2 for p in points) / len(points)
            var_z = sum((p[2] - center_z)**2 for p in points) / len(points)
            
            return {
                'center': (center_x, center_y, center_z),
                'variance': (var_x, var_y, var_z),
                'spread': math.sqrt(var_x + var_y + var_z)
            }
            
        except Exception as e:
            self.logger.error(f"Spatial dağılım hesaplama hatası: {e}")
            return {}
    
    def _generate_collision_summary(self, collisions: List[Collision3DEvent]) -> Dict[str, Any]:
        """Çakışma özeti oluştur"""
        try:
            meaningful_collisions = [c for c in collisions if c.duration >= self.min_meaningful_duration]
            
            if not meaningful_collisions:
                return {'total_collisions': 0, 'meaningful_collisions': 0}
            
            # En uzun çakışma
            longest_collision = max(meaningful_collisions, key=lambda c: c.duration)
            
            # En yüksek confidence
            highest_confidence = max(meaningful_collisions, key=lambda c: c.collision_confidence)
            
            return {
                'total_collisions': len(collisions),
                'meaningful_collisions': len(meaningful_collisions),
                'longest_collision': {
                    'object_label': longest_collision.object_label,
                    'duration': longest_collision.duration,
                    'confidence': longest_collision.collision_confidence
                },
                'highest_confidence_collision': {
                    'object_label': highest_confidence.object_label,
                    'duration': highest_confidence.duration,
                    'confidence': highest_confidence.collision_confidence
                },
                'total_attention_time': sum(c.duration for c in meaningful_collisions)
            }
            
        except Exception as e:
            self.logger.error(f"Çakışma özeti oluşturma hatası: {e}")
            return {} 