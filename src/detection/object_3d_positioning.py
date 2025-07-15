"""
Objelerin 3D Uzayda Konumlandırılması Modülü
"""

import numpy as np
import cv2
import math
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque


@dataclass
class Object3DPosition:
    """3D uzayda konumlandırılmış obje verisi"""
    object_id: str
    label: str
    class_id: int
    confidence: float
    
    # 2D bilgileri
    bbox_2d: List[int]  # [x1, y1, x2, y2]
    center_2d: Tuple[int, int]
    
    # 3D konum bilgileri
    position_3d: Tuple[float, float, float]  # (x, y, z) kamera koordinatında
    dimensions_3d: Tuple[float, float, float]  # (width, height, depth) mm
    orientation_3d: Tuple[float, float, float]  # (rx, ry, rz) radyan
    
    # Gerçek dünya koordinatları
    world_position: Tuple[float, float, float]
    
    # Geometri bilgileri
    bounding_box_3d: List[Tuple[float, float, float]]  # 8 köşe noktası
    volume: float  # mm³
    
    # Güvenilirlik ve tracking
    position_confidence: float
    depth_confidence: float
    tracking_id: Optional[str] = None
    
    # Temporal bilgileri
    timestamp: float = field(default_factory=time.time)
    velocity_3d: Optional[Tuple[float, float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary formatına çevir"""
        return {
            'object_id': self.object_id,
            'label': self.label,
            'class_id': self.class_id,
            'confidence': self.confidence,
            'bbox_2d': self.bbox_2d,
            'center_2d': self.center_2d,
            'position_3d': self.position_3d,
            'dimensions_3d': self.dimensions_3d,
            'orientation_3d': self.orientation_3d,
            'world_position': self.world_position,
            'bounding_box_3d': self.bounding_box_3d,
            'volume': self.volume,
            'position_confidence': self.position_confidence,
            'depth_confidence': self.depth_confidence,
            'tracking_id': self.tracking_id,
            'timestamp': self.timestamp,
            'velocity_3d': self.velocity_3d
        }


class ObjectDimensionEstimator:
    """Obje boyutları tahmini"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Referans obje boyutları (mm cinsinden)
        self.reference_dimensions = {
            'person': {'width': 500, 'height': 1700, 'depth': 300},
            'laptop': {'width': 350, 'height': 250, 'depth': 25},
            'phone': {'width': 75, 'height': 150, 'depth': 8},
            'bottle': {'width': 70, 'height': 250, 'depth': 70},
            'cup': {'width': 80, 'height': 100, 'depth': 80},
            'book': {'width': 150, 'height': 200, 'depth': 20},
            'chair': {'width': 500, 'height': 800, 'depth': 500},
            'table': {'width': 1000, 'height': 750, 'depth': 600},
            'tv': {'width': 1200, 'height': 700, 'depth': 100},
            'monitor': {'width': 600, 'height': 400, 'depth': 50},
            'keyboard': {'width': 400, 'height': 150, 'depth': 30},
            'mouse': {'width': 120, 'height': 60, 'depth': 40},
            'car': {'width': 1800, 'height': 1500, 'depth': 4500},
            'bicycle': {'width': 600, 'height': 1100, 'depth': 1800},
            'backpack': {'width': 350, 'height': 450, 'depth': 200},
            'handbag': {'width': 300, 'height': 200, 'depth': 150},
            'suitcase': {'width': 400, 'height': 600, 'depth': 250},
            'umbrella': {'width': 100, 'height': 800, 'depth': 100},
            'baseball_bat': {'width': 70, 'height': 800, 'depth': 70},
            'tennis_racket': {'width': 300, 'height': 680, 'depth': 30},
            'frisbee': {'width': 270, 'height': 270, 'depth': 25},
            'skis': {'width': 100, 'height': 1700, 'depth': 15},
            'snowboard': {'width': 250, 'height': 1500, 'depth': 15},
            'sports_ball': {'width': 220, 'height': 220, 'depth': 220},
            'kite': {'width': 1000, 'height': 800, 'depth': 10},
            'baseball_glove': {'width': 250, 'height': 300, 'depth': 150},
            'skateboard': {'width': 200, 'height': 800, 'depth': 100},
            'surfboard': {'width': 600, 'height': 2100, 'depth': 80},
            'tennis_ball': {'width': 65, 'height': 65, 'depth': 65},
            'wine_glass': {'width': 75, 'height': 200, 'depth': 75},
            'fork': {'width': 20, 'height': 200, 'depth': 5},
            'knife': {'width': 20, 'height': 250, 'depth': 5},
            'spoon': {'width': 30, 'height': 180, 'depth': 8},
            'bowl': {'width': 150, 'height': 80, 'depth': 150},
            'banana': {'width': 30, 'height': 200, 'depth': 30},
            'apple': {'width': 80, 'height': 80, 'depth': 80},
            'orange': {'width': 75, 'height': 75, 'depth': 75},
            'broccoli': {'width': 120, 'height': 150, 'depth': 120},
            'carrot': {'width': 25, 'height': 200, 'depth': 25},
            'pizza': {'width': 300, 'height': 300, 'depth': 20},
            'donut': {'width': 100, 'height': 100, 'depth': 30},
            'cake': {'width': 200, 'height': 100, 'depth': 200},
            'couch': {'width': 2000, 'height': 900, 'depth': 1000},
            'bed': {'width': 1500, 'height': 500, 'depth': 2000},
            'dining_table': {'width': 1500, 'height': 750, 'depth': 900},
            'toilet': {'width': 400, 'height': 800, 'depth': 650},
            'sink': {'width': 600, 'height': 200, 'depth': 400},
            'refrigerator': {'width': 600, 'height': 1700, 'depth': 650},
            'microwave': {'width': 500, 'height': 300, 'depth': 400},
            'oven': {'width': 600, 'height': 600, 'depth': 600},
            'toaster': {'width': 250, 'height': 200, 'depth': 150},
            'clock': {'width': 300, 'height': 300, 'depth': 50},
            'vase': {'width': 100, 'height': 250, 'depth': 100},
            'scissors': {'width': 20, 'height': 200, 'depth': 10},
            'hair_dryer': {'width': 100, 'height': 250, 'depth': 200},
            'toothbrush': {'width': 15, 'height': 200, 'depth': 15}
        }
        
    def estimate_dimensions(self, bbox: List[int], label: str, depth_mm: float,
                           frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """Obje boyutlarını tahmin et"""
        try:
            if len(bbox) < 4:
                return (100.0, 100.0, 100.0)
            
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Referans boyutları al
            ref_dims = self.reference_dimensions.get(label, self.reference_dimensions['person'])
            
            # Kamera parametreleri
            h, w = frame_shape[:2]
            focal_length = min(w, h)  # Basit tahmin
            
            # Gerçek boyutları hesapla
            real_width = (bbox_width * depth_mm) / focal_length
            real_height = (bbox_height * depth_mm) / focal_length
            
            # Derinlik tahmini (aspect ratio'ya göre)
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
            ref_aspect = ref_dims['width'] / ref_dims['height'] if ref_dims['height'] > 0 else 1.0
            
            # Referans boyutlarına göre scaling
            scale_factor = min(real_width / ref_dims['width'], real_height / ref_dims['height'])
            
            estimated_depth = ref_dims['depth'] * scale_factor
            
            # Sınırlar içinde tut
            real_width = max(10, min(5000, real_width))
            real_height = max(10, min(5000, real_height))
            estimated_depth = max(5, min(2000, estimated_depth))
            
            return (real_width, real_height, estimated_depth)
            
        except Exception as e:
            self.logger.error(f"Boyut tahmini hatası: {e}")
            return (100.0, 100.0, 100.0)


class Depth3DEstimator:
    """Gelişmiş 3D derinlik tahmini"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Derinlik parametreleri
        self.depth_range = config.get('depth_range', (100, 5000))  # mm
        self.baseline_depth = config.get('baseline_depth', 1000)  # mm
        
        # Kamera kalibrasyonu
        self.focal_length = config.get('focal_length', 800)
        self.camera_height = config.get('camera_height', 1200)  # mm
        
        # Dimension estimator
        self.dimension_estimator = ObjectDimensionEstimator(config)
        
    def estimate_depth_multiple_methods(self, bbox: List[int], label: str, 
                                       frame_shape: Tuple[int, int],
                                       additional_cues: Dict[str, Any] = None) -> Tuple[float, float]:
        """Çoklu yöntemle derinlik tahmini"""
        try:
            depth_estimates = []
            confidences = []
            
            # Yöntem 1: Obje boyutu bazlı
            size_depth, size_confidence = self._estimate_depth_from_size(bbox, label, frame_shape)
            depth_estimates.append(size_depth)
            confidences.append(size_confidence)
            
            # Yöntem 2: Yer düzlemi bazlı
            ground_depth, ground_confidence = self._estimate_depth_from_ground_plane(bbox, frame_shape)
            depth_estimates.append(ground_depth)
            confidences.append(ground_confidence)
            
            # Yöntem 3: Perspektif bazlı
            perspective_depth, perspective_confidence = self._estimate_depth_from_perspective(bbox, frame_shape)
            depth_estimates.append(perspective_depth)
            confidences.append(perspective_confidence)
            
            # Yöntem 4: Bağlam bazlı (eğer mevcut)
            if additional_cues and 'nearby_objects' in additional_cues:
                context_depth, context_confidence = self._estimate_depth_from_context(
                    bbox, additional_cues['nearby_objects']
                )
                depth_estimates.append(context_depth)
                confidences.append(context_confidence)
            
            # Ağırlıklı ortalama
            if len(depth_estimates) > 0:
                total_confidence = sum(confidences)
                if total_confidence > 0:
                    weighted_depth = sum(d * c for d, c in zip(depth_estimates, confidences)) / total_confidence
                    final_confidence = total_confidence / len(confidences)
                else:
                    weighted_depth = sum(depth_estimates) / len(depth_estimates)
                    final_confidence = 0.5
            else:
                weighted_depth = self.baseline_depth
                final_confidence = 0.3
            
            # Sınırlar içinde tut
            weighted_depth = max(self.depth_range[0], min(self.depth_range[1], weighted_depth))
            
            return weighted_depth, final_confidence
            
        except Exception as e:
            self.logger.error(f"Çoklu derinlik tahmini hatası: {e}")
            return self.baseline_depth, 0.3
    
    def _estimate_depth_from_size(self, bbox: List[int], label: str, 
                                 frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Obje boyutundan derinlik tahmini"""
        try:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Referans boyutları
            ref_dims = self.dimension_estimator.reference_dimensions.get(
                label, self.dimension_estimator.reference_dimensions['person']
            )
            
            # Derinlik hesaplama
            depth_from_width = (ref_dims['width'] * self.focal_length) / bbox_width if bbox_width > 0 else self.baseline_depth
            depth_from_height = (ref_dims['height'] * self.focal_length) / bbox_height if bbox_height > 0 else self.baseline_depth
            
            # Ortalama
            estimated_depth = (depth_from_width + depth_from_height) / 2
            
            # Güvenilirlik
            confidence = 0.7 if bbox_width > 20 and bbox_height > 20 else 0.4
            
            return estimated_depth, confidence
            
        except Exception as e:
            self.logger.error(f"Boyut bazlı derinlik tahmini hatası: {e}")
            return self.baseline_depth, 0.3
    
    def _estimate_depth_from_ground_plane(self, bbox: List[int], 
                                        frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Yer düzlemi bazlı derinlik tahmini"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame_shape[:2]
            
            # Objenin alt noktası
            bottom_y = y2
            
            # Vanishing point tahmini (frame ortasında)
            vanishing_point_y = h * 0.4  # Horizon line tahmini
            
            # Yer düzlemi derinliği
            if bottom_y > vanishing_point_y:
                # Kamera altında
                y_distance = bottom_y - vanishing_point_y
                depth = self.camera_height * self.focal_length / y_distance
                confidence = 0.6
            else:
                # Kamera üstünde (nadiren)
                depth = self.baseline_depth
                confidence = 0.3
            
            return depth, confidence
            
        except Exception as e:
            self.logger.error(f"Yer düzlemi derinlik tahmini hatası: {e}")
            return self.baseline_depth, 0.3
    
    def _estimate_depth_from_perspective(self, bbox: List[int], 
                                       frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Perspektif bazlı derinlik tahmini"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame_shape[:2]
            
            # Obje merkezi
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Merkeze uzaklık
            distance_from_center = math.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
            normalized_distance = distance_from_center / max(w, h)
            
            # Perspektif etkisi
            perspective_factor = 1.0 + normalized_distance * 0.3
            estimated_depth = self.baseline_depth * perspective_factor
            
            confidence = 0.4
            
            return estimated_depth, confidence
            
        except Exception as e:
            self.logger.error(f"Perspektif derinlik tahmini hatası: {e}")
            return self.baseline_depth, 0.3
    
    def _estimate_depth_from_context(self, bbox: List[int], 
                                   nearby_objects: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Bağlam bazlı derinlik tahmini"""
        try:
            if not nearby_objects:
                return self.baseline_depth, 0.2
            
            # Yakındaki objelerin derinliklerini kullan
            context_depths = []
            for obj in nearby_objects:
                if 'depth' in obj:
                    context_depths.append(obj['depth'])
            
            if context_depths:
                avg_context_depth = sum(context_depths) / len(context_depths)
                confidence = 0.5
                return avg_context_depth, confidence
            
            return self.baseline_depth, 0.2
            
        except Exception as e:
            self.logger.error(f"Bağlam derinlik tahmini hatası: {e}")
            return self.baseline_depth, 0.2


class Object3DPositioner:
    """Ana 3D obje konumlandırma sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alt modüller
        self.depth_estimator = Depth3DEstimator(config.get('depth_estimation', {}))
        self.dimension_estimator = ObjectDimensionEstimator(config.get('dimension_estimation', {}))
        
        # Kamera kalibrasyonu
        self.focal_length = config.get('focal_length', 800)
        self.principal_point = config.get('principal_point', (640, 360))
        self.camera_matrix = self._setup_camera_matrix()
        
        # Tracking
        self.tracked_objects = {}  # tracking_id -> Object3DPosition
        self.object_history = deque(maxlen=config.get('max_history', 100))
        
        # World coordinate system
        self.world_origin = config.get('world_origin', (0, 0, 0))
        self.world_scale = config.get('world_scale', 1.0)
        
        self.logger.info("Object3DPositioner başlatıldı")
    
    def _setup_camera_matrix(self) -> np.ndarray:
        """Kamera matrisini ayarla"""
        return np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def position_objects_3d(self, detections: List[Dict[str, Any]], 
                           frame_shape: Tuple[int, int],
                           spatial_context: Dict[str, Any] = None) -> List[Object3DPosition]:
        """Objeleri 3D uzayda konumlandır"""
        try:
            positioned_objects = []
            current_time = time.time()
            
            for detection in detections:
                try:
                    # Temel bilgileri al
                    bbox = detection.get('bbox', [])
                    label = detection.get('label', 'unknown')
                    class_id = detection.get('class_id', 0)
                    confidence = detection.get('confidence', 0.0)
                    
                    if len(bbox) < 4:
                        continue
                    
                    # 2D merkez hesapla
                    x1, y1, x2, y2 = bbox
                    center_2d = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # 3D derinlik tahmini
                    depth_mm, depth_confidence = self.depth_estimator.estimate_depth_multiple_methods(
                        bbox, label, frame_shape, spatial_context
                    )
                    
                    # 3D pozisyon hesapla
                    position_3d = self._calculate_3d_position(center_2d, depth_mm, frame_shape)
                    
                    # 3D boyutları tahmin et
                    dimensions_3d = self.dimension_estimator.estimate_dimensions(
                        bbox, label, depth_mm, frame_shape
                    )
                    
                    # Obje yönelimi (basit yaklaşım)
                    orientation_3d = self._estimate_object_orientation(bbox, label)
                    
                    # Gerçek dünya koordinatları
                    world_position = self._camera_to_world_coordinates(position_3d)
                    
                    # 3D bounding box hesapla
                    bounding_box_3d = self._calculate_3d_bounding_box(
                        position_3d, dimensions_3d, orientation_3d
                    )
                    
                    # Hacim hesapla
                    volume = dimensions_3d[0] * dimensions_3d[1] * dimensions_3d[2]
                    
                    # Tracking ID
                    tracking_id = self._assign_tracking_id(position_3d, label, current_time)
                    
                    # Hız hesapla (eğer tracking mevcut)
                    velocity_3d = self._calculate_velocity(tracking_id, position_3d, current_time)
                    
                    # Obje ID oluştur
                    object_id = f"{label}_{center_2d[0]}_{center_2d[1]}_{int(current_time * 1000) % 10000}"
                    
                    # Object3DPosition oluştur
                    positioned_object = Object3DPosition(
                        object_id=object_id,
                        label=label,
                        class_id=class_id,
                        confidence=confidence,
                        bbox_2d=bbox,
                        center_2d=center_2d,
                        position_3d=position_3d,
                        dimensions_3d=dimensions_3d,
                        orientation_3d=orientation_3d,
                        world_position=world_position,
                        bounding_box_3d=bounding_box_3d,
                        volume=volume,
                        position_confidence=min(confidence, depth_confidence),
                        depth_confidence=depth_confidence,
                        tracking_id=tracking_id,
                        timestamp=current_time,
                        velocity_3d=velocity_3d
                    )
                    
                    positioned_objects.append(positioned_object)
                    
                except Exception as e:
                    self.logger.error(f"Obje konumlandırma hatası: {e}")
                    continue
            
            # Geçmişe kaydet
            self.object_history.append({
                'timestamp': current_time,
                'objects': positioned_objects
            })
            
            return positioned_objects
            
        except Exception as e:
            self.logger.error(f"3D obje konumlandırma hatası: {e}")
            return []
    
    def _calculate_3d_position(self, center_2d: Tuple[int, int], depth_mm: float, 
                              frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """2D merkez ve derinlikten 3D pozisyon hesapla"""
        try:
            x_px, y_px = center_2d
            
            # Kamera koordinatlarına çevir
            x_cam = (x_px - self.principal_point[0]) * depth_mm / self.focal_length
            y_cam = (y_px - self.principal_point[1]) * depth_mm / self.focal_length
            z_cam = depth_mm
            
            return (x_cam, y_cam, z_cam)
            
        except Exception as e:
            self.logger.error(f"3D pozisyon hesaplama hatası: {e}")
            return (0, 0, depth_mm)
    
    def _estimate_object_orientation(self, bbox: List[int], label: str) -> Tuple[float, float, float]:
        """Obje yönelimini tahmin et"""
        try:
            # Basit yaklaşım - gelecekte geliştirilecek
            x1, y1, x2, y2 = bbox
            
            # Aspect ratio'dan basit yönelim tahmini
            aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
            
            # Varsayılan yönelim
            rx = 0.0  # Pitch
            ry = 0.0  # Yaw
            rz = 0.0  # Roll
            
            # Bazı objeler için özel durumlar
            if label in ['laptop', 'book', 'tablet']:
                rx = -0.2  # Hafif eğim
            elif label in ['bottle', 'cup']:
                rx = 0.0  # Dik
            elif label in ['car', 'truck']:
                ry = 0.0  # Düz
            
            return (rx, ry, rz)
            
        except Exception as e:
            self.logger.error(f"Obje yönelimi tahmini hatası: {e}")
            return (0.0, 0.0, 0.0)
    
    def _camera_to_world_coordinates(self, camera_position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Kamera koordinatlarını dünya koordinatlarına çevir"""
        try:
            # Basit transformasyon
            world_x = camera_position[0] * self.world_scale + self.world_origin[0]
            world_y = camera_position[1] * self.world_scale + self.world_origin[1]
            world_z = camera_position[2] * self.world_scale + self.world_origin[2]
            
            return (world_x, world_y, world_z)
            
        except Exception as e:
            self.logger.error(f"Dünya koordinatı dönüşümü hatası: {e}")
            return camera_position
    
    def _calculate_3d_bounding_box(self, position: Tuple[float, float, float], 
                                  dimensions: Tuple[float, float, float], 
                                  orientation: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """3D bounding box köşe noktalarını hesapla"""
        try:
            x, y, z = position
            w, h, d = dimensions
            
            # Basit bounding box (orientasyon şimdilik ihmal ediliyor)
            corners = [
                (x - w/2, y - h/2, z - d/2),  # 0: left-bottom-front
                (x + w/2, y - h/2, z - d/2),  # 1: right-bottom-front
                (x + w/2, y + h/2, z - d/2),  # 2: right-top-front
                (x - w/2, y + h/2, z - d/2),  # 3: left-top-front
                (x - w/2, y - h/2, z + d/2),  # 4: left-bottom-back
                (x + w/2, y - h/2, z + d/2),  # 5: right-bottom-back
                (x + w/2, y + h/2, z + d/2),  # 6: right-top-back
                (x - w/2, y + h/2, z + d/2),  # 7: left-top-back
            ]
            
            # Gelecekte rotasyon uygulanabilir
            # TODO: Orientation kullanarak köşeleri döndür
            
            return corners
            
        except Exception as e:
            self.logger.error(f"3D bounding box hesaplama hatası: {e}")
            return []
    
    def _assign_tracking_id(self, position_3d: Tuple[float, float, float], 
                           label: str, timestamp: float) -> Optional[str]:
        """Tracking ID ataması"""
        try:
            # Basit tracking - pozisyon yakınlığı
            min_distance = float('inf')
            best_tracking_id = None
            
            for tracking_id, tracked_obj in self.tracked_objects.items():
                if tracked_obj.label == label:
                    distance = math.sqrt(
                        (position_3d[0] - tracked_obj.position_3d[0])**2 +
                        (position_3d[1] - tracked_obj.position_3d[1])**2 +
                        (position_3d[2] - tracked_obj.position_3d[2])**2
                    )
                    
                    if distance < min_distance and distance < 200:  # 200mm threshold
                        min_distance = distance
                        best_tracking_id = tracking_id
            
            # Yeni tracking ID oluştur
            if best_tracking_id is None:
                best_tracking_id = f"track_{label}_{int(timestamp * 1000) % 100000}"
            
            return best_tracking_id
            
        except Exception as e:
            self.logger.error(f"Tracking ID ataması hatası: {e}")
            return None
    
    def _calculate_velocity(self, tracking_id: Optional[str], 
                           current_position: Tuple[float, float, float], 
                           timestamp: float) -> Optional[Tuple[float, float, float]]:
        """Obje hızını hesapla"""
        try:
            if tracking_id is None or tracking_id not in self.tracked_objects:
                return None
            
            previous_obj = self.tracked_objects[tracking_id]
            dt = timestamp - previous_obj.timestamp
            
            if dt > 0:
                vx = (current_position[0] - previous_obj.position_3d[0]) / dt
                vy = (current_position[1] - previous_obj.position_3d[1]) / dt
                vz = (current_position[2] - previous_obj.position_3d[2]) / dt
                
                return (vx, vy, vz)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Hız hesaplama hatası: {e}")
            return None
    
    def get_spatial_relationships(self, positioned_objects: List[Object3DPosition]) -> Dict[str, Any]:
        """Objelerin uzaysal ilişkilerini analiz et"""
        try:
            relationships = {
                'object_distances': {},
                'spatial_clusters': [],
                'nearest_neighbors': {},
                'spatial_density': {}
            }
            
            # Obje-obje uzaklıkları
            for i, obj1 in enumerate(positioned_objects):
                for j, obj2 in enumerate(positioned_objects[i+1:], i+1):
                    distance = math.sqrt(
                        (obj1.position_3d[0] - obj2.position_3d[0])**2 +
                        (obj1.position_3d[1] - obj2.position_3d[1])**2 +
                        (obj1.position_3d[2] - obj2.position_3d[2])**2
                    )
                    
                    relationships['object_distances'][f"{obj1.object_id}-{obj2.object_id}"] = distance
            
            # En yakın komşular
            for obj in positioned_objects:
                nearest = []
                for other_obj in positioned_objects:
                    if obj.object_id != other_obj.object_id:
                        distance = math.sqrt(
                            (obj.position_3d[0] - other_obj.position_3d[0])**2 +
                            (obj.position_3d[1] - other_obj.position_3d[1])**2 +
                            (obj.position_3d[2] - other_obj.position_3d[2])**2
                        )
                        nearest.append((other_obj.object_id, other_obj.label, distance))
                
                # En yakın 3 obje
                nearest.sort(key=lambda x: x[2])
                relationships['nearest_neighbors'][obj.object_id] = nearest[:3]
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Uzaysal ilişki analizi hatası: {e}")
            return {}
    
    def update_object_tracking(self, positioned_objects: List[Object3DPosition]):
        """Obje tracking'i güncelle"""
        try:
            # Tracked objects güncelle
            for obj in positioned_objects:
                if obj.tracking_id:
                    self.tracked_objects[obj.tracking_id] = obj
            
            # Eski tracking'leri temizle
            current_time = time.time()
            expired_ids = []
            
            for tracking_id, tracked_obj in self.tracked_objects.items():
                if current_time - tracked_obj.timestamp > 5.0:  # 5 saniye timeout
                    expired_ids.append(tracking_id)
            
            for tracking_id in expired_ids:
                del self.tracked_objects[tracking_id]
                
        except Exception as e:
            self.logger.error(f"Obje tracking güncelleme hatası: {e}") 