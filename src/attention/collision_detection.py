"""
Çakışma Tespiti Algoritmaları
"""

import math
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from .gaze_3d_types import GazePoint3D, GazeCollisionEvent


class CollisionDetector:
    """Çakışma tespiti sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collision_threshold = config.get('collision_threshold', 40.0)
        self.logger = logging.getLogger(__name__)
        
    def check_object_collision(self, gaze_3d: GazePoint3D, obj_data: Dict[str, Any],
                              frame_shape: Tuple[int, int]) -> Tuple[bool, float]:
        """Gelişmiş çakışma kontrolü"""
        try:
            bbox = obj_data.get('bbox', [])
            if len(bbox) != 4:
                return False, float('inf')
            
            # Bounding box genişlet (3D için)
            x1, y1, x2, y2 = bbox
            expanded_bbox = [
                x1 - self.collision_threshold,
                y1 - self.collision_threshold,
                x2 + self.collision_threshold,
                y2 + self.collision_threshold
            ]
            
            # 3D noktayı 2D'ye projekte et
            gaze_2d = gaze_3d.to_2d(frame_shape)
            
            # Çakışma kontrolü
            is_inside = (expanded_bbox[0] <= gaze_2d[0] <= expanded_bbox[2] and
                        expanded_bbox[1] <= gaze_2d[1] <= expanded_bbox[3])
            
            if is_inside:
                # Obje merkezine uzaklık hesapla
                obj_center_x = (x1 + x2) / 2
                obj_center_y = (y1 + y2) / 2
                
                distance = math.sqrt((gaze_2d[0] - obj_center_x)**2 + (gaze_2d[1] - obj_center_y)**2)
                
                return True, distance
            
            return False, float('inf')
            
        except Exception as e:
            self.logger.error(f"Çakışma kontrolü hatası: {e}")
            return False, float('inf')
    
    def check_multiple_collisions(self, gaze_3d: GazePoint3D, detected_objects: List[Dict[str, Any]],
                                 frame_shape: Tuple[int, int]) -> List[Tuple[str, Dict[str, Any], float]]:
        """Birden fazla obje ile çakışma kontrolü"""
        collisions = []
        
        for obj_data in detected_objects:
            obj_id = self._generate_object_id(obj_data)
            is_collision, distance = self.check_object_collision(gaze_3d, obj_data, frame_shape)
            
            if is_collision:
                collisions.append((obj_id, obj_data, distance))
        
        # Uzaklığa göre sırala (en yakın önce)
        collisions.sort(key=lambda x: x[2])
        
        return collisions
    
    def _generate_object_id(self, obj_data: Dict[str, Any]) -> str:
        """Obje için benzersiz ID oluştur"""
        label = obj_data.get('label', 'unknown')
        bbox = obj_data.get('bbox', [0, 0, 0, 0])
        
        # Bbox merkezi
        if len(bbox) >= 4:
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
        else:
            center_x, center_y = 0, 0
            
        return f"{label}_{center_x}_{center_y}"


class CollisionManager:
    """Çakışma yönetimi sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collision_detector = CollisionDetector(config)
        self.active_collisions = {}  # object_id -> GazeCollisionEvent
        self.collision_history = []
        self.max_history_size = config.get('max_collision_history', 200)
        self.collision_timeout = config.get('collision_timeout', 15.0)
        self.logger = logging.getLogger(__name__)
    
    def process_collisions(self, gaze_3d: GazePoint3D, detected_objects: List[Dict[str, Any]],
                          frame_shape: Tuple[int, int]) -> List[GazeCollisionEvent]:
        """Çakışmaları işle"""
        current_time = time.time()
        collision_events = []
        
        # Mevcut çakışmaları güncelle
        self._update_active_collisions(current_time)
        
        if gaze_3d is None:
            return collision_events
        
        # Çakışma tespiti
        collisions = self.collision_detector.check_multiple_collisions(
            gaze_3d, detected_objects, frame_shape
        )
        
        # Mevcut çakışmaları kontrol et
        current_collision_ids = set()
        
        for obj_id, obj_data, distance in collisions:
            current_collision_ids.add(obj_id)
            
            if obj_id not in self.active_collisions:
                # Yeni çakışma
                collision_event = GazeCollisionEvent(
                    gaze_point=gaze_3d.to_2d(frame_shape),
                    object_data=obj_data,
                    collision_depth=gaze_3d.z,
                    collision_timestamp=current_time
                )
                
                self.active_collisions[obj_id] = collision_event
                collision_events.append(collision_event)
                
                # Log çakışma başlangıcı
                self._log_collision_start(collision_event, obj_data)
                
            else:
                # Mevcut çakışmayı güncelle
                existing_collision = self.active_collisions[obj_id]
                existing_collision.update_duration(current_time)
                collision_events.append(existing_collision)
        
        # Bitmiş çakışmaları tespit et
        finished_collisions = set(self.active_collisions.keys()) - current_collision_ids
        
        for obj_id in finished_collisions:
            collision_event = self.active_collisions[obj_id]
            collision_event.end_collision(current_time)
            self._add_to_history(collision_event)
            del self.active_collisions[obj_id]
            
            # Log çakışma bitişi
            self._log_collision_end(collision_event)
        
        return collision_events
    
    def _update_active_collisions(self, current_time: float):
        """Aktif çakışmaları güncelle"""
        to_remove = []
        
        for obj_id, collision_event in self.active_collisions.items():
            collision_event.update_duration(current_time)
            
            # Çok eski çakışmaları temizle
            if collision_event.duration > self.collision_timeout:
                collision_event.end_collision(current_time)
                self._add_to_history(collision_event)
                to_remove.append(obj_id)
        
        # Eski çakışmaları sil
        for obj_id in to_remove:
            del self.active_collisions[obj_id]
    
    def _add_to_history(self, collision_event: GazeCollisionEvent):
        """Çakışmayı geçmişe ekle"""
        self.collision_history.append(collision_event)
        
        # History boyutunu kontrol et
        if len(self.collision_history) > self.max_history_size:
            self.collision_history = self.collision_history[-self.max_history_size:]
    
    def _log_collision_start(self, collision_event: GazeCollisionEvent, obj_data: Dict[str, Any]):
        """Çakışma başlangıcını logla"""
        obj_label = obj_data.get('label', 'Unknown')
        confidence = obj_data.get('confidence', 0.0)
        
        self.logger.info(f"🎯 YENİ ÇAKIŞMA: {obj_label} (güven: {confidence:.2f}) - "
                        f"Derinlik: {collision_event.collision_depth:.1f}mm")
    
    def _log_collision_end(self, collision_event: GazeCollisionEvent):
        """Çakışma bitişini logla"""
        obj_label = collision_event.object_data.get('label', 'Unknown')
        
        self.logger.info(f"🎯 ÇAKIŞMA BİTTİ: {obj_label} - "
                        f"Toplam süre: {collision_event.duration:.2f}s")
    
    def get_collision_summary(self) -> Dict[str, Any]:
        """Çakışma özeti"""
        current_time = time.time()
        
        # Aktif çakışmaları güncelle
        for collision_event in self.active_collisions.values():
            collision_event.update_duration(current_time)
        
        # En uzun çakışma
        all_collisions = list(self.active_collisions.values()) + self.collision_history
        longest_collision = max(all_collisions, key=lambda x: x.duration, default=None)
        
        # Çakışma kategorileri
        collision_categories = defaultdict(int)
        total_duration = 0.0
        
        for collision in self.collision_history:
            category = collision.object_data.get('label', 'unknown')
            collision_categories[category] += 1
            total_duration += collision.duration
        
        # En çok çakışan objeler
        most_collided = self._get_most_collided_objects()
        
        return {
            'active_collisions': len(self.active_collisions),
            'total_collision_events': len(self.collision_history) + len(self.active_collisions),
            'collision_events': list(self.active_collisions.values()),
            'recent_collisions': self.collision_history[-20:],  # Son 20 çakışma
            'average_collision_duration': total_duration / max(len(self.collision_history), 1),
            'longest_collision_duration': longest_collision.duration if longest_collision else 0.0,
            'collision_categories': dict(collision_categories),
            'most_collided_objects': most_collided
        }
    
    def _get_most_collided_objects(self) -> List[Dict[str, Any]]:
        """En çok çakışan objeleri döndür"""
        object_counts = defaultdict(lambda: {'count': 0, 'total_duration': 0.0})
        
        for event in self.collision_history:
            obj_label = event.object_data.get('label', 'unknown')
            object_counts[obj_label]['count'] += 1
            object_counts[obj_label]['total_duration'] += event.duration
        
        # En çok çakışan 10 obje
        most_collided = sorted(
            object_counts.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:10]
        
        return [
            {
                'object': obj, 
                'collision_count': data['count'],
                'total_duration': data['total_duration'],
                'avg_duration': data['total_duration'] / data['count']
            } 
            for obj, data in most_collided
        ]
    
    def reset_collision_data(self):
        """Çakışma verilerini sıfırla"""
        self.active_collisions.clear()
        self.collision_history.clear()
        self.logger.info("Çakışma verileri sıfırlandı")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Performans istatistikleri"""
        return {
            'active_collisions': len(self.active_collisions),
            'collision_history_size': len(self.collision_history),
            'collision_threshold': self.collision_threshold,
            'collision_timeout': self.collision_timeout,
            'max_history_size': self.max_history_size
        } 