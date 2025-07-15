"""
Çakışma Analytics Modülü
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from .base_logger import BaseFirestoreLogger, convert_numpy_types


class CollisionAnalytics(BaseFirestoreLogger):
    """Çakışma analizi için özel analytics"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.collision_stats = {
            'total_collisions': 0,
            'active_collisions': 0,
            'average_duration': 0.0,
            'collision_types': {},
            'most_collided_objects': []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def log_collision_event(self, collision_event):
        """3D Gaze çakışma olayını kaydet"""
        if not self.db:
            return
        
        try:
            collision_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'collision_timestamp': collision_event.collision_timestamp,
                'duration': collision_event.duration,
                'is_active': collision_event.is_active,
                'gaze_point': {
                    'x': collision_event.gaze_point[0],
                    'y': collision_event.gaze_point[1]
                },
                'collision_depth': collision_event.collision_depth,
                'object_data': convert_numpy_types({
                    'label': collision_event.object_data.get('label', 'unknown'),
                    'category': collision_event.object_data.get('category', 'unknown'),
                    'confidence': collision_event.object_data.get('confidence', 0.0),
                    'bbox': collision_event.object_data.get('bbox', [0, 0, 0, 0]),
                    'scene_context': collision_event.object_data.get('scene_context', 'unknown')
                }),
                'collision_type': 'gaze_3d_collision'
            }
            
            task = {
                'collection': 'collision_events',
                'data': collision_data,
                'operation': 'add'
            }
            self.write_queue.put(task)
            
            # İstatistikleri güncelle
            self._update_collision_stats(collision_event)
            
            # Kısa süreli cache'e de ekle
            self.detection_cache.append(collision_data)
            
        except Exception as e:
            self.logger.error(f"Collision event logging hatası: {e}")
    
    def log_collision_summary(self, collision_summary: Dict[str, Any]):
        """Çakışma özeti kaydet"""
        if not self.db:
            return
        
        try:
            summary_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'active_collisions': collision_summary.get('active_collisions', 0),
                'total_collision_events': collision_summary.get('total_collision_events', 0),
                'average_collision_duration': collision_summary.get('average_collision_duration', 0.0),
                'most_collided_objects': convert_numpy_types(collision_summary.get('most_collided_objects', [])),
                'collision_stats': {
                    'total_events': len(collision_summary.get('recent_collisions', [])),
                    'active_events': len(collision_summary.get('collision_events', [])),
                    'avg_duration': collision_summary.get('average_collision_duration', 0.0)
                }
            }
            
            task = {
                'collection': 'collision_summaries',
                'data': summary_data,
                'operation': 'add'
            }
            self.write_queue.put(task)
            
        except Exception as e:
            self.logger.error(f"Collision summary logging hatası: {e}")
    
    def log_collision_pattern(self, pattern_data: Dict[str, Any]):
        """Çakışma paterni kaydet"""
        if not self.db:
            return
        
        try:
            pattern_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'pattern_type': pattern_data.get('pattern_type', 'unknown'),
                'object_sequence': pattern_data.get('object_sequence', []),
                'duration_pattern': pattern_data.get('duration_pattern', []),
                'frequency': pattern_data.get('frequency', 0),
                'confidence': pattern_data.get('confidence', 0.0),
                'spatial_pattern': pattern_data.get('spatial_pattern', {}),
                'temporal_pattern': pattern_data.get('temporal_pattern', {})
            }
            
            task = {
                'collection': 'collision_patterns',
                'data': convert_numpy_types(pattern_log),
                'operation': 'add'
            }
            self.write_queue.put(task)
            
        except Exception as e:
            self.logger.error(f"Collision pattern logging hatası: {e}")
    
    def log_collision_heatmap(self, heatmap_data: Dict[str, Any]):
        """Çakışma heatmap verisi kaydet"""
        if not self.db:
            return
        
        try:
            heatmap_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'resolution': heatmap_data.get('resolution', [0, 0]),
                'collision_density': heatmap_data.get('collision_density', []),
                'hotspots': heatmap_data.get('hotspots', []),
                'time_window': heatmap_data.get('time_window', 0),
                'total_samples': heatmap_data.get('total_samples', 0),
                'max_density': heatmap_data.get('max_density', 0.0)
            }
            
            task = {
                'collection': 'collision_heatmaps',
                'data': convert_numpy_types(heatmap_log),
                'operation': 'add'
            }
            self.write_queue.put(task)
            
        except Exception as e:
            self.logger.error(f"Collision heatmap logging hatası: {e}")
    
    def log_collision_metrics(self, metrics: Dict[str, Any]):
        """Çakışma metrikleri kaydet"""
        if not self.db:
            return
        
        try:
            metrics_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'collision_rate': metrics.get('collision_rate', 0.0),
                'average_duration': metrics.get('average_duration', 0.0),
                'collision_accuracy': metrics.get('collision_accuracy', 0.0),
                'false_positive_rate': metrics.get('false_positive_rate', 0.0),
                'object_distribution': metrics.get('object_distribution', {}),
                'depth_distribution': metrics.get('depth_distribution', {}),
                'time_distribution': metrics.get('time_distribution', {})
            }
            
            task = {
                'collection': 'collision_metrics',
                'data': convert_numpy_types(metrics_log),
                'operation': 'add'
            }
            self.write_queue.put(task)
            
        except Exception as e:
            self.logger.error(f"Collision metrics logging hatası: {e}")
    
    def _update_collision_stats(self, collision_event):
        """Çakışma istatistiklerini güncelle"""
        try:
            # Toplam çakışma sayısı
            self.collision_stats['total_collisions'] += 1
            
            # Aktif çakışmalar
            if collision_event.is_active:
                self.collision_stats['active_collisions'] += 1
            
            # Ortalama süre hesaplama
            if collision_event.duration > 0:
                current_avg = self.collision_stats['average_duration']
                total_count = self.collision_stats['total_collisions']
                
                new_avg = ((current_avg * (total_count - 1)) + collision_event.duration) / total_count
                self.collision_stats['average_duration'] = new_avg
            
            # Obje türü istatistikleri
            obj_label = collision_event.object_data.get('label', 'unknown')
            if obj_label not in self.collision_stats['collision_types']:
                self.collision_stats['collision_types'][obj_label] = 0
            self.collision_stats['collision_types'][obj_label] += 1
            
        except Exception as e:
            self.logger.error(f"Collision stats update hatası: {e}")
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """Çakışma istatistiklerini döndür"""
        return {
            'session_stats': self.collision_stats.copy(),
            'session_id': self.session_id,
            'session_duration': (datetime.now(timezone.utc) - self.session_start).total_seconds(),
            'collision_rate': self.collision_stats['total_collisions'] / max(1, self.session_metrics.total_frames),
            'cache_size': len(self.detection_cache)
        }
    
    def reset_collision_stats(self):
        """Çakışma istatistiklerini sıfırla"""
        self.collision_stats = {
            'total_collisions': 0,
            'active_collisions': 0,
            'average_duration': 0.0,
            'collision_types': {},
            'most_collided_objects': []
        }
        self.logger.info("Çakışma istatistikleri sıfırlandı")
    
    def export_collision_data(self, format_type: str = 'json') -> Optional[str]:
        """Çakışma verilerini export et"""
        try:
            export_data = {
                'session_id': self.session_id,
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'statistics': self.collision_stats,
                'cache_data': self.detection_cache[-100:],  # Son 100 kayıt
                'session_summary': self.get_session_summary()
            }
            
            if format_type == 'json':
                import json
                return json.dumps(export_data, indent=2, default=str)
            else:
                return str(export_data)
                
        except Exception as e:
            self.logger.error(f"Collision data export hatası: {e}")
            return None 