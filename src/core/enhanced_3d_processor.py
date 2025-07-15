"""
Gelişmiş 3D İşleme Orchestrator Modülü
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional

# Yeni modüller
from spatial.spatial_environment_analyzer import SpatialEnvironmentAnalyzer
from attention.enhanced_head_pose import Enhanced3DHeadPoseEstimator, Enhanced3DGazeEstimator
from detection.object_3d_positioning import Object3DPositioner
from attention.true_3d_collision_detection import True3DCollisionDetector
from analytics.enhanced_firestore_manager import Enhanced3DFirestoreManager


class Enhanced3DProcessor:
    """Ana 3D işleme orchestrator sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alt modüller
        self.spatial_analyzer = SpatialEnvironmentAnalyzer(config.get('spatial_analysis', {}))
        self.head_pose_estimator = Enhanced3DHeadPoseEstimator(config.get('head_pose_estimation', {}))
        self.gaze_estimator = Enhanced3DGazeEstimator(config.get('gaze_estimation', {}))
        self.object_positioner = Object3DPositioner(config.get('object_positioning', {}))
        self.collision_detector = True3DCollisionDetector(config.get('collision_detection', {}))
        self.firestore_manager = Enhanced3DFirestoreManager(config.get('firestore_analytics', {}))
        
        # Performance tracking
        self.processing_times = {
            'spatial_analysis': [],
            'head_pose_estimation': [],
            'gaze_estimation': [],
            'object_positioning': [],
            'collision_detection': [],
            'firestore_logging': []
        }
        
        # Frame counter
        self.frame_count = 0
        
        # Session management
        self.session_active = True
        self.last_snapshot_time = 0
        self.snapshot_interval = config.get('snapshot_interval', 30.0)  # 30 saniye
        
        self.logger.info("Enhanced3DProcessor başlatıldı")
    
    def process_frame_enhanced_3d(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                                gaze_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ana 3D frame işleme fonksiyonu"""
        try:
            start_time = time.time()
            self.frame_count += 1
            
            # Sonuç dictionary'si
            result = {
                'frame_count': self.frame_count,
                'timestamp': start_time,
                'spatial_analysis': {},
                'head_pose_3d': None,
                'gaze_vector_3d': None,
                'objects_3d': [],
                'collision_events_3d': [],
                'firestore_logged': False,
                'processing_times': {},
                'success': True
            }
            
            # 1. Uzaysal çevre analizi
            spatial_start = time.time()
            spatial_result = self.spatial_analyzer.analyze_environment(frame, detections, gaze_data)
            spatial_time = time.time() - spatial_start
            self.processing_times['spatial_analysis'].append(spatial_time)
            
            result['spatial_analysis'] = spatial_result
            
            # 2. Kafa pozisyonu ve bakış yönü tahmini
            head_pose_start = time.time()
            head_pose_3d = None
            gaze_vector_3d = None
            
            if gaze_data and gaze_data.get('face_landmarks'):
                head_pose_3d = self.head_pose_estimator.estimate_head_pose(frame, gaze_data['face_landmarks'])
                gaze_vector_3d = self.gaze_estimator.estimate_gaze_vector(frame, gaze_data['face_landmarks'])
            
            head_pose_time = time.time() - head_pose_start
            self.processing_times['head_pose_estimation'].append(head_pose_time)
            
            result['head_pose_3d'] = head_pose_3d
            result['gaze_vector_3d'] = gaze_vector_3d
            
            # 3. Objelerin 3D konumlandırılması
            object_positioning_start = time.time()
            objects_3d = self.object_positioner.position_objects_3d(
                detections, frame.shape, spatial_result
            )
            object_positioning_time = time.time() - object_positioning_start
            self.processing_times['object_positioning'].append(object_positioning_time)
            
            result['objects_3d'] = objects_3d
            
            # 4. 3D çakışma tespiti
            collision_start = time.time()
            collision_events_3d = []
            
            if gaze_vector_3d and objects_3d:
                collision_events_3d = self.collision_detector.detect_3d_collisions(
                    gaze_vector_3d, objects_3d, spatial_result.get('spatial_people', [])
                )
            
            collision_time = time.time() - collision_start
            self.processing_times['collision_detection'].append(collision_time)
            
            result['collision_events_3d'] = collision_events_3d
            
            # 5. Firestore logging
            firestore_start = time.time()
            firestore_success = self._log_to_firestore(
                gaze_vector_3d, head_pose_3d, objects_3d, collision_events_3d, spatial_result
            )
            firestore_time = time.time() - firestore_start
            self.processing_times['firestore_logging'].append(firestore_time)
            
            result['firestore_logged'] = firestore_success
            
            # 6. Periodic snapshot
            current_time = time.time()
            if current_time - self.last_snapshot_time >= self.snapshot_interval:
                self._create_environment_snapshot(spatial_result, objects_3d)
                self.last_snapshot_time = current_time
            
            # Performance metrics
            total_time = time.time() - start_time
            result['processing_times'] = {
                'spatial_analysis': spatial_time,
                'head_pose_estimation': head_pose_time,
                'object_positioning': object_positioning_time,
                'collision_detection': collision_time,
                'firestore_logging': firestore_time,
                'total_time': total_time
            }
            
            # Keep processing times limited
            self._cleanup_processing_times()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced 3D frame processing hatası: {e}")
            return {
                'frame_count': self.frame_count,
                'timestamp': time.time(),
                'success': False,
                'error': str(e)
            }
    
    def _log_to_firestore(self, gaze_vector_3d, head_pose_3d, objects_3d, 
                         collision_events_3d, spatial_result) -> bool:
        """Firestore'a log kaydetme"""
        try:
            # Gaze tracking log
            if gaze_vector_3d:
                self.firestore_manager.log_3d_gaze_tracking(
                    gaze_vector_3d, head_pose_3d, spatial_result
                )
            
            # Collision events log
            for collision_event in collision_events_3d:
                self.firestore_manager.log_3d_collision_event(
                    collision_event, spatial_result
                )
            
            # Enhanced attention analytics (her 10 frame'de bir)
            if self.frame_count % 10 == 0:
                attention_analysis = self._analyze_attention_patterns(collision_events_3d)
                collision_analysis = self._analyze_collision_patterns(collision_events_3d)
                
                self.firestore_manager.log_enhanced_attention_analytics(
                    attention_analysis, collision_analysis
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Firestore logging hatası: {e}")
            return False
    
    def _create_environment_snapshot(self, spatial_result, objects_3d):
        """Çevre snapshot'ı oluştur"""
        try:
            spatial_people = spatial_result.get('spatial_people', [])
            environment_analysis = spatial_result.get('spatial_relationships', {})
            
            self.firestore_manager.log_spatial_environment_snapshot(
                objects_3d, spatial_people, environment_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Environment snapshot oluşturma hatası: {e}")
    
    def _analyze_attention_patterns(self, collision_events_3d) -> Dict[str, Any]:
        """Dikkat desenlerini analiz et"""
        try:
            if not collision_events_3d:
                return {}
            
            # Nesne türlerine göre dikkat
            object_attention = {}
            for collision in collision_events_3d:
                obj_label = collision.object_label
                if obj_label not in object_attention:
                    object_attention[obj_label] = {
                        'total_duration': 0.0,
                        'collision_count': 0,
                        'avg_confidence': 0.0
                    }
                
                object_attention[obj_label]['total_duration'] += collision.duration
                object_attention[obj_label]['collision_count'] += 1
                object_attention[obj_label]['avg_confidence'] += collision.collision_confidence
            
            # Ortalamaları hesapla
            for obj_label, stats in object_attention.items():
                if stats['collision_count'] > 0:
                    stats['avg_confidence'] /= stats['collision_count']
            
            return {
                'object_attention': object_attention,
                'total_attention_time': sum(c.duration for c in collision_events_3d),
                'most_attended_objects': sorted(object_attention.items(), 
                                              key=lambda x: x[1]['total_duration'], 
                                              reverse=True)[:5]
            }
            
        except Exception as e:
            self.logger.error(f"Dikkat deseni analizi hatası: {e}")
            return {}
    
    def _analyze_collision_patterns(self, collision_events_3d) -> Dict[str, Any]:
        """Çakışma desenlerini analiz et"""
        try:
            if not collision_events_3d:
                return {}
            
            # Temel istatistikler
            durations = [c.duration for c in collision_events_3d if c.duration > 0]
            confidences = [c.collision_confidence for c in collision_events_3d]
            distances = [c.collision_distance for c in collision_events_3d if c.collision_distance < float('inf')]
            
            return {
                'total_collisions': len(collision_events_3d),
                'meaningful_collisions': len([c for c in collision_events_3d if c.duration >= 1.0]),
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'avg_distance': sum(distances) / len(distances) if distances else 0,
                'collision_distribution': self._get_collision_distribution(collision_events_3d)
            }
            
        except Exception as e:
            self.logger.error(f"Çakışma deseni analizi hatası: {e}")
            return {}
    
    def _get_collision_distribution(self, collision_events_3d) -> Dict[str, int]:
        """Çakışma dağılımını al"""
        try:
            distribution = {}
            for collision in collision_events_3d:
                obj_label = collision.object_label
                distribution[obj_label] = distribution.get(obj_label, 0) + 1
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Çakışma dağılımı alma hatası: {e}")
            return {}
    
    def _cleanup_processing_times(self):
        """Processing time'ları temizle"""
        try:
            max_size = 100
            for key in self.processing_times:
                if len(self.processing_times[key]) > max_size:
                    self.processing_times[key] = self.processing_times[key][-max_size:]
        except Exception as e:
            self.logger.error(f"Processing time temizleme hatası: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Performance metriklerini al"""
        try:
            metrics = {}
            
            for key, times in self.processing_times.items():
                if times:
                    metrics[key] = {
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'sample_count': len(times)
                    }
                else:
                    metrics[key] = {
                        'avg_time': 0,
                        'min_time': 0,
                        'max_time': 0,
                        'sample_count': 0
                    }
            
            return {
                'frame_count': self.frame_count,
                'processing_metrics': metrics,
                'module_status': {
                    'spatial_analyzer': 'active',
                    'head_pose_estimator': 'active',
                    'gaze_estimator': 'active',
                    'object_positioner': 'active',
                    'collision_detector': 'active',
                    'firestore_manager': 'active'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrik alma hatası: {e}")
            return {}
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """Çakışma istatistiklerini al"""
        try:
            return self.collision_detector.get_collision_statistics()
        except Exception as e:
            self.logger.error(f"Çakışma istatistikleri alma hatası: {e}")
            return {}
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Session özetini al"""
        try:
            firestore_stats = self.firestore_manager.get_session_statistics()
            performance_metrics = self.get_performance_metrics()
            collision_stats = self.get_collision_statistics()
            
            return {
                'session_info': firestore_stats,
                'performance_metrics': performance_metrics,
                'collision_statistics': collision_stats,
                'system_status': 'active' if self.session_active else 'inactive'
            }
            
        except Exception as e:
            self.logger.error(f"Session özeti alma hatası: {e}")
            return {}
    
    def finalize_session(self):
        """Session'ı sonlandır"""
        try:
            self.session_active = False
            self.firestore_manager.finalize_session()
            
            # Final analytics
            final_summary = self.get_session_summary()
            
            self.logger.info("Enhanced 3D Processing session sonlandırıldı")
            return final_summary
            
        except Exception as e:
            self.logger.error(f"Session sonlandırma hatası: {e}")
            return {}
    
    def reset_session(self):
        """Session'ı sıfırla"""
        try:
            self.finalize_session()
            
            # Reset counters
            self.frame_count = 0
            self.last_snapshot_time = 0
            
            # Clear processing times
            for key in self.processing_times:
                self.processing_times[key].clear()
            
            # Reset components
            self.collision_detector.clear_collision_history()
            
            self.session_active = True
            
            self.logger.info("Enhanced 3D Processing session sıfırlandı")
            
        except Exception as e:
            self.logger.error(f"Session sıfırlama hatası: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Sistem durumunu al"""
        try:
            return {
                'session_active': self.session_active,
                'frame_count': self.frame_count,
                'modules_loaded': {
                    'spatial_analyzer': self.spatial_analyzer is not None,
                    'head_pose_estimator': self.head_pose_estimator is not None,
                    'gaze_estimator': self.gaze_estimator is not None,
                    'object_positioner': self.object_positioner is not None,
                    'collision_detector': self.collision_detector is not None,
                    'firestore_manager': self.firestore_manager is not None
                },
                'last_snapshot_time': self.last_snapshot_time,
                'snapshot_interval': self.snapshot_interval
            }
            
        except Exception as e:
            self.logger.error(f"Sistem durumu alma hatası: {e}")
            return {'error': str(e)} 