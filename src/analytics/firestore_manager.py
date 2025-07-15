"""
Firestore Analytics Yöneticisi - Ana Sınıf
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .base_logger import BaseFirestoreLogger, convert_numpy_types, SessionMetrics
from .collision_analytics import CollisionAnalytics


@dataclass
class DetectionEvent:
    """Tespit eventi"""
    timestamp: datetime
    detection_type: str
    object_class: str
    confidence: float
    bbox: List[int]
    scene_context: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'detection_type': self.detection_type,
            'object_class': self.object_class,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'scene_context': self.scene_context
        }


@dataclass
class AttentionEvent:
    """Dikkat eventi"""
    timestamp: datetime
    attention_score: float
    gaze_point: List[float]
    focused_objects: List[str]
    interest_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'attention_score': self.attention_score,
            'gaze_point': self.gaze_point,
            'focused_objects': self.focused_objects,
            'interest_level': self.interest_level
        }


@dataclass
class GazeVectorLog:
    """Gaze vector log"""
    timestamp: datetime
    session_id: str
    gaze_point: List[float]
    gaze_vector: List[float]
    vector_strength: float
    vector_direction_degrees: float
    left_eye_center: List[float]
    right_eye_center: List[float]
    eye_distance: float
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'gaze_point': self.gaze_point,
            'gaze_vector': self.gaze_vector,
            'vector_strength': self.vector_strength,
            'vector_direction_degrees': self.vector_direction_degrees,
            'left_eye_center': self.left_eye_center,
            'right_eye_center': self.right_eye_center,
            'eye_distance': self.eye_distance,
            'confidence_score': self.confidence_score
        }


@dataclass
class InteractionEvent:
    """Etkileşim eventi"""
    timestamp: datetime
    session_id: str
    event_type: str
    object_id: str
    object_class: str
    interaction_duration: float
    attention_intensity: float
    gaze_stability: float
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'event_type': self.event_type,
            'object_id': self.object_id,
            'object_class': self.object_class,
            'interaction_duration': self.interaction_duration,
            'attention_intensity': self.attention_intensity,
            'gaze_stability': self.gaze_stability,
            'context': self.context
        }


@dataclass
class HumanReadableEvent:
    """İnsan tarafından okunabilir event"""
    timestamp: datetime
    event_type: str
    description: str
    details: Dict[str, Any]
    severity: str  # 'info', 'warning', 'important'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'description': self.description,
            'details': self.details,
            'severity': self.severity
        }


@dataclass
class AttentionSummary:
    """Dikkat özeti"""
    session_id: str
    total_people: int
    objects_looked_at: List[Dict[str, Any]]
    attention_duration: float
    most_interesting_object: str
    gaze_stability: float
    
    def to_readable_summary(self) -> str:
        """Okunabilir özet oluştur"""
        if self.total_people == 0:
            return "Hiç kimse sistemin görüş alanında değil."
        
        if self.total_people == 1:
            person_text = "1 kişi"
        else:
            person_text = f"{self.total_people} kişi"
        
        # Ana mesaj
        main_message = f"{person_text} sistemi kullanıyor"
        
        # En çok bakılan obje
        if self.most_interesting_object:
            main_message += f", en çok {self.most_interesting_object} objesine bakıyor"
        
        # Toplam dikkat süresi
        if self.attention_duration > 0:
            main_message += f". Toplam {self.attention_duration:.1f} saniye boyunca objeler incelendi"
        
        # Bakış kararlılığı
        if self.gaze_stability > 0.7:
            main_message += ". Bakış kararlı"
        elif self.gaze_stability > 0.4:
            main_message += ". Bakış orta kararlılıkta"
        else:
            main_message += ". Bakış dağınık"
        
        return main_message + "."


class FirestoreAnalytics(CollisionAnalytics):
    """Ana Firestore Analytics sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Detection türleri istatistikleri
        self.detection_types = {}
        
        # Batch processing
        self.batch_size = config.get('batch_size', 10)
        self.batch_timeout = config.get('batch_timeout', 30)
        
        # Human-readable tracking
        self.current_attention_summary = None
        self.last_readable_log_time = time.time()
        self.readable_log_interval = config.get('readable_log_interval', 10)  # 10 saniye
        
        # Persistent counters
        self.attention_counters = {
            'total_people_detected': 0,
            'unique_objects_looked_at': set(),
            'total_attention_time': 0.0,
            'gaze_sessions': []
        }
        
        self.logger.info("FirestoreAnalytics başlatıldı")
    
    def log_detection_event(self, detection_data: Dict[str, Any]):
        """Tespit eventi kaydet"""
        try:
            detection_event = DetectionEvent(
                timestamp=datetime.now(timezone.utc),
                detection_type=detection_data.get('type', 'unknown'),
                object_class=detection_data.get('label', 'unknown'),
                confidence=detection_data.get('confidence', 0.0),
                bbox=detection_data.get('bbox', [0, 0, 0, 0]),
                scene_context=detection_data.get('scene_context', 'unknown')
            )
            
            self.log_basic_event('detection_events', detection_event.to_dict())
            
            # İstatistikleri güncelle
            self._update_detection_stats(detection_event)
            
        except Exception as e:
            self.logger.error(f"Detection event logging hatası: {e}")
    
    def log_attention_event(self, attention_data: Dict[str, Any]):
        """Dikkat eventi kaydet"""
        try:
            clean_attention = convert_numpy_types(attention_data)
            
            attention_event = AttentionEvent(
                timestamp=datetime.now(timezone.utc),
                attention_score=clean_attention.get('attention_score', 0.0),
                gaze_point=clean_attention.get('gaze_point', [0, 0]),
                focused_objects=clean_attention.get('focused_objects', []),
                interest_level=clean_attention.get('interest_level', 0.0)
            )
            
            self.log_basic_event('attention_events', attention_event.to_dict())
            
            # Cache'e ekle
            self.attention_cache.append(attention_event.to_dict())
            
        except Exception as e:
            self.logger.error(f"Attention event logging hatası: {e}")
    
    def log_gaze_vector_detailed(self, gaze_data: Dict[str, Any]):
        """Detaylı gaze vector kaydet"""
        try:
            clean_gaze = convert_numpy_types(gaze_data)
            
            # Gaze vector verilerini hesapla
            gaze_vector = clean_gaze.get('gaze_vector', (0, 0))
            vector_strength = (gaze_vector[0]**2 + gaze_vector[1]**2)**0.5
            import numpy as np
            vector_direction = np.degrees(np.arctan2(gaze_vector[1], gaze_vector[0]))
            
            left_eye = clean_gaze.get('left_eye_center', [0, 0])
            right_eye = clean_gaze.get('right_eye_center', [0, 0])
            eye_distance = ((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)**0.5
            
            gaze_log = GazeVectorLog(
                timestamp=datetime.now(timezone.utc),
                session_id=self.session_id,
                gaze_point=clean_gaze.get('gaze_point', [0, 0]),
                gaze_vector=list(gaze_vector),
                vector_strength=vector_strength,
                vector_direction_degrees=float(vector_direction),
                left_eye_center=left_eye,
                right_eye_center=right_eye,
                eye_distance=eye_distance,
                confidence_score=1.0
            )
            
            self.log_basic_event('gaze_vectors', gaze_log.to_dict())
            
        except Exception as e:
            self.logger.error(f"Gaze vector logging hatası: {e}")
    
    def log_interaction_event(self, event_type: str, object_data: Dict[str, Any], 
                            attention_data: Dict[str, Any]):
        """Object interaction eventi kaydet"""
        try:
            clean_object = convert_numpy_types(object_data)
            clean_attention = convert_numpy_types(attention_data)
            
            interaction = InteractionEvent(
                timestamp=datetime.now(timezone.utc),
                session_id=self.session_id,
                event_type=event_type,
                object_id=clean_object.get('object_id', 'unknown'),
                object_class=clean_object.get('label', 'unknown'),
                interaction_duration=clean_object.get('attention_duration', 0.0),
                attention_intensity=clean_object.get('attention_score', 0.0),
                gaze_stability=clean_attention.get('gaze_stability', 0.0),
                context={
                    'detection_confidence': clean_object.get('confidence', 0.0),
                    'bbox': clean_object.get('bbox', [0, 0, 0, 0]),
                    'category': clean_object.get('category', 'unknown'),
                    'scene_context': clean_object.get('scene_context', 'unknown'),
                    'interest_level': clean_attention.get('interest_level', 0.0)
                }
            )
            
            self.log_basic_event('interaction_events', interaction.to_dict())
            
        except Exception as e:
            self.logger.error(f"Interaction event logging hatası: {e}")
    
    def log_batch_data(self, batch_data: Dict[str, Any]):
        """Batch data kaydet"""
        try:
            clean_batch = convert_numpy_types(batch_data)
            
            batch_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'frame_count': clean_batch.get('frame_count', 0),
                'detections': clean_batch.get('detections', []),
                'attention': clean_batch.get('attention', {}),
                'camera_settings': clean_batch.get('camera_settings', {})
            }
            
            self.log_basic_event('batch_data', batch_log)
            
        except Exception as e:
            self.logger.error(f"Batch data logging hatası: {e}")
    
    def log_object_attention(self, attention_summary: Dict[str, Any]):
        """Object attention summary kaydet"""
        try:
            clean_summary = convert_numpy_types(attention_summary)
            
            attention_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'total_active_objects': clean_summary.get('total_active_objects', 0),
                'objects': clean_summary.get('objects', []),
                'attention_distribution': clean_summary.get('attention_distribution', {}),
                'top_objects': clean_summary.get('top_objects', [])
            }
            
            self.log_basic_event('object_attention', attention_log)
            
        except Exception as e:
            self.logger.error(f"Object attention logging hatası: {e}")
    
    def log_detailed_attention_event(self, attention_data: Dict[str, Any], 
                                   attention_analysis: Dict[str, Any]):
        """Detaylı attention event kaydet"""
        try:
            clean_attention = convert_numpy_types(attention_data)
            clean_analysis = convert_numpy_types(attention_analysis)
            
            detailed_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'raw_attention': clean_attention,
                'analysis_results': clean_analysis,
                'processing_time': time.time()
            }
            
            self.log_basic_event('detailed_attention', detailed_log)
            
        except Exception as e:
            self.logger.error(f"Detailed attention logging hatası: {e}")
    
    def log_3d_gaze_data(self, gaze_3d_data: Dict[str, Any]):
        """3D gaze verilerini kaydet"""
        try:
            gaze_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'gaze_3d_vector': gaze_3d_data.get('gaze_3d_vector', [0, 0, 0]),
                'depth_estimate': gaze_3d_data.get('depth_estimate', 0.0),
                'active_gaze_objects': len(gaze_3d_data.get('gaze_objects', [])),
                'gaze_objects_data': [
                    {
                        'position': obj.position,
                        'size': obj.size,
                        'color': obj.color,
                        'alpha': obj.alpha,
                        'creation_time': obj.creation_time,
                        'age': time.time() - obj.creation_time
                    }
                    for obj in gaze_3d_data.get('gaze_objects', [])
                ],
                'processing_stats': {
                    'frame_count': gaze_3d_data.get('frame_count', 0),
                    'processing_time': gaze_3d_data.get('processing_time', 0.0)
                }
            }
            
            self.log_basic_event('3d_gaze_data', convert_numpy_types(gaze_data))
            
        except Exception as e:
            self.logger.error(f"3D gaze data logging hatası: {e}")
    
    def log_real_time_analytics(self, frame_data: Dict[str, Any]):
        """Real-time analytics kaydet"""
        try:
            analytics_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'frame_count': frame_data.get('frame_count', 0),
                'processing_intensity': frame_data.get('processing_intensity', 0.0),
                'detection_count': frame_data.get('detection_count', 0),
                'attention_active': frame_data.get('attention_active', False),
                'gaze_tracking_active': frame_data.get('gaze_tracking_active', False),
                'fps_current': frame_data.get('fps_current', 0.0),
                'memory_usage_mb': frame_data.get('memory_usage_mb', 0),
                'cpu_usage_percent': frame_data.get('cpu_usage_percent', 0.0),
                'scene_type': frame_data.get('scene_type', 'unknown'),
                'performance_mode': frame_data.get('performance_mode', 'unknown')
            }
            
            self.log_basic_event('real_time_analytics', analytics_data)
            
        except Exception as e:
            self.logger.error(f"Real-time analytics logging hatası: {e}")
    
    def log_comprehensive_session_state(self, system_metrics: Dict[str, Any], 
                                      camera_metrics: Dict[str, Any],
                                      ai_metrics: Dict[str, Any]):
        """Kapsamlı session durumu kaydet"""
        try:
            comprehensive_state = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'uptime_seconds': time.time() - self.session_start.timestamp(),
                'system_state': {
                    'cpu_percent': system_metrics.get('cpu_percent', 0),
                    'memory_percent': system_metrics.get('memory_percent', 0),
                    'disk_percent': system_metrics.get('disk_percent', 0)
                },
                'camera_state': {
                    'actual_fps': camera_metrics.get('actual_fps', 0),
                    'target_fps': camera_metrics.get('target_fps', 30),
                    'scene_type': camera_metrics.get('scene_type', 'unknown')
                },
                'ai_state': {
                    'model_complexity': str(ai_metrics.get('model_complexity', 'unknown')),
                    'inference_time': ai_metrics.get('inference_time', 0)
                },
                'session_metrics': self.session_metrics.to_dict(),
                'queue_status': {
                    'write_queue_size': self.write_queue.qsize(),
                    'detection_cache_size': len(self.detection_cache),
                    'attention_cache_size': len(self.attention_cache)
                }
            }
            
            self.log_basic_event('comprehensive_session_states', comprehensive_state)
            
        except Exception as e:
            self.logger.error(f"Comprehensive session state logging hatası: {e}")
    
    def log_human_readable_event(self, event_type: str, description: str, 
                                 details: Dict[str, Any] = None, severity: str = 'info'):
        """İnsan tarafından okunabilir event kaydet"""
        try:
            event = HumanReadableEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                description=description,
                details=details or {},
                severity=severity
            )
            
            self.log_basic_event('human_readable_events', event.to_dict())
            
            # Önemli eventleri console'da da göster
            if severity in ['warning', 'important']:
                self.logger.info(f"📝 {description}")
                
        except Exception as e:
            self.logger.error(f"Human readable event logging hatası: {e}")
    
    def log_attention_activity(self, attention_analysis: Dict[str, Any]):
        """Dikkat aktivitesini human-readable format'ta kaydet"""
        try:
            current_time = time.time()
            
            # Sadece belirli aralıklarla log
            if current_time - self.last_readable_log_time < self.readable_log_interval:
                return
                
            self.last_readable_log_time = current_time
            
            # Dikkat analizi verilerini işle
            focused_objects = attention_analysis.get('focused_objects', [])
            interest_level = attention_analysis.get('interest_level', 0.0)
            object_attention_summary = attention_analysis.get('object_attention_summary', {})
            
            # İnsan sayısı (basit tahmin)
            people_count = 1 if focused_objects else 0
            
            # Bakılan objeler
            looked_objects = []
            total_attention_time = 0.0
            
            for obj in focused_objects:
                obj_info = {
                    'object_name': obj.get('label', 'bilinmeyen obje'),
                    'attention_duration': obj.get('attention_duration', 0.0),
                    'confidence': obj.get('confidence', 0.0)
                }
                looked_objects.append(obj_info)
                total_attention_time += obj_info['attention_duration']
            
            # En çok bakılan obje
            most_interesting = None
            if looked_objects:
                most_interesting = max(looked_objects, key=lambda x: x['attention_duration'])['object_name']
            
            # Gaze stabilite
            gaze_stability = attention_analysis.get('gaze_stability', 0.0)
            
            # Human-readable mesaj oluştur
            readable_descriptions = []
            
            if people_count > 0:
                if len(looked_objects) == 0:
                    readable_descriptions.append("1 kişi kamerayı izliyor ama hiçbir objeye odaklanmıyor")
                elif len(looked_objects) == 1:
                    obj_name = looked_objects[0]['object_name']
                    duration = looked_objects[0]['attention_duration']
                    readable_descriptions.append(f"1 kişi {obj_name} objesine {duration:.1f} saniye bakıyor")
                else:
                    readable_descriptions.append(f"1 kişi {len(looked_objects)} farklı objeye bakıyor")
                    
                    # En çok bakılan obje
                    if most_interesting:
                        max_duration = max(obj['attention_duration'] for obj in looked_objects)
                        readable_descriptions.append(f"En çok {most_interesting} objesine bakıyor ({max_duration:.1f} saniye)")
            else:
                readable_descriptions.append("Hiç kimse sistemin görüş alanında değil")
            
            # Gaze stability açıklaması
            if gaze_stability > 0.8:
                readable_descriptions.append("Bakış çok kararlı")
            elif gaze_stability > 0.6:
                readable_descriptions.append("Bakış kararlı")
            elif gaze_stability > 0.4:
                readable_descriptions.append("Bakış orta kararlılıkta")
            else:
                readable_descriptions.append("Bakış dağınık")
            
            # Ana readable event
            main_description = ". ".join(readable_descriptions)
            
            # Detaylar
            details = {
                'people_count': people_count,
                'objects_looked_at': looked_objects,
                'total_attention_time': total_attention_time,
                'interest_level': interest_level,
                'gaze_stability': gaze_stability,
                'session_stats': {
                    'objects_in_view': len(looked_objects),
                    'avg_attention_per_object': total_attention_time / max(len(looked_objects), 1),
                    'most_interesting_object': most_interesting
                }
            }
            
            # Log olarak kaydet
            self.log_human_readable_event(
                event_type='attention_activity',
                description=main_description,
                details=details,
                severity='info'
            )
            
            # Counters güncelle
            self.attention_counters['total_people_detected'] = max(self.attention_counters['total_people_detected'], people_count)
            self.attention_counters['total_attention_time'] += total_attention_time
            
            for obj in looked_objects:
                self.attention_counters['unique_objects_looked_at'].add(obj['object_name'])
            
        except Exception as e:
            self.logger.error(f"Attention activity logging hatası: {e}")
    
    def log_collision_activity(self, collision_events: List[Any]):
        """Çakışma aktivitesini human-readable format'ta kaydet"""
        try:
            if not collision_events:
                return
            
            current_time = time.time()
            
            # Aktif çakışmalar
            active_collisions = [e for e in collision_events if e.is_active]
            
            if not active_collisions:
                return
            
            # Çakışma açıklamaları
            collision_descriptions = []
            
            for collision in active_collisions:
                obj_label = collision.object_data.get('label', 'bilinmeyen obje')
                duration = collision.duration
                depth = collision.collision_depth
                
                if duration < 1.0:
                    time_desc = f"{duration:.1f} saniye"
                else:
                    time_desc = f"{duration:.0f} saniye"
                
                collision_descriptions.append(f"{obj_label} ({time_desc}, {depth:.0f}mm uzaklıkta)")
            
            # Ana mesaj
            if len(collision_descriptions) == 1:
                main_message = f"Kullanıcı {collision_descriptions[0]} ile etkileşimde"
            else:
                main_message = f"Kullanıcı {len(collision_descriptions)} obje ile etkileşimde: {', '.join(collision_descriptions)}"
            
            # Detaylar
            details = {
                'active_collision_count': len(active_collisions),
                'collision_objects': [
                    {
                        'object_name': c.object_data.get('label', 'unknown'),
                        'duration': c.duration,
                        'depth': c.collision_depth
                    } for c in active_collisions
                ],
                'avg_collision_duration': sum(c.duration for c in active_collisions) / len(active_collisions),
                'closest_object': min(active_collisions, key=lambda x: x.collision_depth).object_data.get('label', 'unknown')
            }
            
            # Log
            self.log_human_readable_event(
                event_type='collision_activity',
                description=main_message,
                details=details,
                severity='important'
            )
            
        except Exception as e:
            self.logger.error(f"Collision activity logging hatası: {e}")
    
    def log_session_summary(self):
        """Oturum özeti human-readable format'ta"""
        try:
            current_time = time.time()
            uptime = current_time - self.session_start.timestamp()
            
            # Özet istatistikleri
            unique_objects = len(self.attention_counters['unique_objects_looked_at'])
            total_attention = self.attention_counters['total_attention_time']
            
            # Ana mesaj
            if unique_objects == 0:
                main_message = f"Oturum {uptime/60:.1f} dakika sürdü, hiçbir objeye odaklanılmadı"
            else:
                main_message = f"Oturum {uptime/60:.1f} dakika sürdü, {unique_objects} farklı objeye toplam {total_attention:.1f} saniye bakıldı"
            
            # En çok bakılan objeler
            most_looked_objects = list(self.attention_counters['unique_objects_looked_at'])[:5]
            
            if most_looked_objects:
                main_message += f". En çok bakılan objeler: {', '.join(most_looked_objects)}"
            
            # Detaylar
            details = {
                'session_duration_minutes': uptime / 60,
                'total_people_detected': self.attention_counters['total_people_detected'],
                'unique_objects_looked_at': list(self.attention_counters['unique_objects_looked_at']),
                'total_attention_time': total_attention,
                'avg_attention_per_object': total_attention / max(unique_objects, 1),
                'objects_per_minute': unique_objects / (uptime / 60)
            }
            
            # Log
            self.log_human_readable_event(
                event_type='session_summary',
                description=main_message,
                details=details,
                severity='important'
            )
            
        except Exception as e:
            self.logger.error(f"Session summary logging hatası: {e}")
    
    def log_performance_insight(self, performance_data: Dict[str, Any]):
        """Performans insights human-readable format'ta"""
        try:
            fps = performance_data.get('fps', 0)
            memory_usage = performance_data.get('memory_usage_mb', 0)
            cpu_usage = performance_data.get('cpu_usage_percent', 0)
            
            # Performans değerlendirmesi
            performance_level = "mükemmel"
            if fps < 15:
                performance_level = "düşük"
            elif fps < 25:
                performance_level = "orta"
            elif fps < 30:
                performance_level = "iyi"
            
            # Memory durumu
            memory_status = "normal"
            if memory_usage > 80:
                memory_status = "yüksek"
            elif memory_usage > 60:
                memory_status = "orta"
            
            # Ana mesaj
            main_message = f"Sistem performansı {performance_level} ({fps:.1f} FPS), bellek kullanımı {memory_status} ({memory_usage:.1f}MB)"
            
            # Uyarılar
            warnings = []
            if fps < 20:
                warnings.append("Düşük FPS nedeniyle gaze tracking kalitesi etkilenebilir")
            if memory_usage > 75:
                warnings.append("Yüksek bellek kullanımı")
            if cpu_usage > 80:
                warnings.append("Yüksek CPU kullanımı")
            
            if warnings:
                main_message += f". Uyarılar: {', '.join(warnings)}"
            
            # Detaylar
            details = {
                'fps': fps,
                'memory_usage_mb': memory_usage,
                'cpu_usage_percent': cpu_usage,
                'performance_level': performance_level,
                'memory_status': memory_status,
                'warnings': warnings
            }
            
            # Severity
            severity = 'warning' if warnings else 'info'
            
            # Log
            self.log_human_readable_event(
                event_type='performance_insight',
                description=main_message,
                details=details,
                severity=severity
            )
            
        except Exception as e:
            self.logger.error(f"Performance insight logging hatası: {e}")
    
    def get_readable_analytics_summary(self) -> Dict[str, Any]:
        """Okunabilir analytics özetini döndür"""
        try:
            current_time = time.time()
            uptime = current_time - self.session_start.timestamp()
            
            # Ana istatistikler
            unique_objects = len(self.attention_counters['unique_objects_looked_at'])
            total_attention = self.attention_counters['total_attention_time']
            
            # Okunabilir özet
            summary = {
                'session_duration_readable': f"{uptime/60:.1f} dakika",
                'people_detected_readable': f"{self.attention_counters['total_people_detected']} kişi tespit edildi",
                'objects_interaction_readable': f"{unique_objects} farklı obje ile etkileşim",
                'total_attention_readable': f"{total_attention:.1f} saniye toplam dikkat",
                'avg_attention_readable': f"{total_attention/max(unique_objects, 1):.1f} saniye/obje ortalama dikkat",
                'most_interesting_objects': list(self.attention_counters['unique_objects_looked_at'])[:5],
                'activity_level': self._get_activity_level_description(unique_objects, total_attention, uptime)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Readable analytics summary hatası: {e}")
            return {}
    
    def _get_activity_level_description(self, unique_objects: int, total_attention: float, uptime: float) -> str:
        """Aktivite seviyesi açıklaması"""
        if uptime < 60:  # 1 dakikadan az
            return "Çok kısa oturum"
        
        objects_per_minute = unique_objects / (uptime / 60)
        attention_per_minute = total_attention / (uptime / 60)
        
        if objects_per_minute > 3 and attention_per_minute > 30:
            return "Çok aktif - birçok obje ile yoğun etkileşim"
        elif objects_per_minute > 2 and attention_per_minute > 20:
            return "Aktif - düzenli obje etkileşimi"
        elif objects_per_minute > 1 and attention_per_minute > 10:
            return "Orta - bazı obje etkileşimi"
        elif objects_per_minute > 0.5:
            return "Düşük - sınırlı obje etkileşimi"
        else:
            return "Çok düşük - minimal etkileşim"
    
    def close(self):
        """Kapatırken son özeti kaydet"""
        self.log_session_summary()
        super().close()
    
    def _update_detection_stats(self, detection_event: DetectionEvent):
        """Detection istatistiklerini güncelle"""
        try:
            detection_type = detection_event.detection_type
            if detection_type not in self.detection_types:
                self.detection_types[detection_type] = 0
            self.detection_types[detection_type] += 1
            
            # Session metrics güncelle
            self.session_metrics.total_detections += 1
            
        except Exception as e:
            self.logger.error(f"Detection stats update hatası: {e}")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Detection istatistiklerini döndür"""
        return {
            'detection_types': self.detection_types.copy(),
            'total_detections': self.session_metrics.total_detections,
            'session_id': self.session_id,
            'avg_confidence': self._calculate_avg_confidence()
        }
    
    def _calculate_avg_confidence(self) -> float:
        """Ortalama confidence hesapla"""
        if not self.detection_cache:
            return 0.0
        
        total_confidence = sum(item.get('confidence', 0) for item in self.detection_cache)
        return total_confidence / len(self.detection_cache) 