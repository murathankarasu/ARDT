"""
ARDS Frame İşleme Modülü
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone


class FrameProcessor:
    """Frame işleme ve analiz sınıfı"""
    
    def __init__(self, components: Dict[str, Any]):
        """
        Args:
            components: Sistem bileşenleri dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Bileşenler
        self.object_detector = components.get('object_detector')
        self.ai_classifier = components.get('ai_classifier')
        self.gaze_tracker = components.get('gaze_tracker')
        self.attention_analyzer = components.get('attention_analyzer')
        self.gaze_3d_processor = components.get('gaze_3d_processor')
        self.visualization_manager = components.get('visualization_manager')
        self.analytics = components.get('analytics')
        self.camera_manager = components.get('camera_manager')
        self.system_monitor = components.get('system_monitor')
        
        # Frame sayacı
        self.frame_count = 0
        
        # Performance tracking
        self.processing_times = []
        self.detection_stats = {
            'total_detections': 0,
            'total_collisions': 0,
            'last_collision_time': None
        }
    
    def process_frame(self, frame, mode: str = 'mixed') -> Tuple[Any, Dict[str, Any]]:
        """
        Ana frame işleme fonksiyonu
        
        Args:
            frame: Giriş frame'i
            mode: Tespit modu ('street', 'store', 'mixed')
            
        Returns:
            Tuple[processed_frame, processing_stats]
        """
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # 1. Nesne tespiti
            detections = self._detect_objects(frame, mode)
            
            # 2. AI sınıflandırma (her 8 frame'de bir)
            enhanced_detections = self._enhance_with_ai(frame, detections)
            
            # 3. Gaze tracking ve 3D processing
            gaze_3d_data, collision_events, attention_analysis = self._process_gaze_data(frame, enhanced_detections)
            
            # 4. Analytics logging
            self._log_analytics_data(enhanced_detections, gaze_3d_data, collision_events, attention_analysis)
            
            # 5. Görselleştirme
            result_frame = self._render_frame(frame, gaze_3d_data, enhanced_detections, collision_events, attention_analysis)
            
            # 6. Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Processing stats
            stats = {
                'frame_count': self.frame_count,
                'processing_time': processing_time,
                'detection_count': len(enhanced_detections),
                'collision_count': len(collision_events),
                'gaze_active': gaze_3d_data is not None,
                'avg_processing_time': sum(self.processing_times[-10:]) / len(self.processing_times[-10:]) if self.processing_times else 0
            }
            
            return result_frame, stats
            
        except Exception as e:
            self.logger.error(f"Frame processing hatası: {e}")
            # Hata durumunda orijinal frame'i döndür
            return frame, {'error': str(e), 'frame_count': self.frame_count}
    
    def _detect_objects(self, frame, mode: str) -> List[Dict[str, Any]]:
        """Nesne tespiti"""
        try:
            if not self.object_detector:
                return []
            
            detections = self.object_detector.detect(frame, mode)
            self.detection_stats['total_detections'] += len(detections)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Nesne tespiti hatası: {e}")
            return []
    
    def _enhance_with_ai(self, frame, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """AI ile tespit verilerini geliştirir"""
        try:
            if not self.ai_classifier or not detections:
                return detections
            
            # Her 8 frame'de bir AI kullan
            if self.frame_count % 8 != 0:
                return detections
            
            enhanced = []
            camera_status = self.camera_manager.get_camera_status() if self.camera_manager else {}
            current_scene = camera_status.get('current_scene', 'unknown')
            
            for detection in detections:
                try:
                    ai_result = self.ai_classifier.classify_region(
                        frame, detection['bbox'], context=current_scene
                    )
                    
                    enhanced_detection = detection.copy()
                    enhanced_detection.update({
                        'ai_class': ai_result.get('class', 'unknown'),
                        'ai_confidence': ai_result.get('confidence', 0.0),
                        'ai_method': ai_result.get('method', 'unknown'),
                        'scene_context': current_scene
                    })
                    enhanced.append(enhanced_detection)
                    
                except Exception as e:
                    self.logger.error(f"AI classification hatası: {e}")
                    enhanced.append(detection)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"AI enhancement hatası: {e}")
            return detections
    
    def _process_gaze_data(self, frame, detections: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Any], Optional[Dict[str, Any]]]:
        """Gaze verilerini işle"""
        try:
            gaze_3d_data = {}
            collision_events = []
            attention_analysis = None
            
            if not self.gaze_tracker:
                return gaze_3d_data, collision_events, attention_analysis
            
            # Temel gaze analizi
            gaze_data = self.gaze_tracker.analyze(frame)
            
            if gaze_data:
                # 3D gaze processing
                if self.gaze_3d_processor:
                    gaze_3d_data = self.gaze_3d_processor.process_gaze_data(gaze_data, frame.shape)
                    
                    # Çakışma kontrolü
                    collision_events = self.gaze_3d_processor.check_collisions(gaze_3d_data, detections, frame.shape)
                    
                    # Çakışma istatistikleri
                    if collision_events:
                        self.detection_stats['total_collisions'] += len(collision_events)
                        self.detection_stats['last_collision_time'] = time.time()
                
                # Attention analizi
                if self.attention_analyzer:
                    attention_analysis = self.attention_analyzer.analyze_attention_objects(
                        gaze_data, detections
                    )
            
            return gaze_3d_data, collision_events, attention_analysis
            
        except Exception as e:
            self.logger.error(f"Gaze processing hatası: {e}")
            return {}, [], None
    
    def _log_analytics_data(self, detections: List[Dict[str, Any]], gaze_3d_data: Dict[str, Any], 
                           collision_events: List[Any], attention_analysis: Optional[Dict[str, Any]]):
        """Analytics verilerini logla"""
        try:
            if not self.analytics:
                return
            
            # Çakışma olaylarını logla
            for collision_event in collision_events:
                if collision_event.is_active:
                    self.analytics.log_collision_event(collision_event)
                    
                    # Terminal log
                    obj_label = collision_event.object_data.get('label', 'Unknown')
                    self.logger.info(f"🎯 ÇAKIŞMA: {obj_label} - Süre: {collision_event.duration:.1f}s - Derinlik: {collision_event.collision_depth:.1f}")
            
            # Human-readable collision activity (her frame değil, sadece aktif çakışma olunca)
            if collision_events:
                self.analytics.log_collision_activity(collision_events)
            
            # Human-readable attention activity (her 30 frame'de bir)
            if self.frame_count % 30 == 0 and attention_analysis:
                self.analytics.log_attention_activity(attention_analysis)
            
            # 3D gaze verilerini logla (her 10 frame'de bir)
            if self.frame_count % 10 == 0 and gaze_3d_data:
                self.analytics.log_3d_gaze_data(gaze_3d_data)
            
            # Çakışma özeti (her 60 frame'de bir)
            if self.frame_count % 60 == 0 and self.gaze_3d_processor:
                collision_summary = self.gaze_3d_processor.get_collision_summary()
                if collision_summary['total_collision_events'] > 0:
                    self.analytics.log_collision_summary(collision_summary)
            
            # Performance insights (her 120 frame'de bir)
            if self.frame_count % 120 == 0:
                self._log_performance_insights()
            
            # Genel analytics (her 30 frame'de bir)
            if self.frame_count % 30 == 0:
                self._log_realtime_analytics(detections, gaze_3d_data, collision_events, attention_analysis)
            
        except Exception as e:
            self.logger.error(f"Analytics logging hatası: {e}")
    
    def _log_performance_insights(self):
        """Performans insights'ı logla"""
        try:
            if not self.analytics:
                return
            
            # Mevcut performans verilerini topla
            camera_status = self.camera_manager.get_camera_status() if self.camera_manager else {}
            system_metrics = self.system_monitor.get_current_metrics() if self.system_monitor else type('obj', (object,), {'memory_percent': 0, 'cpu_percent': 0})()
            
            # FPS hesaplama
            recent_times = self.processing_times[-30:] if self.processing_times else [0.033]
            avg_processing_time = sum(recent_times) / len(recent_times)
            current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            
            performance_data = {
                'fps': current_fps,
                'memory_usage_mb': system_metrics.memory_percent * 8192 / 100,  # Approximate MB
                'cpu_usage_percent': system_metrics.cpu_percent,
                'processing_time_ms': avg_processing_time * 1000,
                'detection_count': len(self.detection_stats) if hasattr(self, 'detection_stats') else 0,
                'gaze_active': self.gaze_tracker is not None,
                'collision_tracking_active': self.gaze_3d_processor is not None
            }
            
            # Human-readable performance insight
            self.analytics.log_performance_insight(performance_data)
            
        except Exception as e:
            self.logger.error(f"Performance insights logging hatası: {e}")
    
    def _log_realtime_analytics(self, detections: List[Dict[str, Any]], gaze_3d_data: Dict[str, Any], 
                               collision_events: List[Any], attention_analysis: Optional[Dict[str, Any]]):
        """Real-time analytics kaydet"""
        try:
            camera_status = self.camera_manager.get_camera_status() if self.camera_manager else {}
            system_metrics = self.system_monitor.get_current_metrics() if self.system_monitor else type('obj', (object,), {'memory_percent': 0, 'cpu_percent': 0})()
            
            # Gaze 3D istatistikleri
            gaze_3d_stats = gaze_3d_data.get('processing_stats', {}) if gaze_3d_data else {}
            
            analytics_data = {
                'frame_count': self.frame_count,
                'detection_count': len(detections),
                'attention_active': attention_analysis is not None,
                'gaze_tracking_active': self.gaze_tracker is not None,
                'active_gaze_objects': len(gaze_3d_data.get('gaze_history', [])),
                'active_collisions': len([e for e in collision_events if e.is_active]),
                'fps_current': camera_status.get('actual_fps', 0),
                'memory_usage_mb': system_metrics.memory_percent * 81.92,  # Approximate conversion
                'cpu_usage_percent': system_metrics.cpu_percent,
                'scene_type': camera_status.get('current_scene', 'unknown'),
                'performance_mode': 'balanced',  # Default
                'gaze_stability': gaze_3d_data.get('stability_score', 0.0) if gaze_3d_data else 0.0,
                'avg_gaze_depth': gaze_3d_stats.get('avg_depth', 0.0),
                'avg_gaze_confidence': gaze_3d_stats.get('avg_confidence', 0.0)
            }
            
            self.analytics.log_real_time_analytics(analytics_data)
            
        except Exception as e:
            self.logger.error(f"Real-time analytics logging hatası: {e}")
    
    def _render_frame(self, frame, gaze_3d_data: Dict[str, Any], detections: List[Dict[str, Any]], 
                     collision_events: List[Any], attention_analysis: Optional[Dict[str, Any]]):
        """Frame'i render et"""
        try:
            if not self.visualization_manager:
                return frame
            
            return self.visualization_manager.render_frame(
                frame, gaze_3d_data, detections, collision_events, attention_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Frame rendering hatası: {e}")
            return frame
    
    def get_readable_analytics_summary(self) -> Dict[str, Any]:
        """Okunabilir analytics özetini döndür"""
        try:
            if not self.analytics:
                return {}
            
            # Analytics'ten okunabilir özet al
            readable_summary = self.analytics.get_readable_analytics_summary()
            
            # Frame processor istatistikleri ekle
            processing_stats = self.get_processing_statistics()
            
            readable_summary.update({
                'processing_performance': f"{processing_stats['avg_processing_time']*1000:.1f}ms ortalama işlem süresi",
                'detection_activity': f"{processing_stats['detection_rate']:.2f} tespit/frame oranı",
                'collision_activity': f"{processing_stats['collision_rate']:.3f} çakışma/frame oranı",
                'total_frames_processed': f"{processing_stats['frame_count']:,} frame işlendi"
            })
            
            return readable_summary
            
        except Exception as e:
            self.logger.error(f"Readable analytics summary hatası: {e}")
            return {}
    
    def log_session_end_summary(self):
        """Session bitiminde özet logla"""
        try:
            if not self.analytics:
                return
            
            # Final session summary
            self.analytics.log_session_summary()
            
            # Console'da da göster
            readable_summary = self.get_readable_analytics_summary()
            
            print("\n" + "="*60)
            print("📊 OTURUM ÖZETİ - İNSAN OKUNAKLI")
            print("="*60)
            
            for key, value in readable_summary.items():
                if isinstance(value, str):
                    print(f"• {value}")
                elif isinstance(value, list) and key == 'most_interesting_objects':
                    if value:
                        print(f"• En çok bakılan objeler: {', '.join(value)}")
            
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Session end summary logging hatası: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """İşleme istatistiklerini döndür"""
        return {
            'frame_count': self.frame_count,
            'total_detections': self.detection_stats['total_detections'],
            'total_collisions': self.detection_stats['total_collisions'],
            'last_collision_time': self.detection_stats['last_collision_time'],
            'avg_processing_time': sum(self.processing_times[-100:]) / len(self.processing_times[-100:]) if self.processing_times else 0,
            'processing_time_history': self.processing_times[-10:],  # Son 10 frame
            'detection_rate': self.detection_stats['total_detections'] / max(1, self.frame_count),
            'collision_rate': self.detection_stats['total_collisions'] / max(1, self.frame_count),
            'current_fps': 1.0 / self.processing_times[-1] if self.processing_times else 0,
            'gaze_active': self.gaze_tracker is not None,
            'collision_tracking_active': self.gaze_3d_processor is not None
        }
    
    def reset_statistics(self):
        """İstatistikleri sıfırla"""
        self.frame_count = 0
        self.processing_times = []
        self.detection_stats = {
            'total_detections': 0,
            'total_collisions': 0,
            'last_collision_time': None
        }
        self.logger.info("Frame processing istatistikleri sıfırlandı")
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Mevcut performans bilgilerini döndür"""
        recent_times = self.processing_times[-10:] if self.processing_times else [0]
        
        return {
            'current_fps': 1.0 / recent_times[-1] if recent_times[-1] > 0 else 0,
            'avg_fps': 1.0 / (sum(recent_times) / len(recent_times)) if recent_times else 0,
            'processing_time_ms': recent_times[-1] * 1000 if recent_times else 0,
            'frame_count': self.frame_count,
            'gaze_active': self.gaze_tracker is not None,
            'collision_tracking_active': self.gaze_3d_processor is not None
        } 