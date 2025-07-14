"""
Firestore Analitik Yöneticisi
"""

import logging
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
from queue import Queue

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("Firebase modülleri bulunamadı. pip install firebase-admin")


def convert_numpy_types(obj):
    """
    Numpy türlerini Python native türlere dönüştür
    Recursive olarak tüm data yapılarını temizler
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@dataclass
class DetectionEvent:
    """Tespit eventi"""
    timestamp: datetime
    detection_type: str
    object_class: str
    confidence: float
    bbox: List[int]
    scene_context: str
    camera_settings: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir"""
        data = {
            'timestamp': self.timestamp.isoformat(),
            'detection_type': self.detection_type,
            'object_class': self.object_class,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'scene_context': self.scene_context,
            'camera_settings': self.camera_settings
        }
        # Güvenlik için son kez temizle
        return convert_numpy_types(data)


@dataclass
class AttentionEvent:
    """Dikkat eventi"""
    timestamp: datetime
    gaze_point: List[int]
    focused_objects: List[Dict[str, Any]]
    attention_distribution: Dict[str, float]
    interest_level: float
    gaze_stability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir"""
        data = {
            'timestamp': self.timestamp.isoformat(),
            'gaze_point': self.gaze_point,
            'focused_objects': self.focused_objects,
            'attention_distribution': self.attention_distribution,
            'interest_level': self.interest_level,
            'gaze_stability': self.gaze_stability
        }
        # Güvenlik için son kez temizle
        return convert_numpy_types(data)


@dataclass
class SessionMetrics:
    """Oturum metrikleri"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_detections: int
    detection_types: Dict[str, int]
    avg_confidence: float
    scene_changes: int
    active_duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_detections': self.total_detections,
            'detection_types': self.detection_types,
            'avg_confidence': self.avg_confidence,
            'scene_changes': self.scene_changes,
            'active_duration': self.active_duration
        }


class FirestoreAnalytics:
    """Firestore tabanlı analitik sistem"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Firebase konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not FIREBASE_AVAILABLE:
            self.logger.error("Firebase modülleri yüklü değil")
            self.db = None
            return
        
        # Firebase bağlantısı
        self.db = None
        self._initialize_firebase()
        
        # Oturum bilgileri
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now(timezone.utc)
        self.session_metrics = SessionMetrics(
            session_id=self.session_id,
            start_time=self.session_start,
            end_time=None,
            total_detections=0,
            detection_types={},
            avg_confidence=0.0,
            scene_changes=0,
            active_duration=0.0
        )
        
        # Async yazım için queue
        self.write_queue = Queue()
        self.writer_thread = None
        self.running = False
        
        # Cache
        self.detection_cache = []
        self.attention_cache = []
        
        if self.db:
            self._start_writer_thread()
            self.logger.info(f"Firestore Analytics başlatıldı - Session: {self.session_id}")
    
    def _initialize_firebase(self):
        """Firebase'i başlat"""
        try:
            # Firebase Admin SDK
            cred_path = self.config.get('credentials_path')
            project_id = self.config.get('project_id')
            
            if cred_path:
                # Servis hesabı anahtarı ile
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': project_id
                })
            else:
                # Varsayılan credentials ile
                firebase_admin.initialize_app()
            
            self.db = firestore.client()
            self.logger.info("Firestore bağlantısı başarılı")
            
        except Exception as e:
            self.logger.error(f"Firebase başlatma hatası: {e}")
            self.db = None
    
    def _start_writer_thread(self):
        """Async yazıcı thread'i başlat"""
        self.running = True
        self.writer_thread = threading.Thread(target=self._writer_loop)
        self.writer_thread.daemon = True
        self.writer_thread.start()
    
    def _writer_loop(self):
        """Async yazım döngüsü"""
        while self.running:
            try:
                if not self.write_queue.empty():
                    write_task = self.write_queue.get()
                    self._execute_write_task(write_task)
                    self.write_queue.task_done()
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Writer loop hatası: {e}")
    
    def _execute_write_task(self, task: Dict[str, Any]):
        """Yazım görevini gerçekleştir"""
        try:
            collection = task['collection']
            document_id = task.get('document_id')
            data = task['data']
            operation = task.get('operation', 'add')
            
            # CRITICAL FIX: Final safeguard - tüm veriyi temizle
            clean_data = convert_numpy_types(data)
            
            if operation == 'add':
                if document_id:
                    self.db.collection(collection).document(document_id).set(clean_data)
                else:
                    self.db.collection(collection).add(clean_data)
            elif operation == 'update':
                self.db.collection(collection).document(document_id).update(clean_data)
            
        except Exception as e:
            self.logger.error(f"Firestore yazım hatası: {e}")
    
    def log_detection(self, detection_data: Dict[str, Any], 
                     camera_settings: Dict[str, Any] = None):
        """Tespit eventini kaydet"""
        if not self.db:
            return
        
        try:
            # CRITICAL FIX: Tüm numpy türlerini dönüştür
            clean_detection_data = convert_numpy_types(detection_data)
            clean_camera_settings = convert_numpy_types(camera_settings or {})
            
            event = DetectionEvent(
                timestamp=datetime.now(timezone.utc),
                detection_type=clean_detection_data.get('type', 'unknown'),
                object_class=clean_detection_data.get('label', 'unknown'),
                confidence=clean_detection_data.get('confidence', 0.0),
                bbox=clean_detection_data.get('bbox', [0, 0, 0, 0]),
                scene_context=clean_detection_data.get('scene_context', 'unknown'),
                camera_settings=clean_camera_settings
            )
            
            # Cache'e ekle
            self.detection_cache.append(event)
            
            # Session metrikleri güncelle
            self._update_session_metrics(event)
            
            # Async yazım için queue'ya ekle
            self.write_queue.put({
                'collection': f'sessions/{self.session_id}/detections',
                'data': event.to_dict(),
                'operation': 'add'
            })
            
            # Batch processing
            if len(self.detection_cache) >= 10:
                self._batch_process_detections()
            
        except Exception as e:
            self.logger.error(f"Detection log hatası: {e}")
    
    def log_attention(self, attention_data: Dict[str, Any]):
        """Dikkat eventini kaydet"""
        if not self.db:
            return
        
        try:
            # CRITICAL FIX: Tüm numpy türlerini dönüştür
            clean_attention_data = convert_numpy_types(attention_data)
            
            event = AttentionEvent(
                timestamp=datetime.now(timezone.utc),
                gaze_point=clean_attention_data.get('gaze_point', [0, 0]),
                focused_objects=clean_attention_data.get('focused_objects', []),
                attention_distribution=clean_attention_data.get('attention_distribution', {}),
                interest_level=clean_attention_data.get('interest_level', 0.0),
                gaze_stability=clean_attention_data.get('gaze_stability', 0.0)
            )
            
            # Cache'e ekle
            self.attention_cache.append(event)
            
            # Async yazım
            self.write_queue.put({
                'collection': f'sessions/{self.session_id}/attention',
                'data': event.to_dict(),
                'operation': 'add'
            })
            
            # Batch processing
            if len(self.attention_cache) >= 5:
                self._batch_process_attention()
            
        except Exception as e:
            self.logger.error(f"Attention log hatası: {e}")
    
    def log_session_metrics(self):
        """Oturum metriklerini kaydet"""
        if not self.db:
            return
        
        try:
            self.session_metrics.end_time = datetime.now(timezone.utc)
            self.session_metrics.active_duration = (
                self.session_metrics.end_time - self.session_metrics.start_time
            ).total_seconds()
            
            # Session'ı güncelle
            self.write_queue.put({
                'collection': 'sessions',
                'document_id': self.session_id,
                'data': self.session_metrics.to_dict(),
                'operation': 'update'
            })
            
        except Exception as e:
            self.logger.error(f"Session metrics log hatası: {e}")
    
    def _update_session_metrics(self, event: DetectionEvent):
        """Session metriklerini güncelle"""
        self.session_metrics.total_detections += 1
        
        # Detection türlerini say
        det_type = event.detection_type
        if det_type not in self.session_metrics.detection_types:
            self.session_metrics.detection_types[det_type] = 0
        self.session_metrics.detection_types[det_type] += 1
        
        # Ortalama güven skorunu güncelle
        total_conf = (self.session_metrics.avg_confidence * 
                     (self.session_metrics.total_detections - 1) + event.confidence)
        self.session_metrics.avg_confidence = total_conf / self.session_metrics.total_detections
    
    def _batch_process_detections(self):
        """Detection'ları batch olarak işle"""
        if not self.detection_cache:
            return
        
        try:
            # Analiz sonuçları
            analysis = self._analyze_detection_batch(self.detection_cache)
            
            # Analiz sonuçlarını kaydet
            self.write_queue.put({
                'collection': f'sessions/{self.session_id}/analysis',
                'data': {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'type': 'detection_batch',
                    'analysis': analysis,
                    'batch_size': len(self.detection_cache)
                },
                'operation': 'add'
            })
            
            # Cache'i temizle
            self.detection_cache = []
            
        except Exception as e:
            self.logger.error(f"Batch detection processing hatası: {e}")
    
    def _batch_process_attention(self):
        """Attention verilerini batch olarak işle"""
        if not self.attention_cache:
            return
        
        try:
            # Analiz sonuçları
            analysis = self._analyze_attention_batch(self.attention_cache)
            
            # Analiz sonuçlarını kaydet
            self.write_queue.put({
                'collection': f'sessions/{self.session_id}/analysis',
                'data': {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'type': 'attention_batch',
                    'analysis': analysis,
                    'batch_size': len(self.attention_cache)
                },
                'operation': 'add'
            })
            
            # Cache'i temizle
            self.attention_cache = []
            
        except Exception as e:
            self.logger.error(f"Batch attention processing hatası: {e}")
    
    def _analyze_detection_batch(self, detections: List[DetectionEvent]) -> Dict[str, Any]:
        """Detection batch analizi"""
        if not detections:
            return {}
        
        # İstatistikler
        confidences = [d.confidence for d in detections]
        object_classes = [d.object_class for d in detections]
        
        # Sınıf dağılımı
        class_counts = {}
        for cls in object_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        return {
            'total_detections': len(detections),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'class_distribution': class_counts,
            'most_detected_class': max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None
        }
    
    def _analyze_attention_batch(self, attention_events: List[AttentionEvent]) -> Dict[str, Any]:
        """Attention batch analizi"""
        if not attention_events:
            return {}
        
        # İstatistikler
        interest_levels = [a.interest_level for a in attention_events]
        stability_scores = [a.gaze_stability for a in attention_events]
        
        # Dikkat dağılımı analizi
        combined_distribution = {}
        for event in attention_events:
            for category, value in event.attention_distribution.items():
                if category not in combined_distribution:
                    combined_distribution[category] = []
                combined_distribution[category].append(value)
        
        # Ortalama dağılım
        avg_distribution = {
            category: sum(values) / len(values)
            for category, values in combined_distribution.items()
        }
        
        return {
            'total_events': len(attention_events),
            'avg_interest_level': sum(interest_levels) / len(interest_levels),
            'avg_gaze_stability': sum(stability_scores) / len(stability_scores),
            'attention_distribution': avg_distribution,
            'peak_interest': max(interest_levels),
            'most_focused_category': max(avg_distribution.items(), key=lambda x: x[1])[0] if avg_distribution else None
        }
    
    def _generate_session_id(self) -> str:
        """Benzersiz session ID oluştur"""
        timestamp = int(time.time())
        return f"session_{timestamp}"
    
    def close(self):
        """Analitik sistemini kapat"""
        # Son verileri işle
        if self.detection_cache:
            self._batch_process_detections()
        if self.attention_cache:
            self._batch_process_attention()
        
        # Session'ı sonlandır
        self.log_session_metrics()
        
        # Thread'i durdur
        self.running = False
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5)
        
        self.logger.info(f"Analitik sistem kapatıldı - Session: {self.session_id}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Oturum özeti"""
        return {
            'session_id': self.session_id,
            'duration': (datetime.now(timezone.utc) - self.session_start).total_seconds(),
            'total_detections': self.session_metrics.total_detections,
            'avg_confidence': self.session_metrics.avg_confidence,
            'detection_types': self.session_metrics.detection_types,
            'cached_detections': len(self.detection_cache),
            'cached_attention': len(self.attention_cache),
            'queue_size': self.write_queue.qsize()
        } 