"""
Temel Firebase Analytics Logger
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
class SessionMetrics:
    """Session metrikleri"""
    total_detections: int = 0
    total_frames: int = 0
    average_fps: float = 0.0
    session_duration: float = 0.0
    detection_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseFirestoreLogger:
    """Temel Firestore logging işlemleri"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Firebase konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db = None
        self.session_id = None
        self.session_start = datetime.now(timezone.utc)
        self.session_metrics = SessionMetrics()
        
        # Write queue ve thread
        self.write_queue = Queue()
        self.write_thread = None
        self.stop_writing = False
        
        # Cache
        self.detection_cache = []
        self.attention_cache = []
        
        # Firebase'i başlat
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Firebase bağlantısını başlat"""
        if not FIREBASE_AVAILABLE:
            self.logger.warning("Firebase modülleri bulunamadı")
            return
        
        try:
            # Firebase config
            cred_path = self.config.get('credentials_path')
            if not cred_path:
                self.logger.warning("Firebase credentials yolu bulunamadı")
                return
            
            # Firebase admin SDK'yı başlat
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            
            # Firestore client
            self.db = firestore.client()
            
            # Session ID oluştur
            self.session_id = f"session_{int(time.time())}"
            
            # Write thread'i başlat
            self._start_write_thread()
            
            # Session başlangıcını kaydet
            self._log_session_start()
            
            self.logger.info(f"Firebase başlatıldı. Session ID: {self.session_id}")
            
        except Exception as e:
            self.logger.error(f"Firebase başlatma hatası: {e}")
            self.db = None
    
    def _start_write_thread(self):
        """Background write thread'i başlat"""
        self.write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.write_thread.start()
        self.logger.info("Write thread başlatıldı")
    
    def _write_worker(self):
        """Background write worker"""
        while not self.stop_writing:
            try:
                # Queue'dan task al
                task = self.write_queue.get(timeout=1.0)
                
                if task is None:  # Stop signal
                    break
                
                # Firestore'a yaz
                self._execute_write_task(task)
                
                # Task'i tamamla
                self.write_queue.task_done()
                
            except Exception as e:
                if not self.stop_writing:
                    self.logger.error(f"Write worker hatası: {e}")
                    time.sleep(0.1)
    
    def _execute_write_task(self, task: Dict[str, Any]):
        """Firestore write task'ini çalıştır"""
        try:
            collection = task['collection']
            data = task['data']
            operation = task.get('operation', 'add')
            
            if operation == 'add':
                self.db.collection(collection).add(data)
            elif operation == 'set':
                doc_id = task.get('doc_id')
                if doc_id:
                    self.db.collection(collection).document(doc_id).set(data)
                else:
                    self.db.collection(collection).add(data)
            elif operation == 'update':
                doc_id = task.get('doc_id')
                if doc_id:
                    self.db.collection(collection).document(doc_id).update(data)
                    
        except Exception as e:
            self.logger.error(f"Firestore write hatası: {e}")
    
    def _log_session_start(self):
        """Session başlangıcını kaydet"""
        if not self.db:
            return
            
        session_data = {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'status': 'active'
        }
        
        task = {
            'collection': 'sessions',
            'data': session_data,
            'operation': 'add'
        }
        self.write_queue.put(task)
    
    def log_basic_event(self, collection: str, data: Dict[str, Any]):
        """Temel event logging"""
        if not self.db:
            return
        
        try:
            # Numpy türlerini temizle
            clean_data = convert_numpy_types(data)
            
            # Timestamp ekle
            clean_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            clean_data['session_id'] = self.session_id
            
            task = {
                'collection': collection,
                'data': clean_data,
                'operation': 'add'
            }
            self.write_queue.put(task)
            
        except Exception as e:
            self.logger.error(f"Basic event logging hatası: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Session özeti döndür"""
        current_time = datetime.now(timezone.utc)
        duration = (current_time - self.session_start).total_seconds()
        
        return {
            'session_id': self.session_id,
            'duration': duration,
            'total_detections': self.session_metrics.total_detections,
            'total_frames': self.session_metrics.total_frames,
            'average_fps': self.session_metrics.average_fps,
            'queue_size': self.write_queue.qsize(),
            'cache_sizes': {
                'detection': len(self.detection_cache),
                'attention': len(self.attention_cache)
            }
        }
    
    def close(self):
        """Kaynakları temizle"""
        self.logger.info("Firebase Analytics kapatılıyor...")
        
        # Write thread'i durdur
        self.stop_writing = True
        self.write_queue.put(None)  # Stop signal
        
        if self.write_thread:
            self.write_thread.join(timeout=5.0)
        
        # Session sonunu kaydet
        self._log_session_end()
        
        self.logger.info("Firebase Analytics kapatıldı")
    
    def _log_session_end(self):
        """Session sonunu kaydet"""
        if not self.db:
            return
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.session_start).total_seconds()
        
        session_summary = {
            'session_id': self.session_id,
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'status': 'completed',
            'final_metrics': self.session_metrics.to_dict()
        }
        
        task = {
            'collection': 'session_summaries',
            'data': session_summary,
            'operation': 'add'
        }
        self.write_queue.put(task) 