"""
ARDS Uygulama Başlatıcısı
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from core.adaptive_camera import AdaptiveCamera, PerformanceMode
from detection.object_detector import ObjectDetector
from detection.ai_classifier import AIClassifier
from attention.gaze_tracker import GazeTracker
from attention.attention_analyzer import AttentionAnalyzer
from attention.gaze_3d_processor import Gaze3DProcessor
from analytics.firestore_manager import FirestoreAnalytics
from visualization.visualization_manager import VisualizationManager
from utils.config_loader import ConfigLoader
from utils.system_monitor import SystemMonitor


class ARDSInitializer:
    """ARDS sistem bileşenlerini başlatır"""
    
    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.config = None
        
        # Sistem monitoring
        self.system_monitor = SystemMonitor(monitoring_interval=2.0)
        
        # Bileşenler
        self.camera_manager = None
        self.object_detector = None
        self.ai_classifier = None
        self.gaze_tracker = None
        self.attention_analyzer = None
        self.gaze_3d_processor = None
        self.visualization_manager = None
        self.analytics = None
    
    def initialize_all_components(self) -> Dict[str, Any]:
        """Tüm sistem bileşenlerini başlat"""
        print("🔧 Sistem bileşenleri yükleniyor...")
        
        # Konfigürasyon yükleme
        self._load_configuration()
        
        # Çıktı klasörü
        self._create_output_directory()
        
        # Analytics sistemi
        self._initialize_analytics()
        
        # Kamera sistemi
        self._initialize_camera_system()
        
        # AI ve tespit sistemleri
        self._initialize_detection_systems()
        
        # Gaze tracking sistemi
        self._initialize_gaze_system()
        
        # 3D Gaze processor
        self._initialize_3d_gaze_system()
        
        # Görselleştirme yöneticisi
        self._initialize_visualization_system()
        
        # Sistem monitoring
        self.system_monitor.start_monitoring()
        
        # Durum mesajları
        self._show_system_status()
        
        return {
            'config': self.config,
            'system_monitor': self.system_monitor,
            'camera_manager': self.camera_manager,
            'object_detector': self.object_detector,
            'ai_classifier': self.ai_classifier,
            'gaze_tracker': self.gaze_tracker,
            'attention_analyzer': self.attention_analyzer,
            'gaze_3d_processor': self.gaze_3d_processor,
            'visualization_manager': self.visualization_manager,
            'analytics': self.analytics
        }
    
    def _load_configuration(self):
        """Konfigürasyon yükleme"""
        try:
            self.config = ConfigLoader(self.args.config)
            if self.args.verbose:
                self.logger.info(f"Konfigürasyon yüklendi: {self.args.config}")
        except Exception as e:
            self.logger.error(f"Konfigürasyon yükleme hatası: {e}")
            raise
    
    def _create_output_directory(self):
        """Çıktı klasörünü oluştur"""
        try:
            output_path = Path(self.args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            if self.args.verbose:
                self.logger.info(f"Çıktı klasörü: {output_path}")
        except Exception as e:
            self.logger.error(f"Çıktı klasörü oluşturma hatası: {e}")
            raise
    
    def _initialize_analytics(self):
        """Analytics sistemini başlat"""
        if self.args.firebase:
            try:
                firebase_config = self.config.get('firebase', {})
                self.analytics = FirestoreAnalytics(firebase_config)
                if self.args.verbose:
                    self.logger.info("📊 Firestore Analytics etkinleştirildi")
            except Exception as e:
                if self.args.verbose:
                    self.logger.warning(f"Analytics başlatılamadı: {e}")
                self.analytics = None
        else:
            self.analytics = None
    
    def _initialize_camera_system(self):
        """Kamera sistemini başlat"""
        try:
            self.camera_manager = AdaptiveCamera(
                camera_id=self.args.camera,
                config=self.config.camera_config
            )
            
            # Performans modunu ayarla
            performance_mode = getattr(PerformanceMode, self.args.performance.upper())
            self.camera_manager.set_performance_mode(performance_mode)
            
            if self.args.verbose:
                self.logger.info(f"🎥 Kamera başlatıldı - ID: {self.args.camera}")
        except Exception as e:
            self.logger.error(f"Kamera başlatma hatası: {e}")
            raise
    
    def _initialize_detection_systems(self):
        """AI ve tespit sistemlerini başlat"""
        try:
            # AI sınıflandırıcı
            self.ai_classifier = AIClassifier(self.config.get('ai_classifier', {}))
            self.ai_classifier.optimize_for_performance()
            
            # Nesne tespit sistemi
            self.object_detector = ObjectDetector(self.config.detection_config)
            
            if self.args.verbose:
                self.logger.info("🤖 AI ve tespit sistemleri başlatıldı")
        except Exception as e:
            self.logger.error(f"Detection sistemleri başlatma hatası: {e}")
            raise
    
    def _initialize_gaze_system(self):
        """Gaze tracking sistemini başlat"""
        try:
            memory_usage = self.system_monitor.get_current_metrics().memory_percent
            
            if memory_usage < 80:
                self.gaze_tracker = GazeTracker(self.config.attention_config)
                self.attention_analyzer = AttentionAnalyzer(self.config.get('attention_analysis', {}))
                
                if self.args.verbose:
                    self.logger.info("👁️ Gaze tracking etkinleştirildi")
            else:
                self.gaze_tracker = None
                self.attention_analyzer = None
                print("⚠️ Gaze tracking devre dışı - yüksek memory kullanımı")
                
        except Exception as e:
            self.logger.error(f"Gaze sistem başlatma hatası: {e}")
            self.gaze_tracker = None
            self.attention_analyzer = None
    
    def _initialize_3d_gaze_system(self):
        """3D Gaze processing sistemini başlat"""
        try:
            self.gaze_3d_processor = Gaze3DProcessor(self.config.get('gaze_3d', {}))
            
            if self.args.verbose:
                self.logger.info("🎯 3D Gaze processor başlatıldı")
        except Exception as e:
            self.logger.error(f"3D Gaze processor başlatma hatası: {e}")
            self.gaze_3d_processor = None
    
    def _initialize_visualization_system(self):
        """Görselleştirme sistemini başlat"""
        try:
            self.visualization_manager = VisualizationManager(self.config.get('visualization', {}))
            
            if self.args.verbose:
                self.logger.info("🎨 Görselleştirme sistemi başlatıldı")
        except Exception as e:
            self.logger.error(f"Görselleştirme sistem başlatma hatası: {e}")
            self.visualization_manager = None
    
    def _show_system_status(self):
        """Sistem durumu mesajları"""
        print("✅ Sistem hazır!")
        print("📹 Kamera başlatıldı")
        print("🎯 3D Gaze tracking etkin" if self.gaze_tracker else "⚠️ Gaze tracking devre dışı")
        print("🔍 Çakışma kontrolü etkin" if self.gaze_3d_processor else "⚠️ Çakışma kontrolü devre dışı")
        print("📊 Firestore logging etkin" if self.analytics else "⚠️ Firestore devre dışı")
        
        print("\n🎮 Kontroller:")
        print("  'q' = Çıkış")
        print("  's' = Screenshot")
        print("  'r' = Gaze trail reset")
        print("  'c' = Çakışma verileri temizle")
        print("  'g' = 3D gaze görüntü aç/kapat")
        print("  'i' = Detaylı bilgi")
        print("  'h' = İnsan okunakli özet")
    
    def cleanup(self):
        """Bileşenleri temizle"""
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
        
        if self.camera_manager:
            self.camera_manager.release()
        
        if self.analytics:
            self.analytics.close()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Sistem bilgilerini döndür"""
        return {
            'gaze_tracking_active': self.gaze_tracker is not None,
            'gaze_3d_active': self.gaze_3d_processor is not None,
            'analytics_active': self.analytics is not None,
            'visualization_active': self.visualization_manager is not None,
            'camera_active': self.camera_manager is not None,
            'memory_usage': self.system_monitor.get_current_metrics().memory_percent if self.system_monitor else 0
        } 