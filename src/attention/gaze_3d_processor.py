"""
3D Gaze Processing Ana Modülü - Modüler Yapı
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import deque

# Modüler bileşenler
from .gaze_3d_types import GazePoint3D, GazeCollisionEvent
from .gaze_calculation import Enhanced3DGazeCalculator
from .collision_detection import CollisionManager


class Gaze3DProcessor:
    """Ana 3D gaze processing sınıfı - Modüler yapı"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 3D gaze konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Modüler bileşenler
        self.gaze_calculator = Enhanced3DGazeCalculator(config)
        self.collision_manager = CollisionManager(config)
        
        # Gaze geçmişi
        self.max_gaze_history = config.get('max_gaze_history', 20)
        self.gaze_history = deque(maxlen=self.max_gaze_history)
        
        # Performans tracking
        self.last_update_time = time.time()
        self.frame_count = 0
        
        # İstatistikler
        self.stats = {
            'total_gaze_points': 0,
            'avg_depth': 0.0,
            'avg_confidence': 0.0,
            'stability_score': 0.0,
            'depth_variance': 0.0
        }
        
        self.logger.info("Modüler Gaze3DProcessor başlatıldı")
    
    def process_gaze_data(self, gaze_data: Dict[str, Any], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Ana gaze processing fonksiyonu
        
        Args:
            gaze_data: Gaze tracking verisi
            frame_shape: Frame boyutları
            
        Returns:
            3D gaze işleme sonuçları
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        self.frame_count += 1
        
        # Varsayılan sonuç
        result = {
            'gaze_3d_point': None,
            'gaze_history': [],
            'collision_events': [],
            'active_collisions': [],
            'depth_estimate': 0.0,
            'confidence': 0.0,
            'stability_score': 0.0,
            'processing_stats': {}
        }
        
        # Gaze verilerini kontrol et
        if not gaze_data or not gaze_data.get('gaze_point'):
            return result
        
        # 3D gaze hesapla
        gaze_3d = self.gaze_calculator.calculate_3d_gaze(gaze_data, frame_shape)
        
        if gaze_3d is None:
            return result
        
        # Gaze geçmişine ekle
        self.gaze_history.append(gaze_3d)
        
        # İstatistikleri güncelle
        self._update_statistics(gaze_3d)
        
        # Sonuçları hazırla
        result.update({
            'gaze_3d_point': gaze_3d,
            'gaze_history': list(self.gaze_history),
            'depth_estimate': gaze_3d.z,
            'confidence': gaze_3d.confidence,
            'stability_score': self.gaze_calculator.get_gaze_stability(),
            'processing_stats': self._get_processing_stats()
        })
        
        return result
    
    def check_collisions(self, gaze_3d_data: Dict[str, Any], detected_objects: List[Dict[str, Any]],
                        frame_shape: Tuple[int, int]) -> List[GazeCollisionEvent]:
        """
        Çakışma kontrolü
        
        Args:
            gaze_3d_data: 3D gaze verisi
            detected_objects: Tespit edilen objeler
            frame_shape: Frame boyutları
            
        Returns:
            Çakışma olayları listesi
        """
        gaze_3d_point = gaze_3d_data.get('gaze_3d_point')
        
        if gaze_3d_point is None:
            return []
        
        # Collision manager ile çakışmaları işle
        return self.collision_manager.process_collisions(gaze_3d_point, detected_objects, frame_shape)
    
    def get_collision_summary(self) -> Dict[str, Any]:
        """Çakışma özeti"""
        base_summary = self.collision_manager.get_collision_summary()
        
        # Processing stats ekle
        base_summary['processing_stats'] = self._get_processing_stats()
        
        return base_summary
    
    def reset_collision_data(self):
        """Çakışma ve gaze verilerini sıfırla"""
        self.collision_manager.reset_collision_data()
        self.gaze_history.clear()
        self.gaze_calculator.reset_smoothing()
        
        # İstatistikleri sıfırla
        self.stats = {
            'total_gaze_points': 0,
            'avg_depth': 0.0,
            'avg_confidence': 0.0,
            'stability_score': 0.0,
            'depth_variance': 0.0
        }
        
        self.logger.info("Gaze ve çakışma verileri sıfırlandı")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Detaylı performans istatistikleri"""
        current_time = time.time()
        uptime = current_time - (self.last_update_time - (self.frame_count * 0.033))  # ~30 FPS
        
        # Temel performans stats
        base_stats = {
            'uptime_seconds': uptime,
            'frame_count': self.frame_count,
            'fps': self.frame_count / max(uptime, 1),
            'gaze_history_size': len(self.gaze_history),
            'avg_depth': self.stats['avg_depth'],
            'avg_confidence': self.stats['avg_confidence'],
            'stability_score': self.stats['stability_score'],
            'depth_variance': self.stats['depth_variance']
        }
        
        # Collision manager stats
        collision_stats = self.collision_manager.get_performance_stats()
        
        # Gaze calculator stats
        smoothing_stats = self.gaze_calculator.get_smoothing_stats()
        
        # Hepsini birleştir
        base_stats.update({
            'collision_stats': collision_stats,
            'smoothing_stats': smoothing_stats,
            'memory_usage': {
                'gaze_history': len(self.gaze_history),
                'collision_history': collision_stats['collision_history_size'],
                'active_collisions': collision_stats['active_collisions']
            }
        })
        
        return base_stats
    
    def _update_statistics(self, gaze_3d: GazePoint3D):
        """İstatistikleri güncelle"""
        self.stats['total_gaze_points'] += 1
        
        # Exponential moving average
        alpha = 0.05  # Smoothing factor
        
        # Ortalama derinlik
        self.stats['avg_depth'] = (self.stats['avg_depth'] * (1 - alpha)) + (gaze_3d.z * alpha)
        
        # Ortalama confidence
        self.stats['avg_confidence'] = (self.stats['avg_confidence'] * (1 - alpha)) + (gaze_3d.confidence * alpha)
        
        # Stabilite skoru
        self.stats['stability_score'] = self.gaze_calculator.get_gaze_stability()
        
        # Derinlik varyansı
        if len(self.gaze_history) > 1:
            depths = [g.z for g in self.gaze_history]
            self.stats['depth_variance'] = np.var(depths)
    
    def _get_processing_stats(self) -> Dict[str, Any]:
        """Processing istatistikleri"""
        return {
            'frame_count': self.frame_count,
            'gaze_history_size': len(self.gaze_history),
            'total_gaze_points': self.stats['total_gaze_points'],
            'avg_depth': self.stats['avg_depth'],
            'avg_confidence': self.stats['avg_confidence'],
            'stability_score': self.stats['stability_score'],
            'depth_variance': self.stats['depth_variance']
        }
    
    def get_gaze_stability(self) -> float:
        """Gaze stabilitesi"""
        return self.gaze_calculator.get_gaze_stability()
    
    def get_current_gaze_point(self) -> Optional[GazePoint3D]:
        """Mevcut gaze noktası"""
        return self.gaze_history[-1] if self.gaze_history else None
    
    def get_gaze_history(self, limit: int = None) -> List[GazePoint3D]:
        """Gaze geçmişi"""
        if limit is None:
            return list(self.gaze_history)
        else:
            return list(self.gaze_history)[-limit:]
    
    def configure_collision_threshold(self, threshold: float):
        """Çakışma threshold'ını ayarla"""
        self.collision_manager.collision_detector.collision_threshold = threshold
        self.logger.info(f"Çakışma threshold'ı {threshold} olarak ayarlandı")
    
    def configure_smoothing(self, smoothing_factor: float):
        """Smoothing faktörünü ayarla"""
        self.gaze_calculator.smoothing.smoothing_factor = smoothing_factor
        self.logger.info(f"Smoothing faktörü {smoothing_factor} olarak ayarlandı")
    
    def get_module_info(self) -> Dict[str, Any]:
        """Modül bilgileri"""
        return {
            'version': '2.0.0',
            'modules': {
                'gaze_calculator': 'Enhanced3DGazeCalculator',
                'collision_manager': 'CollisionManager',
                'depth_estimator': 'AdvancedDepthEstimator',
                'smoothing': 'GazeSmoothing'
            },
            'configuration': {
                'max_gaze_history': self.max_gaze_history,
                'collision_threshold': self.collision_manager.collision_detector.collision_threshold,
                'smoothing_factor': self.gaze_calculator.smoothing.smoothing_factor
            }
        } 