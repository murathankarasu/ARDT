"""
3D Gaze Hesaplama Algoritmaları
"""

import math
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from .gaze_3d_types import GazePoint3D
from .gaze_smoothing import GazeSmoothing
from .depth_estimation import AdvancedDepthEstimator


class Enhanced3DGazeCalculator:
    """Gelişmiş 3D gaze hesaplama"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.depth_estimator = AdvancedDepthEstimator(config.get('depth_estimation', {}))
        self.smoothing = GazeSmoothing(
            window_size=config.get('smoothing_window', 5),
            smoothing_factor=config.get('smoothing_factor', 0.3)
        )
        self.logger = logging.getLogger(__name__)
        
    def calculate_3d_gaze(self, gaze_data: Dict[str, Any], frame_shape: Tuple[int, int]) -> Optional[GazePoint3D]:
        """3D gaze pozisyonu hesapla"""
        try:
            gaze_point = gaze_data.get('gaze_point')
            gaze_vector = gaze_data.get('gaze_vector')
            face_landmarks = gaze_data.get('face_landmarks')
            
            if not gaze_point or not gaze_vector:
                return None
            
            # Derinlik tahminleri
            face_depth = self.depth_estimator.estimate_depth_from_face(
                face_landmarks, frame_shape
            ) if face_landmarks else self.depth_estimator.baseline_depth
            
            vector_depth = self.depth_estimator.estimate_depth_from_gaze_vector(
                gaze_vector, gaze_point, frame_shape
            )
            
            # Derinlik tahmini birleştir
            estimated_depth = self.depth_estimator.fuse_depth_estimates(face_depth, vector_depth)
            
            # 3D koordinatları hesapla
            h, w = frame_shape[:2]
            
            # Gaze noktasını merkez koordinat sistemine çevir
            x_centered = gaze_point[0] - w // 2
            y_centered = gaze_point[1] - h // 2
            
            # 3D pozisyon hesaplama
            # Basit perspektif projeksiyonu tersine çevirme
            focal_length = min(w, h)
            
            x_3d = (x_centered * estimated_depth) / focal_length
            y_3d = (y_centered * estimated_depth) / focal_length
            z_3d = estimated_depth
            
            # Confidence hesaplama
            confidence = self._calculate_confidence(gaze_data, estimated_depth)
            
            # 3D gaze noktası oluştur
            gaze_3d = GazePoint3D(
                x=x_3d,
                y=y_3d,
                z=z_3d,
                confidence=confidence,
                timestamp=time.time()
            )
            
            # Yumuşatma uygula
            smoothed_gaze = self.smoothing.smooth_gaze_point(gaze_3d)
            
            return smoothed_gaze
            
        except Exception as e:
            self.logger.error(f"3D gaze hesaplama hatası: {e}")
            return None
    
    def _calculate_confidence(self, gaze_data: Dict[str, Any], depth: float) -> float:
        """Gaze confidence hesapla"""
        try:
            confidence = 0.5  # Başlangıç
            
            # Gaze vector gücü
            gaze_vector = gaze_data.get('gaze_vector', (0, 0))
            vector_strength = math.sqrt(gaze_vector[0]**2 + gaze_vector[1]**2)
            
            # Vector gücü ile confidence artır
            confidence += min(vector_strength * 0.3, 0.3)
            
            # Derinlik güvenilirliği
            depth_confidence = self.depth_estimator.get_depth_confidence(depth)
            confidence += depth_confidence * 0.2
            
            # Stabilite skoru
            stability = self.smoothing.get_stability_score()
            confidence += stability * 0.3
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Confidence hesaplama hatası: {e}")
            return 0.5
    
    def get_gaze_stability(self) -> float:
        """Gaze stabilitesi döndür"""
        return self.smoothing.get_stability_score()
    
    def reset_smoothing(self):
        """Smoothing'i sıfırla"""
        self.smoothing.reset()
    
    def get_smoothing_stats(self) -> Dict[str, Any]:
        """Smoothing istatistikleri"""
        return {
            'history_size': len(self.smoothing.gaze_history),
            'stability_score': self.get_gaze_stability(),
            'has_smoothed_gaze': self.smoothing.smoothed_gaze is not None
        } 