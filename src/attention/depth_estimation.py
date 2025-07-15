"""
Derinlik Tahmini Algoritmaları
"""

import math
import logging
from typing import Dict, List, Any, Tuple, Optional


class AdvancedDepthEstimator:
    """Gelişmiş derinlik tahmini"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline_depth = config.get('baseline_depth', 300.0)
        self.depth_range = config.get('depth_range', (100.0, 800.0))
        self.face_width_mm = config.get('face_width_mm', 140.0)  # Ortalama yüz genişliği
        self.logger = logging.getLogger(__name__)
        
    def estimate_depth_from_face(self, face_landmarks, frame_shape: Tuple[int, int]) -> float:
        """Yüz landmark'larından derinlik tahmini"""
        try:
            h, w = frame_shape[:2]
            
            # Sol ve sağ göz köşeleri
            left_eye_corner = face_landmarks.landmark[33]  # Sol göz iç köşe
            right_eye_corner = face_landmarks.landmark[362]  # Sağ göz iç köşe
            
            # Pixel cinsinden göz arası mesafe
            eye_distance_px = abs(left_eye_corner.x - right_eye_corner.x) * w
            
            # Eğer göz arası mesafe çok küçükse baseline kullan
            if eye_distance_px < 20:
                return self.baseline_depth
            
            # Kamera focal length tahmini
            focal_length = min(w, h)
            
            # Derinlik hesaplama (similar triangles)
            # depth = (real_width * focal_length) / pixel_width
            depth = (self.face_width_mm * focal_length) / (eye_distance_px * 2)
            
            # Sınırlar içinde tut
            depth = max(self.depth_range[0], min(self.depth_range[1], depth))
            
            return depth
            
        except Exception as e:
            self.logger.error(f"Face depth estimation hatası: {e}")
            return self.baseline_depth
    
    def estimate_depth_from_gaze_vector(self, gaze_vector: Tuple[float, float], 
                                       gaze_point: Tuple[int, int], 
                                       frame_shape: Tuple[int, int]) -> float:
        """Gaze vector'den derinlik tahmini"""
        try:
            h, w = frame_shape[:2]
            
            # Vector magnitude
            vector_magnitude = math.sqrt(gaze_vector[0]**2 + gaze_vector[1]**2)
            
            # Merkeze uzaklık
            center_x, center_y = w // 2, h // 2
            distance_from_center = math.sqrt((gaze_point[0] - center_x)**2 + (gaze_point[1] - center_y)**2)
            
            # Normalize edilmiş mesafe
            normalized_distance = distance_from_center / max(w, h)
            
            # Derinlik hesaplama
            base_depth = self.baseline_depth
            
            # Vector gücü ile derinlik ayarlama
            vector_depth_factor = 1.0 + (vector_magnitude * 0.5)
            
            # Merkez uzaklığı ile derinlik ayarlama
            center_depth_factor = 1.0 + (normalized_distance * 0.3)
            
            estimated_depth = base_depth * vector_depth_factor * center_depth_factor
            
            # Sınırlar içinde tut
            return max(self.depth_range[0], min(self.depth_range[1], estimated_depth))
            
        except Exception as e:
            self.logger.error(f"Vector depth estimation hatası: {e}")
            return self.baseline_depth
    
    def estimate_depth_from_object_size(self, bbox: List[int], known_object_size: float = None) -> float:
        """Obje boyutundan derinlik tahmini"""
        try:
            if len(bbox) < 4:
                return self.baseline_depth
            
            x1, y1, x2, y2 = bbox
            object_width = x2 - x1
            object_height = y2 - y1
            
            # Ortalama obje boyutu (varsayılan)
            avg_object_size = known_object_size or 100.0  # mm
            
            # Basit derinlik tahmini
            if object_width > 0:
                # Daha büyük obje = daha yakın
                depth_factor = 50.0 / object_width  # Calibration factor
                estimated_depth = self.baseline_depth * depth_factor
                
                return max(self.depth_range[0], min(self.depth_range[1], estimated_depth))
            
            return self.baseline_depth
            
        except Exception as e:
            self.logger.error(f"Object size depth estimation hatası: {e}")
            return self.baseline_depth
    
    def fuse_depth_estimates(self, face_depth: float, vector_depth: float, 
                           object_depth: float = None, 
                           face_confidence: float = 0.7,
                           vector_confidence: float = 0.2,
                           object_confidence: float = 0.1) -> float:
        """Farklı derinlik tahminlerini birleştir"""
        try:
            # Ağırlıkları normalize et
            total_confidence = face_confidence + vector_confidence
            if object_depth is not None:
                total_confidence += object_confidence
            
            # Ağırlıklı ortalama
            fused_depth = (face_depth * face_confidence + vector_depth * vector_confidence) / total_confidence
            
            # Obje derinliği varsa ekle
            if object_depth is not None:
                fused_depth = (fused_depth * (total_confidence - object_confidence) + 
                              object_depth * object_confidence) / total_confidence
            
            # Sınırlar içinde tut
            return max(self.depth_range[0], min(self.depth_range[1], fused_depth))
            
        except Exception as e:
            self.logger.error(f"Depth fusion hatası: {e}")
            return self.baseline_depth
    
    def get_depth_confidence(self, depth: float) -> float:
        """Derinlik tahmininin güvenilirliğini hesapla"""
        try:
            # Orta aralıkta daha yüksek güven
            mid_range = (self.depth_range[0] + self.depth_range[1]) / 2
            range_width = self.depth_range[1] - self.depth_range[0]
            
            # Orta noktadan uzaklık
            distance_from_mid = abs(depth - mid_range)
            
            # Güven skoru (0-1 aralığında)
            confidence = max(0.0, 1.0 - (distance_from_mid / (range_width / 2)))
            
            return confidence
            
        except Exception:
            return 0.5  # Varsayılan güven 