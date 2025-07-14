"""
Bakış takip sınıfı
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Dict, Tuple, Any


class GazeTracker:
    """Göz takibi ve bakış yönü analizi sınıfı"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Bakış takip konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.get('max_faces', 3),
            refine_landmarks=True,
            min_detection_confidence=config.get('detection_confidence', 0.5),
            min_tracking_confidence=config.get('tracking_confidence', 0.5)
        )
        
        # Göz landmark indeksleri
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        self.logger.info("GazeTracker başlatıldı")
    
    def analyze(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Görüntüde bakış analizi yap
        
        Args:
            frame: Giriş görüntüsü
            
        Returns:
            Bakış analizi sonuçları
        """
        try:
            # RGB'ye çevir
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face mesh tespiti
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            # İlk yüzü analiz et
            face_landmarks = results.multi_face_landmarks[0]
            
            # Göz merkezlerini hesapla
            left_eye_center = self._get_eye_center(face_landmarks, self.LEFT_EYE_LANDMARKS, frame.shape)
            right_eye_center = self._get_eye_center(face_landmarks, self.RIGHT_EYE_LANDMARKS, frame.shape)
            
            # Bakış yönünü hesapla
            gaze_vector = self._calculate_gaze_vector(face_landmarks, frame.shape)
            
            # Bakış noktasını tahmin et
            gaze_point = self._estimate_gaze_point(
                left_eye_center, right_eye_center, gaze_vector, frame.shape
            )
            
            return {
                'left_eye_center': left_eye_center,
                'right_eye_center': right_eye_center,
                'gaze_vector': gaze_vector,
                'gaze_point': gaze_point,
                'face_landmarks': face_landmarks
            }
            
        except Exception as e:
            self.logger.error(f"Bakış analizi hatası: {e}")
            return None
    
    def _get_eye_center(self, landmarks, eye_indices: list, frame_shape: tuple) -> Tuple[int, int]:
        """Göz merkezini hesapla"""
        h, w = frame_shape[:2]
        
        x_coords = []
        y_coords = []
        
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x_coords.append(landmark.x * w)
            y_coords.append(landmark.y * h)
        
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        return (center_x, center_y)
    
    def _calculate_gaze_vector(self, landmarks, frame_shape: tuple) -> Tuple[float, float]:
        """Bakış vektörünü hesapla"""
        h, w = frame_shape[:2]
        
        # Burun ucu ve göz merkezleri kullanarak basit bakış vektörü
        nose_tip = landmarks.landmark[1]  # Burun ucu
        left_eye = landmarks.landmark[33]  # Sol göz içi köşe
        right_eye = landmarks.landmark[362]  # Sağ göz içi köşe
        
        # Göz merkezi
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2
        
        # Bakış vektörü (basitleştirilmiş)
        gaze_x = nose_tip.x - eye_center_x
        gaze_y = nose_tip.y - eye_center_y
        
        return (gaze_x, gaze_y)
    
    def _estimate_gaze_point(self, left_eye: Tuple[int, int], right_eye: Tuple[int, int], 
                           gaze_vector: Tuple[float, float], frame_shape: tuple) -> Tuple[int, int]:
        """Bakış noktasını tahmin et"""
        h, w = frame_shape[:2]
        
        # Gözler arası merkez
        eye_center_x = (left_eye[0] + right_eye[0]) // 2
        eye_center_y = (left_eye[1] + right_eye[1]) // 2
        
        # Bakış noktası tahmini (basit projeksiyon)
        scale_factor = 300  # Ayarlanabilir
        gaze_point_x = int(eye_center_x + gaze_vector[0] * scale_factor)
        gaze_point_y = int(eye_center_y + gaze_vector[1] * scale_factor)
        
        # Ekran sınırları içinde tut
        gaze_point_x = max(0, min(w-1, gaze_point_x))
        gaze_point_y = max(0, min(h-1, gaze_point_y))
        
        return (gaze_point_x, gaze_point_y) 