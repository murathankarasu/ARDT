"""
Gelişmiş Kafa Pozisyonu ve Bakış Yönü Analizi
"""

import numpy as np
import cv2
import math
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import mediapipe as mp


@dataclass
class HeadPose3D:
    """3D kafa pozisyonu verisi"""
    position_3d: Tuple[float, float, float]  # Kafa merkezi 3D pozisyonu
    rotation_angles: Tuple[float, float, float]  # (pitch, yaw, roll) radyan
    rotation_matrix: np.ndarray  # 3x3 rotasyon matrisi
    confidence: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary formatına çevir"""
        return {
            'position_3d': self.position_3d,
            'rotation_angles': self.rotation_angles,
            'rotation_matrix': self.rotation_matrix.tolist(),
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


@dataclass
class GazeVector3D:
    """3D bakış vektörü verisi"""
    origin_3d: Tuple[float, float, float]  # Gözün 3D pozisyonu
    direction_3d: Tuple[float, float, float]  # Normalize edilmiş yön vektörü
    target_point_3d: Optional[Tuple[float, float, float]]  # Hedef nokta (eğer hesaplanabilirse)
    confidence: float
    left_eye_3d: Tuple[float, float, float]
    right_eye_3d: Tuple[float, float, float]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary formatına çevir"""
        return {
            'origin_3d': self.origin_3d,
            'direction_3d': self.direction_3d,
            'target_point_3d': self.target_point_3d,
            'confidence': self.confidence,
            'left_eye_3d': self.left_eye_3d,
            'right_eye_3d': self.right_eye_3d,
            'timestamp': self.timestamp
        }


class Enhanced3DHeadPoseEstimator:
    """Gelişmiş 3D kafa pozisyonu tahmini"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 3D Model noktaları (Face mesh referans noktaları)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Burun ucu
            (0.0, -330.0, -65.0),        # Çene
            (-225.0, 170.0, -135.0),     # Sol göz sol köşesi
            (225.0, 170.0, -135.0),      # Sağ göz sağ köşesi
            (-150.0, -150.0, -125.0),    # Sol ağız köşesi
            (150.0, -150.0, -125.0)      # Sağ ağız köşesi
        ])
        
        # Kamera kalibrasyonu
        self.camera_matrix = None
        self.dist_coeffs = None
        self._setup_camera_calibration()
        
        # Kafa pozisyonu geçmişi
        self.pose_history = []
        self.max_history = config.get('max_history', 10)
        
        # Smoothing parametreleri
        self.smoothing_factor = config.get('smoothing_factor', 0.7)
        self.previous_pose = None
        
        self.logger.info("Enhanced3DHeadPoseEstimator başlatıldı")
    
    def _setup_camera_calibration(self):
        """Kamera kalibrasyonu ayarları"""
        try:
            # Varsayılan kamera parametreleri
            focal_length = 800
            center = (640, 360)
            
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            self.dist_coeffs = np.zeros((4, 1))
            
        except Exception as e:
            self.logger.error(f"Kamera kalibrasyonu hatası: {e}")
    
    def estimate_head_pose(self, frame: np.ndarray, 
                          face_landmarks: Any = None) -> Optional[HeadPose3D]:
        """3D kafa pozisyonu tahmini"""
        try:
            # Face landmarks al
            if face_landmarks is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if not results.multi_face_landmarks:
                    return None
                    
                face_landmarks = results.multi_face_landmarks[0]
            
            # 2D landmark noktalarını al
            h, w = frame.shape[:2]
            image_points = []
            
            # Referans noktaları
            landmark_indices = [1, 152, 33, 362, 61, 291]  # Burun, çene, gözler, ağız
            
            for idx in landmark_indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                image_points.append([x, y])
            
            image_points = np.array(image_points, dtype=np.float64)
            
            # PnP problemi çözme
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points_3d,
                image_points,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if not success:
                return None
            
            # Rotasyon matrisini al
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Euler açılarını hesapla
            pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
            
            # Kafa merkezi 3D pozisyonu
            head_position_3d = (
                float(translation_vector[0][0]),
                float(translation_vector[1][0]),
                float(translation_vector[2][0])
            )
            
            # Confidence hesapla
            confidence = self._calculate_pose_confidence(face_landmarks, frame.shape)
            
            # HeadPose3D oluştur
            head_pose = HeadPose3D(
                position_3d=head_position_3d,
                rotation_angles=(pitch, yaw, roll),
                rotation_matrix=rotation_matrix,
                confidence=confidence,
                timestamp=time.time()
            )
            
            # Smoothing uygula
            if self.previous_pose is not None:
                head_pose = self._smooth_head_pose(head_pose, self.previous_pose)
            
            self.previous_pose = head_pose
            
            # Geçmişe ekle
            self.pose_history.append(head_pose)
            if len(self.pose_history) > self.max_history:
                self.pose_history.pop(0)
            
            return head_pose
            
        except Exception as e:
            self.logger.error(f"Kafa pozisyonu tahmini hatası: {e}")
            return None
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """Rotasyon matrisini Euler açılarına çevir"""
        try:
            # ZYX Euler açıları (yaw, pitch, roll)
            sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  
                          rotation_matrix[1,0] * rotation_matrix[1,0])
            
            singular = sy < 1e-6
            
            if not singular:
                x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = 0
            
            return (x, y, z)  # (pitch, yaw, roll)
            
        except Exception as e:
            self.logger.error(f"Euler açısı hesaplama hatası: {e}")
            return (0.0, 0.0, 0.0)
    
    def _calculate_pose_confidence(self, face_landmarks: Any, frame_shape: Tuple[int, int]) -> float:
        """Kafa pozisyonu güvenilirliği hesapla"""
        try:
            confidence = 0.5  # Başlangıç
            
            # Landmark kalitesi
            h, w = frame_shape[:2]
            
            # Yüz büyüklüğü kontrolü
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[362]
            eye_distance = abs(left_eye.x - right_eye.x) * w
            
            if eye_distance > 30:  # Yeterli büyüklük
                confidence += 0.3
            
            # Merkez pozisyonu kontrolü
            nose_tip = face_landmarks.landmark[1]
            nose_x = nose_tip.x * w
            nose_y = nose_tip.y * h
            
            center_x, center_y = w // 2, h // 2
            center_distance = math.sqrt((nose_x - center_x)**2 + (nose_y - center_y)**2)
            
            # Merkeze yakınlık
            if center_distance < w * 0.3:
                confidence += 0.2
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Pose confidence hesaplama hatası: {e}")
            return 0.5
    
    def _smooth_head_pose(self, current_pose: HeadPose3D, previous_pose: HeadPose3D) -> HeadPose3D:
        """Kafa pozisyonunu yumuşat"""
        try:
            alpha = self.smoothing_factor
            
            # Pozisyon smoothing
            smoothed_position = (
                previous_pose.position_3d[0] * (1 - alpha) + current_pose.position_3d[0] * alpha,
                previous_pose.position_3d[1] * (1 - alpha) + current_pose.position_3d[1] * alpha,
                previous_pose.position_3d[2] * (1 - alpha) + current_pose.position_3d[2] * alpha
            )
            
            # Açı smoothing
            smoothed_angles = (
                previous_pose.rotation_angles[0] * (1 - alpha) + current_pose.rotation_angles[0] * alpha,
                previous_pose.rotation_angles[1] * (1 - alpha) + current_pose.rotation_angles[1] * alpha,
                previous_pose.rotation_angles[2] * (1 - alpha) + current_pose.rotation_angles[2] * alpha
            )
            
            # Yeni HeadPose3D oluştur
            return HeadPose3D(
                position_3d=smoothed_position,
                rotation_angles=smoothed_angles,
                rotation_matrix=current_pose.rotation_matrix,  # Matrix smoothing karmaşık
                confidence=max(current_pose.confidence, previous_pose.confidence * 0.9),
                timestamp=current_pose.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Pose smoothing hatası: {e}")
            return current_pose


class Enhanced3DGazeEstimator:
    """Gelişmiş 3D bakış yönü tahmini"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Head pose estimator
        self.head_pose_estimator = Enhanced3DHeadPoseEstimator(config)
        
        # Göz landmark indeksleri
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Gaze geçmişi
        self.gaze_history = []
        self.max_history = config.get('max_history', 10)
        
        # Smoothing
        self.smoothing_factor = config.get('gaze_smoothing_factor', 0.6)
        self.previous_gaze = None
        
        self.logger.info("Enhanced3DGazeEstimator başlatıldı")
    
    def estimate_gaze_vector(self, frame: np.ndarray, 
                           face_landmarks: Any = None) -> Optional[GazeVector3D]:
        """3D bakış vektörü tahmini"""
        try:
            # Kafa pozisyonunu al
            head_pose = self.head_pose_estimator.estimate_head_pose(frame, face_landmarks)
            
            if head_pose is None:
                return None
            
            # Face landmarks kontrolü
            if face_landmarks is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.head_pose_estimator.face_mesh.process(rgb_frame)
                
                if not results.multi_face_landmarks:
                    return None
                    
                face_landmarks = results.multi_face_landmarks[0]
            
            # Göz pozisyonlarını hesapla
            left_eye_3d = self._calculate_eye_position_3d(face_landmarks, self.LEFT_EYE_LANDMARKS, 
                                                         frame.shape, head_pose)
            right_eye_3d = self._calculate_eye_position_3d(face_landmarks, self.RIGHT_EYE_LANDMARKS, 
                                                          frame.shape, head_pose)
            
            # Gaze origin (gözler arası merkez)
            gaze_origin = (
                (left_eye_3d[0] + right_eye_3d[0]) / 2,
                (left_eye_3d[1] + right_eye_3d[1]) / 2,
                (left_eye_3d[2] + right_eye_3d[2]) / 2
            )
            
            # Gaze direction hesapla
            gaze_direction = self._calculate_gaze_direction(
                face_landmarks, head_pose, frame.shape
            )
            
            # Hedef nokta hesapla (isteğe bağlı)
            target_point = self._calculate_target_point(gaze_origin, gaze_direction)
            
            # Confidence hesapla
            confidence = self._calculate_gaze_confidence(face_landmarks, head_pose)
            
            # GazeVector3D oluştur
            gaze_vector = GazeVector3D(
                origin_3d=gaze_origin,
                direction_3d=gaze_direction,
                target_point_3d=target_point,
                confidence=confidence,
                left_eye_3d=left_eye_3d,
                right_eye_3d=right_eye_3d,
                timestamp=time.time()
            )
            
            # Smoothing uygula
            if self.previous_gaze is not None:
                gaze_vector = self._smooth_gaze_vector(gaze_vector, self.previous_gaze)
            
            self.previous_gaze = gaze_vector
            
            # Geçmişe ekle
            self.gaze_history.append(gaze_vector)
            if len(self.gaze_history) > self.max_history:
                self.gaze_history.pop(0)
            
            return gaze_vector
            
        except Exception as e:
            self.logger.error(f"Gaze vector tahmini hatası: {e}")
            return None
    
    def _calculate_eye_position_3d(self, face_landmarks: Any, eye_landmarks: List[int], 
                                  frame_shape: Tuple[int, int], head_pose: HeadPose3D) -> Tuple[float, float, float]:
        """Göz 3D pozisyonunu hesapla"""
        try:
            h, w = frame_shape[:2]
            
            # Göz merkezi hesapla
            x_coords = []
            y_coords = []
            
            for idx in eye_landmarks:
                landmark = face_landmarks.landmark[idx]
                x_coords.append(landmark.x * w)
                y_coords.append(landmark.y * h)
            
            eye_center_2d = (np.mean(x_coords), np.mean(y_coords))
            
            # 2D'den 3D'ye dönüşüm
            # Kafa pozisyonu ve kamera kalibrasyonu kullanarak
            camera_matrix = self.head_pose_estimator.camera_matrix
            
            # Basit back-projection
            focal_length = camera_matrix[0, 0]
            center_x = camera_matrix[0, 2]
            center_y = camera_matrix[1, 2]
            
            # Kafa pozisyonuna göre derinlik tahmini
            estimated_depth = abs(head_pose.position_3d[2]) + 20  # Gözler kafanın biraz önünde
            
            # 3D pozisyon hesapla
            eye_3d_x = (eye_center_2d[0] - center_x) * estimated_depth / focal_length
            eye_3d_y = (eye_center_2d[1] - center_y) * estimated_depth / focal_length
            eye_3d_z = estimated_depth
            
            return (eye_3d_x, eye_3d_y, eye_3d_z)
            
        except Exception as e:
            self.logger.error(f"Göz 3D pozisyonu hesaplama hatası: {e}")
            return (0.0, 0.0, 500.0)
    
    def _calculate_gaze_direction(self, face_landmarks: Any, head_pose: HeadPose3D, 
                                frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """Gaze direction hesapla"""
        try:
            # Kafa yönelimini temel al
            pitch, yaw, roll = head_pose.rotation_angles
            
            # Göz hareketlerini analiz et
            h, w = frame_shape[:2]
            
            # Göz bebekleri pozisyonu (basit yaklaşım)
            left_eye_center = face_landmarks.landmark[468]  # Sol göz merkezi
            right_eye_center = face_landmarks.landmark[473]  # Sağ göz merkezi
            
            # Gözler arası merkez
            eye_center_x = (left_eye_center.x + right_eye_center.x) / 2
            eye_center_y = (left_eye_center.y + right_eye_center.y) / 2
            
            # Burun ucuna göre göz pozisyonu
            nose_tip = face_landmarks.landmark[1]
            
            # Göz-burun vektörü (gaze offset)
            gaze_offset_x = eye_center_x - nose_tip.x
            gaze_offset_y = eye_center_y - nose_tip.y
            
            # Kafa rotasyonu ile birleştir
            base_direction = np.array([
                math.sin(yaw) * math.cos(pitch),
                -math.sin(pitch),
                math.cos(yaw) * math.cos(pitch)
            ])
            
            # Göz offset'i ekle
            gaze_adjustment = np.array([
                gaze_offset_x * 0.5,
                gaze_offset_y * 0.5,
                0.0
            ])
            
            # Final gaze direction
            gaze_direction = base_direction + gaze_adjustment
            
            # Normalize et
            gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
            
            return tuple(gaze_direction)
            
        except Exception as e:
            self.logger.error(f"Gaze direction hesaplama hatası: {e}")
            return (0.0, 0.0, 1.0)
    
    def _calculate_target_point(self, origin: Tuple[float, float, float], 
                               direction: Tuple[float, float, float], 
                               distance: float = 1000.0) -> Tuple[float, float, float]:
        """Hedef nokta hesapla"""
        try:
            target_x = origin[0] + direction[0] * distance
            target_y = origin[1] + direction[1] * distance
            target_z = origin[2] + direction[2] * distance
            
            return (target_x, target_y, target_z)
            
        except Exception as e:
            self.logger.error(f"Hedef nokta hesaplama hatası: {e}")
            return None
    
    def _calculate_gaze_confidence(self, face_landmarks: Any, head_pose: HeadPose3D) -> float:
        """Gaze confidence hesapla"""
        try:
            confidence = head_pose.confidence * 0.7  # Kafa pozisyonu güvenilirliği
            
            # Göz açıklığı kontrolü
            left_eye_openness = self._calculate_eye_openness(face_landmarks, self.LEFT_EYE_LANDMARKS)
            right_eye_openness = self._calculate_eye_openness(face_landmarks, self.RIGHT_EYE_LANDMARKS)
            
            eye_openness = (left_eye_openness + right_eye_openness) / 2
            confidence += eye_openness * 0.3
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Gaze confidence hesaplama hatası: {e}")
            return 0.5
    
    def _calculate_eye_openness(self, face_landmarks: Any, eye_landmarks: List[int]) -> float:
        """Göz açıklığı hesapla"""
        try:
            # Göz landmark'larından dikey mesafe
            if len(eye_landmarks) < 6:
                return 0.5
            
            # Üst ve alt göz kapağı noktaları
            upper_points = eye_landmarks[:len(eye_landmarks)//2]
            lower_points = eye_landmarks[len(eye_landmarks)//2:]
            
            # Ortalama dikey mesafe
            total_distance = 0
            count = 0
            
            for i in range(min(len(upper_points), len(lower_points))):
                upper = face_landmarks.landmark[upper_points[i]]
                lower = face_landmarks.landmark[lower_points[i]]
                
                distance = abs(upper.y - lower.y)
                total_distance += distance
                count += 1
            
            if count > 0:
                avg_distance = total_distance / count
                # Normalize et (0-1 aralığında)
                return min(1.0, avg_distance * 10)  # Calibration factor
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Göz açıklığı hesaplama hatası: {e}")
            return 0.5
    
    def _smooth_gaze_vector(self, current_gaze: GazeVector3D, previous_gaze: GazeVector3D) -> GazeVector3D:
        """Gaze vector smoothing"""
        try:
            alpha = self.smoothing_factor
            
            # Origin smoothing
            smoothed_origin = (
                previous_gaze.origin_3d[0] * (1 - alpha) + current_gaze.origin_3d[0] * alpha,
                previous_gaze.origin_3d[1] * (1 - alpha) + current_gaze.origin_3d[1] * alpha,
                previous_gaze.origin_3d[2] * (1 - alpha) + current_gaze.origin_3d[2] * alpha
            )
            
            # Direction smoothing
            smoothed_direction = (
                previous_gaze.direction_3d[0] * (1 - alpha) + current_gaze.direction_3d[0] * alpha,
                previous_gaze.direction_3d[1] * (1 - alpha) + current_gaze.direction_3d[1] * alpha,
                previous_gaze.direction_3d[2] * (1 - alpha) + current_gaze.direction_3d[2] * alpha
            )
            
            # Normalize direction
            direction_norm = math.sqrt(sum(x**2 for x in smoothed_direction))
            if direction_norm > 0:
                smoothed_direction = tuple(x / direction_norm for x in smoothed_direction)
            
            # Target point güncelle
            target_point = self._calculate_target_point(smoothed_origin, smoothed_direction) if current_gaze.target_point_3d else None
            
            # Yeni GazeVector3D oluştur
            return GazeVector3D(
                origin_3d=smoothed_origin,
                direction_3d=smoothed_direction,
                target_point_3d=target_point,
                confidence=max(current_gaze.confidence, previous_gaze.confidence * 0.9),
                left_eye_3d=current_gaze.left_eye_3d,
                right_eye_3d=current_gaze.right_eye_3d,
                timestamp=current_gaze.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Gaze vector smoothing hatası: {e}")
            return current_gaze
    
    def get_gaze_stability_score(self) -> float:
        """Gaze stability skoru"""
        try:
            if len(self.gaze_history) < 3:
                return 0.5
            
            # Son birkaç gaze direction'ın tutarlılığını kontrol et
            recent_directions = [gaze.direction_3d for gaze in self.gaze_history[-5:]]
            
            if len(recent_directions) < 2:
                return 0.5
            
            # Ortalama sapma hesapla
            avg_direction = np.mean(recent_directions, axis=0)
            
            total_deviation = 0
            for direction in recent_directions:
                deviation = np.linalg.norm(np.array(direction) - avg_direction)
                total_deviation += deviation
            
            avg_deviation = total_deviation / len(recent_directions)
            
            # Stability score (düşük sapma = yüksek stability)
            stability = max(0.0, 1.0 - avg_deviation * 2)
            
            return stability
            
        except Exception as e:
            self.logger.error(f"Gaze stability hesaplama hatası: {e}")
            return 0.5 