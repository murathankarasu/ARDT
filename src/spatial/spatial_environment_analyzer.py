"""
3D Uzaysal Çevre Analizi Modülü
"""

import numpy as np
import cv2
import time
import math
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class SpatialObject:
    """3D uzayda konumlandırılmış nesne"""
    object_id: str
    label: str
    position_3d: Tuple[float, float, float]  # (x, y, z) kamera koordinatında
    dimensions_3d: Tuple[float, float, float]  # (width, height, depth) mm cinsinden
    confidence: float
    bbox_2d: List[int]  # [x1, y1, x2, y2]
    distance_from_camera: float
    real_world_position: Tuple[float, float, float]  # Gerçek dünya koordinatları
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary formatına çevir"""
        return {
            'object_id': self.object_id,
            'label': self.label,
            'position_3d': self.position_3d,
            'dimensions_3d': self.dimensions_3d,
            'confidence': self.confidence,
            'bbox_2d': self.bbox_2d,
            'distance_from_camera': self.distance_from_camera,
            'real_world_position': self.real_world_position,
            'timestamp': self.timestamp
        }


@dataclass
class SpatialPerson:
    """3D uzayda konumlandırılmış insan"""
    person_id: str
    head_position_3d: Tuple[float, float, float]
    body_position_3d: Tuple[float, float, float]
    head_pose: Tuple[float, float, float]  # (pitch, yaw, roll) radyan
    gaze_direction_3d: Tuple[float, float, float]  # Normalize edilmiş vektör
    distance_from_camera: float
    confidence: float
    bbox_2d: List[int]
    face_landmarks: Optional[Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary formatına çevir"""
        return {
            'person_id': self.person_id,
            'head_position_3d': self.head_position_3d,
            'body_position_3d': self.body_position_3d,
            'head_pose': self.head_pose,
            'gaze_direction_3d': self.gaze_direction_3d,
            'distance_from_camera': self.distance_from_camera,
            'confidence': self.confidence,
            'bbox_2d': self.bbox_2d,
            'timestamp': self.timestamp
        }


class CameraCalibration:
    """Kamera kalibrasyonu ve 3D projeksiyon"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Kamera parametreleri
        self.focal_length = config.get('focal_length', None)
        self.principal_point = config.get('principal_point', None)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Otomatik kalibrasyonu başlat
        self._setup_camera_parameters()
        
        # Gerçek dünya koordinat sistemi
        self.world_origin = config.get('world_origin', (0, 0, 0))
        self.world_scale = config.get('world_scale', 1.0)  # mm per unit
        
    def _setup_camera_parameters(self):
        """Kamera parametrelerini ayarla"""
        try:
            # Eğer kamera matrisi verilmemişse varsayılan değerler kullan
            if self.focal_length is None:
                # Tipik web kameraları için tahmin
                self.focal_length = 800  # pixel
                
            if self.principal_point is None:
                # Görüntü merkezini varsayılan olarak al
                self.principal_point = (640, 360)  # Varsayılan 1280x720 için
                
            # Kamera matrisi
            self.camera_matrix = np.array([
                [self.focal_length, 0, self.principal_point[0]],
                [0, self.focal_length, self.principal_point[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Distortion coefficients (varsayılan sıfır)
            self.dist_coeffs = np.zeros((4, 1))
            
            self.logger.info(f"Kamera kalibrasyonu tamamlandı: f={self.focal_length}, pp={self.principal_point}")
            
        except Exception as e:
            self.logger.error(f"Kamera kalibrasyonu hatası: {e}")
            
    def pixel_to_3d(self, pixel_coords: Tuple[int, int], depth_mm: float) -> Tuple[float, float, float]:
        """2D pixel koordinatlarını 3D'ye çevir"""
        try:
            x_px, y_px = pixel_coords
            
            # Kamera koordinatlarına çevir
            x_cam = (x_px - self.principal_point[0]) * depth_mm / self.focal_length
            y_cam = (y_px - self.principal_point[1]) * depth_mm / self.focal_length
            z_cam = depth_mm
            
            return (x_cam, y_cam, z_cam)
            
        except Exception as e:
            self.logger.error(f"Pixel to 3D çevirme hatası: {e}")
            return (0, 0, depth_mm)
    
    def camera_to_world(self, camera_coords: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Kamera koordinatlarını dünya koordinatlarına çevir"""
        try:
            # Basit transformasyon (daha karmaşık calibration için genişletilebilir)
            x_world = camera_coords[0] * self.world_scale + self.world_origin[0]
            y_world = camera_coords[1] * self.world_scale + self.world_origin[1]
            z_world = camera_coords[2] * self.world_scale + self.world_origin[2]
            
            return (x_world, y_world, z_world)
            
        except Exception as e:
            self.logger.error(f"Camera to world çevirme hatası: {e}")
            return camera_coords


class DepthEstimator3D:
    """Gelişmiş 3D derinlik tahmini"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Referans obje boyutları (mm)
        self.reference_sizes = {
            'person': {'width': 500, 'height': 1700, 'depth': 300},
            'face': {'width': 140, 'height': 180, 'depth': 120},
            'laptop': {'width': 350, 'height': 250, 'depth': 30},
            'phone': {'width': 75, 'height': 150, 'depth': 8},
            'bottle': {'width': 70, 'height': 250, 'depth': 70},
            'cup': {'width': 80, 'height': 100, 'depth': 80},
            'book': {'width': 150, 'height': 200, 'depth': 20},
            'chair': {'width': 500, 'height': 800, 'depth': 500},
            'table': {'width': 1000, 'height': 750, 'depth': 600}
        }
        
        # Derinlik tahmin parametreleri
        self.depth_range = config.get('depth_range', (200, 3000))  # mm
        self.baseline_depth = config.get('baseline_depth', 1000)  # mm
        
    def estimate_object_depth(self, bbox: List[int], label: str, 
                             additional_info: Dict[str, Any] = None) -> float:
        """Obje boyutundan derinlik tahmini"""
        try:
            if len(bbox) < 4:
                return self.baseline_depth
                
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Obje tipine göre referans boyut
            ref_size = self.reference_sizes.get(label, self.reference_sizes.get('person'))
            
            # Kamera focal length (varsayılan)
            focal_length = 800
            
            # Derinlik hesaplama (similar triangles)
            # depth = (real_size * focal_length) / pixel_size
            
            # Genişlik bazlı derinlik
            depth_from_width = (ref_size['width'] * focal_length) / bbox_width if bbox_width > 0 else self.baseline_depth
            
            # Yükseklik bazlı derinlik
            depth_from_height = (ref_size['height'] * focal_length) / bbox_height if bbox_height > 0 else self.baseline_depth
            
            # Ortalama al
            estimated_depth = (depth_from_width + depth_from_height) / 2
            
            # Sınırlar içinde tut
            estimated_depth = max(self.depth_range[0], min(self.depth_range[1], estimated_depth))
            
            return estimated_depth
            
        except Exception as e:
            self.logger.error(f"Obje derinlik tahmini hatası: {e}")
            return self.baseline_depth
    
    def estimate_person_depth(self, bbox: List[int], face_landmarks: Any = None,
                             additional_cues: Dict[str, Any] = None) -> Tuple[float, float]:
        """İnsan için kafa ve vücut derinliği tahmini"""
        try:
            head_depth = self.baseline_depth
            body_depth = self.baseline_depth
            
            # Yüz landmark'ları varsa daha doğru tahmin
            if face_landmarks is not None:
                try:
                    # Göz arası mesafe kullanarak derinlik tahmini
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[362]
                    
                    # Pixel cinsinden göz arası mesafe
                    eye_distance_px = abs(left_eye.x - right_eye.x) * 1280  # Frame genişliği varsayımı
                    
                    if eye_distance_px > 10:
                        focal_length = 800
                        real_eye_distance = 65  # mm ortalama
                        
                        head_depth = (real_eye_distance * focal_length) / eye_distance_px
                        body_depth = head_depth  # Aynı derinlik varsayımı
                        
                except Exception as e:
                    self.logger.error(f"Face landmark derinlik hatası: {e}")
            
            # Bbox boyutundan da tahmin et
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox
                bbox_height = y2 - y1
                
                if bbox_height > 0:
                    focal_length = 800
                    person_height = 1700  # mm ortalama
                    
                    bbox_depth = (person_height * focal_length) / bbox_height
                    
                    # Landmark tahmini ile ortala
                    head_depth = (head_depth + bbox_depth) / 2
                    body_depth = (body_depth + bbox_depth) / 2
            
            # Sınırlar içinde tut
            head_depth = max(self.depth_range[0], min(self.depth_range[1], head_depth))
            body_depth = max(self.depth_range[0], min(self.depth_range[1], body_depth))
            
            return head_depth, body_depth
            
        except Exception as e:
            self.logger.error(f"İnsan derinlik tahmini hatası: {e}")
            return self.baseline_depth, self.baseline_depth


class SpatialEnvironmentAnalyzer:
    """3D uzaysal çevre analizi ana sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alt modüller
        self.camera_calibration = CameraCalibration(config.get('camera_calibration', {}))
        self.depth_estimator = DepthEstimator3D(config.get('depth_estimation', {}))
        
        # Uzaysal harita
        self.spatial_objects = {}  # object_id -> SpatialObject
        self.spatial_people = {}   # person_id -> SpatialPerson
        self.environment_map = {
            'objects': [],
            'people': [],
            'spatial_relationships': {},
            'environment_bounds': None,
            'last_updated': 0
        }
        
        # Takip geçmişi
        self.max_history_size = config.get('max_history_size', 100)
        self.object_history = deque(maxlen=self.max_history_size)
        self.people_history = deque(maxlen=self.max_history_size)
        
        self.logger.info("SpatialEnvironmentAnalyzer başlatıldı")
    
    def analyze_environment(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                          gaze_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """3D çevre analizi ana fonksiyon"""
        try:
            current_time = time.time()
            
            # Objeleri 3D uzayda konumlandır
            spatial_objects = self._process_objects(detections, frame.shape)
            
            # İnsanları 3D uzayda konumlandır
            spatial_people = self._process_people(detections, gaze_data, frame.shape)
            
            # Uzaysal ilişkileri analiz et
            spatial_relationships = self._analyze_spatial_relationships(spatial_objects, spatial_people)
            
            # Çevre sınırlarını belirle
            environment_bounds = self._calculate_environment_bounds(spatial_objects, spatial_people)
            
            # Çevre haritasını güncelle
            self.environment_map.update({
                'objects': spatial_objects,
                'people': spatial_people,
                'spatial_relationships': spatial_relationships,
                'environment_bounds': environment_bounds,
                'last_updated': current_time
            })
            
            # Geçmişe kaydet
            self.object_history.append({
                'timestamp': current_time,
                'objects': spatial_objects,
                'people': spatial_people
            })
            
            return {
                'spatial_objects': spatial_objects,
                'spatial_people': spatial_people,
                'spatial_relationships': spatial_relationships,
                'environment_bounds': environment_bounds,
                'environment_map': self.environment_map,
                'processing_timestamp': current_time
            }
            
        except Exception as e:
            self.logger.error(f"Çevre analizi hatası: {e}")
            return {
                'spatial_objects': [],
                'spatial_people': [],
                'spatial_relationships': {},
                'environment_bounds': None,
                'environment_map': self.environment_map,
                'processing_timestamp': time.time()
            }
    
    def _process_objects(self, detections: List[Dict[str, Any]], 
                        frame_shape: Tuple[int, int]) -> List[SpatialObject]:
        """Objeleri 3D uzayda konumlandır"""
        spatial_objects = []
        
        for detection in detections:
            try:
                # İnsanları filtrele (ayrı işlenecek)
                if detection.get('label') in ['person', 'face']:
                    continue
                    
                # Obje bilgilerini al
                bbox = detection.get('bbox', [])
                label = detection.get('label', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                if len(bbox) < 4:
                    continue
                
                # Obje merkezini hesapla
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Derinlik tahmini
                depth_mm = self.depth_estimator.estimate_object_depth(bbox, label)
                
                # 3D pozisyon hesapla
                position_3d = self.camera_calibration.pixel_to_3d((center_x, center_y), depth_mm)
                
                # Gerçek dünya koordinatları
                real_world_position = self.camera_calibration.camera_to_world(position_3d)
                
                # 3D boyutları tahmin et
                dimensions_3d = self._estimate_object_dimensions(bbox, label, depth_mm)
                
                # Obje ID oluştur
                object_id = f"{label}_{center_x}_{center_y}_{int(time.time() * 1000) % 10000}"
                
                # SpatialObject oluştur
                spatial_object = SpatialObject(
                    object_id=object_id,
                    label=label,
                    position_3d=position_3d,
                    dimensions_3d=dimensions_3d,
                    confidence=confidence,
                    bbox_2d=bbox,
                    distance_from_camera=depth_mm,
                    real_world_position=real_world_position,
                    timestamp=time.time()
                )
                
                spatial_objects.append(spatial_object)
                
            except Exception as e:
                self.logger.error(f"Obje işleme hatası: {e}")
                continue
        
        return spatial_objects
    
    def _process_people(self, detections: List[Dict[str, Any]], 
                       gaze_data: Dict[str, Any], frame_shape: Tuple[int, int]) -> List[SpatialPerson]:
        """İnsanları 3D uzayda konumlandır"""
        spatial_people = []
        
        for detection in detections:
            try:
                # Sadece insan tespitlerini işle
                if detection.get('label') != 'person':
                    continue
                    
                bbox = detection.get('bbox', [])
                confidence = detection.get('confidence', 0.0)
                
                if len(bbox) < 4:
                    continue
                
                # Kafa ve vücut pozisyonlarını tahmin et
                head_bbox, body_bbox = self._estimate_head_body_regions(bbox)
                
                # Derinlik tahmini
                face_landmarks = gaze_data.get('face_landmarks') if gaze_data else None
                head_depth, body_depth = self.depth_estimator.estimate_person_depth(
                    bbox, face_landmarks
                )
                
                # Kafa merkezi
                head_center = ((head_bbox[0] + head_bbox[2]) // 2, (head_bbox[1] + head_bbox[3]) // 2)
                head_position_3d = self.camera_calibration.pixel_to_3d(head_center, head_depth)
                
                # Vücut merkezi
                body_center = ((body_bbox[0] + body_bbox[2]) // 2, (body_bbox[1] + body_bbox[3]) // 2)
                body_position_3d = self.camera_calibration.pixel_to_3d(body_center, body_depth)
                
                # Kafa pozisyonu ve bakış yönü
                head_pose, gaze_direction = self._estimate_head_pose_gaze(gaze_data, head_depth)
                
                # Person ID oluştur
                person_id = f"person_{head_center[0]}_{head_center[1]}_{int(time.time() * 1000) % 10000}"
                
                # SpatialPerson oluştur
                spatial_person = SpatialPerson(
                    person_id=person_id,
                    head_position_3d=head_position_3d,
                    body_position_3d=body_position_3d,
                    head_pose=head_pose,
                    gaze_direction_3d=gaze_direction,
                    distance_from_camera=head_depth,
                    confidence=confidence,
                    bbox_2d=bbox,
                    face_landmarks=face_landmarks,
                    timestamp=time.time()
                )
                
                spatial_people.append(spatial_person)
                
            except Exception as e:
                self.logger.error(f"İnsan işleme hatası: {e}")
                continue
        
        return spatial_people
    
    def _estimate_head_body_regions(self, person_bbox: List[int]) -> Tuple[List[int], List[int]]:
        """İnsan bbox'ından kafa ve vücut bölgelerini tahmin et"""
        try:
            x1, y1, x2, y2 = person_bbox
            
            # Kafa bölgesi (üst %30)
            head_height = int((y2 - y1) * 0.3)
            head_bbox = [x1, y1, x2, y1 + head_height]
            
            # Vücut bölgesi (alt %70)
            body_bbox = [x1, y1 + head_height, x2, y2]
            
            return head_bbox, body_bbox
            
        except Exception as e:
            self.logger.error(f"Kafa/vücut bölgesi tahmini hatası: {e}")
            return person_bbox, person_bbox
    
    def _estimate_head_pose_gaze(self, gaze_data: Dict[str, Any], 
                               head_depth: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Kafa pozisyonu ve bakış yönünü tahmin et"""
        try:
            # Varsayılan değerler
            head_pose = (0.0, 0.0, 0.0)  # (pitch, yaw, roll)
            gaze_direction = (0.0, 0.0, 1.0)  # İleri doğru bakış
            
            if gaze_data and gaze_data.get('face_landmarks'):
                # Burada daha karmaşık kafa pozisyonu hesaplama yapılabilir
                # Şimdilik basit bir yaklaşım
                
                gaze_vector = gaze_data.get('gaze_vector', (0, 0))
                
                # Gaze vector'den kafa pozisyonu tahmini
                yaw = math.atan2(gaze_vector[0], 1.0)  # Yatay dönüş
                pitch = math.atan2(-gaze_vector[1], 1.0)  # Dikey dönüş
                roll = 0.0  # Basit yaklaşım
                
                head_pose = (pitch, yaw, roll)
                
                # 3D bakış yönü
                gaze_direction = (
                    math.sin(yaw) * math.cos(pitch),
                    -math.sin(pitch),
                    math.cos(yaw) * math.cos(pitch)
                )
            
            return head_pose, gaze_direction
            
        except Exception as e:
            self.logger.error(f"Kafa pozisyonu tahmini hatası: {e}")
            return (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    
    def _estimate_object_dimensions(self, bbox: List[int], label: str, depth_mm: float) -> Tuple[float, float, float]:
        """Obje boyutlarını tahmin et"""
        try:
            # Referans boyutları al
            ref_size = self.depth_estimator.reference_sizes.get(label, 
                         self.depth_estimator.reference_sizes.get('person'))
            
            # Bbox boyutları
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Focal length
            focal_length = 800
            
            # Gerçek boyutları hesapla
            real_width = (bbox_width * depth_mm) / focal_length
            real_height = (bbox_height * depth_mm) / focal_length
            real_depth = ref_size.get('depth', real_width * 0.5)  # Varsayılan derinlik
            
            return (real_width, real_height, real_depth)
            
        except Exception as e:
            self.logger.error(f"Obje boyut tahmini hatası: {e}")
            return (100.0, 100.0, 100.0)
    
    def _analyze_spatial_relationships(self, spatial_objects: List[SpatialObject], 
                                     spatial_people: List[SpatialPerson]) -> Dict[str, Any]:
        """Uzaysal ilişkileri analiz et"""
        try:
            relationships = {
                'object_distances': {},
                'person_distances': {},
                'person_object_distances': {},
                'nearest_objects_to_people': {},
                'spatial_clusters': []
            }
            
            # Obje-obje uzaklıkları
            for i, obj1 in enumerate(spatial_objects):
                for j, obj2 in enumerate(spatial_objects[i+1:], i+1):
                    dist = self._calculate_3d_distance(obj1.position_3d, obj2.position_3d)
                    relationships['object_distances'][f"{obj1.object_id}-{obj2.object_id}"] = dist
            
            # İnsan-insan uzaklıkları
            for i, person1 in enumerate(spatial_people):
                for j, person2 in enumerate(spatial_people[i+1:], i+1):
                    dist = self._calculate_3d_distance(person1.head_position_3d, person2.head_position_3d)
                    relationships['person_distances'][f"{person1.person_id}-{person2.person_id}"] = dist
            
            # İnsan-obje uzaklıkları
            for person in spatial_people:
                nearest_objects = []
                for obj in spatial_objects:
                    dist = self._calculate_3d_distance(person.head_position_3d, obj.position_3d)
                    relationships['person_object_distances'][f"{person.person_id}-{obj.object_id}"] = dist
                    nearest_objects.append((obj.object_id, obj.label, dist))
                
                # En yakın objeleri sırala
                nearest_objects.sort(key=lambda x: x[2])
                relationships['nearest_objects_to_people'][person.person_id] = nearest_objects[:5]
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Uzaysal ilişki analizi hatası: {e}")
            return {}
    
    def _calculate_3d_distance(self, pos1: Tuple[float, float, float], 
                              pos2: Tuple[float, float, float]) -> float:
        """3D uzaklık hesapla"""
        try:
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
        except:
            return float('inf')
    
    def _calculate_environment_bounds(self, spatial_objects: List[SpatialObject], 
                                    spatial_people: List[SpatialPerson]) -> Dict[str, Any]:
        """Çevre sınırlarını hesapla"""
        try:
            all_positions = []
            
            # Obje pozisyonları
            for obj in spatial_objects:
                all_positions.append(obj.position_3d)
            
            # İnsan pozisyonları
            for person in spatial_people:
                all_positions.append(person.head_position_3d)
                all_positions.append(person.body_position_3d)
            
            if not all_positions:
                return None
            
            # Min/max koordinatları
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            z_coords = [pos[2] for pos in all_positions]
            
            return {
                'min_x': min(x_coords),
                'max_x': max(x_coords),
                'min_y': min(y_coords),
                'max_y': max(y_coords),
                'min_z': min(z_coords),
                'max_z': max(z_coords),
                'center': (
                    (min(x_coords) + max(x_coords)) / 2,
                    (min(y_coords) + max(y_coords)) / 2,
                    (min(z_coords) + max(z_coords)) / 2
                ),
                'dimensions': (
                    max(x_coords) - min(x_coords),
                    max(y_coords) - min(y_coords),
                    max(z_coords) - min(z_coords)
                )
            }
            
        except Exception as e:
            self.logger.error(f"Çevre sınırları hesaplama hatası: {e}")
            return None
    
    def get_spatial_context(self, query_position: Tuple[float, float, float], 
                           radius: float = 1000.0) -> Dict[str, Any]:
        """Belirli bir pozisyon etrafındaki uzaysal bağlamı al"""
        try:
            nearby_objects = []
            nearby_people = []
            
            for obj in self.environment_map.get('objects', []):
                dist = self._calculate_3d_distance(query_position, obj.position_3d)
                if dist <= radius:
                    nearby_objects.append({
                        'object': obj,
                        'distance': dist
                    })
            
            for person in self.environment_map.get('people', []):
                dist = self._calculate_3d_distance(query_position, person.head_position_3d)
                if dist <= radius:
                    nearby_people.append({
                        'person': person,
                        'distance': dist
                    })
            
            # Uzaklığa göre sırala
            nearby_objects.sort(key=lambda x: x['distance'])
            nearby_people.sort(key=lambda x: x['distance'])
            
            return {
                'nearby_objects': nearby_objects,
                'nearby_people': nearby_people,
                'query_position': query_position,
                'search_radius': radius
            }
            
        except Exception as e:
            self.logger.error(f"Uzaysal bağlam alma hatası: {e}")
            return {
                'nearby_objects': [],
                'nearby_people': [],
                'query_position': query_position,
                'search_radius': radius
            } 