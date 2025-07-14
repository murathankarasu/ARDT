"""
Ana nesne tespit sınıfı
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any
from ultralytics import YOLO

from .street_detector import StreetDetector
from .store_detector import StoreDetector


class ObjectDetector:
    """Ana nesne tespit sınıfı"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Tespit konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # YOLO modelini yükle - Apple M2 optimize
        self.model = YOLO(config.get('model_path', 'yolov8n.pt'))
        
        # Device seçimi - M2 için MPS
        import torch
        if torch.backends.mps.is_available():
            self.device = 'mps'
            self.logger.info("YOLO Apple M2 GPU (MPS) kullanıyor")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            self.logger.info("YOLO CUDA GPU kullanıyor")
        else:
            self.device = 'cpu'
            self.logger.info("YOLO CPU kullanıyor")
        
        # Özelleşmiş tespit edici sınıfları
        self.street_detector = StreetDetector(config.get('street_config', {}))
        self.store_detector = StoreDetector(config.get('store_config', {}))
        
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
        # Genel nesne kategorileri (ev/ofis/günlük nesneler için)
        self.general_categories = {
            'furniture': ['chair', 'couch', 'bed', 'dining table', 'desk'],
            'plants': ['potted plant'],
            'electronics': ['tv', 'laptop', 'computer', 'cell phone', 'monitor'],
            'kitchen': ['refrigerator', 'microwave', 'oven', 'sink', 'cup', 'bowl', 'bottle'],
            'living': ['book', 'clock', 'vase', 'pillow', 'blanket', 'lamp'],
            'outdoor': ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'traffic light', 'bench'],
            'animals': ['person', 'cat', 'dog', 'bird', 'horse'],
            'accessories': ['handbag', 'tie', 'suitcase', 'umbrella', 'hat']
        }
        
        self.logger.info("ObjectDetector başlatıldı")
    
    def detect(self, frame: np.ndarray, mode: str = 'mixed', gaze_available: bool = True) -> List[Dict[str, Any]]:
        """
        Görüntüde nesne tespiti yap
        
        Args:
            frame: Giriş görüntüsü
            mode: Tespit modu ('street', 'store', 'mixed', 'general')
            gaze_available: Gaze tracking mevcut mu?
            
        Returns:
            Tespit edilen nesnelerin listesi
        """
        detections = []
        
        try:
            # Durum bazlı mod seçimi
            effective_mode = self._determine_effective_mode(mode, gaze_available)
            
            # Genel YOLO tespiti - M2 GPU ile
            results = self.model(frame, conf=self.confidence_threshold, device=self.device)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Bounding box koordinatları
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        label = self.model.names[class_id]
                        
                        # Mod bazlı filtreleme
                        if self._should_include_detection(label, effective_mode):
                            detection = {
                                'bbox': xyxy,
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'label': label,
                                'source': 'yolo',
                                'category': self._get_object_category(label),
                                'mode': effective_mode
                            }
                            detections.append(detection)
            
            # Mod bazlı özelleşmiş tespitler (sadece gaze mevcut değilse)
            if effective_mode in ['street', 'mixed'] and gaze_available:
                street_detections = self.street_detector.detect_street_elements(frame)
                detections.extend(street_detections)
            
            if effective_mode in ['store', 'mixed'] and gaze_available:
                store_detections = self.store_detector.detect_store_elements(frame)
                detections.extend(store_detections)
            
            # Genel mod için ek bilgi
            if effective_mode == 'general':
                detections = self._enhance_general_detections(detections, frame)
            
            # Çakışan tespitleri filtrele
            detections = self._filter_overlapping_detections(detections)
            
        except Exception as e:
            self.logger.error(f"Tespit hatası: {e}")
        
        return detections
    
    def _determine_effective_mode(self, requested_mode: str, gaze_available: bool) -> str:
        """Duruma göre etkili modu belirle"""
        # Gaze tracking yoksa genel moda geç
        if not gaze_available:
            self.logger.info("Gaze tracking mevcut değil - Genel nesne tespit moduna geçiliyor")
            return 'general'
        
        # Sistem yoğunluğuna göre karar ver
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Yüksek sistem yükü varsa basit moda geç
            if memory_percent > 80 or cpu_percent > 85:
                self.logger.info(f"Yüksek sistem yükü (RAM: {memory_percent}%, CPU: {cpu_percent}%) - Genel moda geçiliyor")
                return 'general'
        except:
            pass
        
        return requested_mode
    
    def _should_include_detection(self, label: str, mode: str) -> bool:
        """Tespitin dahil edilip edilmeyeceğini belirle"""
        if mode == 'general':
            # Genel modda hemen hemen her şeyi dahil et
            return True
        elif mode == 'street':
            # Sokak modunda dış mekan nesneleri
            outdoor_objects = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'traffic light', 
                             'stop sign', 'person', 'bench', 'fire hydrant']
            return label in outdoor_objects
        elif mode == 'store':
            # Mağaza modunda iç mekan/alışveriş nesneleri
            store_objects = ['person', 'bottle', 'cup', 'bowl', 'banana', 'apple', 'orange',
                           'broccoli', 'carrot', 'pizza', 'donut', 'cake', 'chair', 'refrigerator']
            return label in store_objects
        else:  # mixed
            return True
    
    def _get_object_category(self, label: str) -> str:
        """Nesne kategorisini belirle"""
        for category, objects in self.general_categories.items():
            if label in objects:
                return category
        return 'other'
    
    def _enhance_general_detections(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """Genel tespit modunda ek bilgiler ekle"""
        enhanced = []
        
        for detection in detections:
            # Nesne boyutu bilgisi
            bbox = detection['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            # Ekran üzerindeki oransal boyut
            frame_area = frame.shape[0] * frame.shape[1]
            size_ratio = area / frame_area
            
            # Boyut kategorisi
            if size_ratio > 0.3:
                size_category = 'large'
            elif size_ratio > 0.1:
                size_category = 'medium'
            else:
                size_category = 'small'
            
            # Pozisyon bilgisi
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Ekranın hangi bölgesinde
            screen_region = self._get_screen_region(center_x, center_y, frame.shape[1], frame.shape[0])
            
            # Ek bilgileri ekle
            enhanced_detection = detection.copy()
            enhanced_detection.update({
                'size_category': size_category,
                'size_ratio': size_ratio,
                'screen_region': screen_region,
                'center_point': (center_x, center_y),
                'area': area
            })
            
            enhanced.append(enhanced_detection)
        
        return enhanced
    
    def _get_screen_region(self, x: int, y: int, width: int, height: int) -> str:
        """Ekran bölgesini belirle"""
        # Ekranı 3x3 grid'e böl
        col = 0 if x < width // 3 else (1 if x < 2 * width // 3 else 2)
        row = 0 if y < height // 3 else (1 if y < 2 * height // 3 else 2)
        
        regions = [
            ['top-left', 'top-center', 'top-right'],
            ['center-left', 'center', 'center-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]
        
        return regions[row][col]
    
    def _filter_overlapping_detections(self, detections: List[Dict[str, Any]], 
                                      iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Çakışan tespitleri filtrele"""
        if len(detections) <= 1:
            return detections
        
        # Güven skoruna göre sırala
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for i, det1 in enumerate(detections):
            keep = True
            for j, det2 in enumerate(filtered):
                iou = self._calculate_iou(det1['bbox'], det2['bbox'])
                if iou > iou_threshold:
                    keep = False
                    break
            
            if keep:
                filtered.append(det1)
        
        return filtered
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """IoU (Intersection over Union) hesapla"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Kesişim alanı
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Birleşim alanı
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0 