"""
Görüntü işleme sınıfı
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.image_utils import ImageUtils


class ImageProcessor:
    """Görüntü işleme ve ön hazırlık sınıfı"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: İşleme konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # İşleme parametreleri
        self.blur_kernel = config.get('blur_kernel_size', 3)
        self.noise_reduction = config.get('noise_reduction', True)
        self.auto_enhance = config.get('auto_enhance', True)
        
        self.logger.info("ImageProcessor başlatıldı")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame'i işle
        
        Args:
            frame: Ham görüntü
            
        Returns:
            İşlenmiş görüntü
        """
        try:
            processed = frame.copy()
            
            # Ön işleme
            if self.noise_reduction:
                processed = ImageUtils.preprocess_image(
                    processed, 
                    blur_kernel=self.blur_kernel,
                    noise_reduction=True
                )
            
            # Otomatik iyileştirme
            if self.auto_enhance:
                processed = self._auto_enhance(processed)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Frame işleme hatası: {e}")
            return frame
    
    def _auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """Otomatik görüntü iyileştirme"""
        # Histogram analizi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Ortalama parlaklığa göre ayarlama
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 100:
            # Karanlık görüntü - parlaklığı artır
            enhanced = ImageUtils.enhance_image(
                image, brightness=1.2, contrast=1.1
            )
        elif mean_brightness > 180:
            # Aydınlık görüntü - kontrastı artır
            enhanced = ImageUtils.enhance_image(
                image, brightness=0.95, contrast=1.15
            )
        else:
            # Normal görüntü - hafif iyileştirme
            enhanced = ImageUtils.enhance_image(
                image, brightness=1.05, contrast=1.05
            )
        
        return enhanced
    
    def prepare_for_detection(self, image: np.ndarray, 
                            target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Tespit için görüntüyü hazırla
        
        Args:
            image: Giriş görüntüsü
            target_size: Hedef boyut (opsiyonel)
            
        Returns:
            Hazırlanmış görüntü
        """
        prepared = image.copy()
        
        # Boyut ayarlama
        if target_size:
            prepared = ImageUtils.resize_with_aspect_ratio(prepared, target_size)
        
        # Normalizasyon (isteğe bağlı)
        prepared = prepared.astype(np.float32) / 255.0
        
        return prepared
    
    def create_detection_overlay(self, image: np.ndarray, 
                               detections: list) -> np.ndarray:
        """
        Tespit sonuçlarını görüntüye çiz
        
        Args:
            image: Ana görüntü
            detections: Tespit listesi
            
        Returns:
            Çizilmiş görüntü
        """
        overlay = image.copy()
        
        # Tespit türüne göre renk paleti
        color_map = {
            'traffic_sign': (0, 255, 0),      # Yeşil
            'billboard': (255, 0, 0),         # Mavi
            'street_sign': (0, 255, 255),     # Sarı
            'direction_sign': (255, 255, 0),  # Cyan
            'shelf': (128, 0, 128),           # Mor
            'product_category': (255, 128, 0), # Turuncu
            'price_tag': (0, 128, 255),       # Açık mavi
            'shopping_cart': (128, 255, 0)    # Açık yeşil
        }
        
        for detection in detections:
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            detection_type = detection.get('type', 'unknown')
            
            color = color_map.get(detection_type, (255, 255, 255))
            
            overlay = ImageUtils.draw_detection_box(
                overlay, bbox, label, confidence, color
            )
        
        return overlay 