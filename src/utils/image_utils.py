"""
Görüntü işleme yardımcı fonksiyonları
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class ImageUtils:
    """Görüntü işleme yardımcı sınıfı"""
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Aspect ratio koruyarak yeniden boyutlandır
        
        Args:
            image: Giriş görüntüsü
            target_size: Hedef boyut (width, height)
            
        Returns:
            Yeniden boyutlandırılmış görüntü
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Aspect ratio hesapla
        aspect_ratio = w / h
        
        if aspect_ratio > target_w / target_h:
            # Genişlik sınırlayıcı
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            # Yükseklik sınırlayıcı
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        return cv2.resize(image, (new_w, new_h))
    
    @staticmethod
    def preprocess_image(image: np.ndarray, blur_kernel: int = 3, 
                        noise_reduction: bool = True) -> np.ndarray:
        """
        Görüntü ön işleme
        
        Args:
            image: Giriş görüntüsü
            blur_kernel: Blur kernel boyutu
            noise_reduction: Gürültü azaltma uygula
            
        Returns:
            İşlenmiş görüntü
        """
        processed = image.copy()
        
        if noise_reduction:
            # Gaussian blur ile gürültü azaltma
            processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
        
        return processed
    
    @staticmethod
    def enhance_image(image: np.ndarray, brightness: float = 1.0, 
                     contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
        """
        Görüntü iyileştirme
        
        Args:
            image: Giriş görüntüsü
            brightness: Parlaklık çarpanı
            contrast: Kontrast çarpanı
            saturation: Doygunluk çarpanı
            
        Returns:
            İyileştirilmiş görüntü
        """
        # Parlaklık ve kontrast ayarı
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness - 1.0)
        
        # Doygunluk ayarı
        if saturation != 1.0:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced
    
    @staticmethod
    def draw_detection_box(image: np.ndarray, bbox: List[int], label: str, 
                          confidence: float, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Tespit kutusunu çiz
        
        Args:
            image: Görüntü
            bbox: Bounding box [x1, y1, x2, y2]
            label: Etiket
            confidence: Güven skoru
            color: Renk (BGR)
            
        Returns:
            Çizilmiş görüntü
        """
        result = image.copy()
        x1, y1, x2, y2 = bbox
        
        # Kutu çiz
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Etiket metni
        label_text = f"{label}: {confidence:.2f}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Etiket arka planı
        cv2.rectangle(result, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Etiket metni
        cv2.putText(result, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    @staticmethod
    def crop_region(image: np.ndarray, bbox: List[int], 
                   padding: int = 10) -> Optional[np.ndarray]:
        """
        Belirtilen bölgeyi kırp
        
        Args:
            image: Görüntü
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Kenar boşluğu
            
        Returns:
            Kırpılmış görüntü veya None
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Padding ile genişlet
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Geçerli boyutları kontrol et
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def create_mosaic(images: List[np.ndarray], grid_size: Tuple[int, int]) -> np.ndarray:
        """
        Görüntülerden mozaik oluştur
        
        Args:
            images: Görüntü listesi
            grid_size: Grid boyutu (rows, cols)
            
        Returns:
            Mozaik görüntü
        """
        rows, cols = grid_size
        
        if len(images) > rows * cols:
            images = images[:rows * cols]
        
        # İlk görüntünün boyutunu al
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        h, w = images[0].shape[:2]
        
        # Eksik görüntüleri siyah ile doldur
        while len(images) < rows * cols:
            images.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        # Grid oluştur
        mosaic_rows = []
        for r in range(rows):
            row_images = images[r * cols:(r + 1) * cols]
            mosaic_row = np.hstack(row_images)
            mosaic_rows.append(mosaic_row)
        
        return np.vstack(mosaic_rows) 