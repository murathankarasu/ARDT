"""
Mağaza öğeleri tespit sınıfı
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any


class StoreDetector:
    """Mağaza öğelerini tespit eden sınıf"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Mağaza tespit konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ürün kategorileri ve renk aralıkları
        self.product_categories = {
            'et_urunleri': ([0, 0, 100], [30, 100, 255]),      # Kırmızı tonları
            'sut_urunleri': ([0, 0, 200], [180, 30, 255]),     # Beyaz tonları
            'meyve_sebze': ([35, 50, 50], [85, 255, 255]),     # Yeşil tonları
            'ekmek_unlu': ([10, 50, 100], [25, 200, 255]),     # Kahverengi tonları
            'icecek': ([90, 50, 50], [120, 255, 255])          # Mavi tonları
        }
        
        # Raf tespiti için parametreler
        self.shelf_min_width = 200
        self.shelf_min_height = 50
        self.shelf_aspect_ratio_range = (2.0, 8.0)
        
        self.logger.info("StoreDetector başlatıldı")
    
    def detect_store_elements(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Mağaza öğelerini tespit et
        
        Args:
            frame: Giriş görüntüsü
            
        Returns:
            Tespit edilen mağaza öğelerinin listesi
        """
        detections = []
        
        try:
            # Rafları tespit et
            shelves = self._detect_shelves(frame)
            detections.extend(shelves)
            
            # Ürün kategorilerini tespit et
            products = self._detect_product_categories(frame)
            detections.extend(products)
            
            # Fiyat etiketlerini tespit et
            price_tags = self._detect_price_tags(frame)
            detections.extend(price_tags)
            
            # Alışveriş araçlarını tespit et
            shopping_equipment = self._detect_shopping_equipment(frame)
            detections.extend(shopping_equipment)
            
        except Exception as e:
            self.logger.error(f"Mağaza öğeleri tespit hatası: {e}")
        
        return detections
    
    def _detect_shelves(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Rafları tespit et"""
        detections = []
        
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Kenar tespiti
        edges = cv2.Canny(gray, 50, 150)
        
        # Yatay çizgileri tespit et (raflar için)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Dikey çizgileri tespit et (raf ayakları için)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Çizgileri birleştir
        shelf_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Konturları bul
        contours, _ = cv2.findContours(shelf_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum alan
                x, y, w, h = cv2.boundingRect(contour)
                
                # Raf özellikleri kontrolü
                if (w >= self.shelf_min_width and h >= self.shelf_min_height and
                    self.shelf_aspect_ratio_range[0] <= w/h <= self.shelf_aspect_ratio_range[1]):
                    
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.8,
                        'label': 'raf',
                        'source': 'store_detector',
                        'type': 'shelf'
                    })
        
        return detections
    
    def _detect_product_categories(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Ürün kategorilerini tespit et"""
        detections = []
        
        # HSV renk uzayına çevir
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for category, (lower, upper) in self.product_categories.items():
            # Renk maskesi
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Gürültüyü azalt
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Konturları bul
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum alan
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Ürün grubu boyut kontrolü
                    if w > 50 and h > 50:
                        detections.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.6,
                            'label': category,
                            'source': 'store_detector',
                            'type': 'product_category'
                        })
        
        return detections
    
    def _detect_price_tags(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Fiyat etiketlerini tespit et"""
        detections = []
        
        # Beyaz/sarı etiketleri ara
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Beyaz etiketler
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        
        # Sarı etiketler
        yellow_mask = cv2.inRange(hsv, np.array([20, 50, 200]), np.array([30, 255, 255]))
        
        # Maskeleri birleştir
        price_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Konturları bul
        contours, _ = cv2.findContours(price_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Fiyat etiketi boyutu
                x, y, w, h = cv2.boundingRect(contour)
                
                # Etiket şekli kontrolü (küçük dikdörtgen)
                aspect_ratio = w / h
                if 1.5 <= aspect_ratio <= 4.0:
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.7,
                        'label': 'fiyat_etiketi',
                        'source': 'store_detector',
                        'type': 'price_tag'
                    })
        
        return detections
    
    def _detect_shopping_equipment(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Alışveriş araçlarını tespit et"""
        detections = []
        
        # Sepet ve arabalar için basit şekil tespiti
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Daire tespiti (sepet tekerlekleri)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=50)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Tekerlekleri grupla (alışveriş arabası)
            for (x, y, r) in circles:
                # Basit gruplandırma mantığı
                detections.append({
                    'bbox': [x-r*2, y-r*2, x+r*2, y+r*2],
                    'confidence': 0.5,
                    'label': 'alışveriş_arabası',
                    'source': 'store_detector',
                    'type': 'shopping_cart'
                })
        
        return detections 