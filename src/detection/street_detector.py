"""
Sokak öğeleri tespit sınıfı
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any


class StreetDetector:
    """Sokak öğelerini tespit eden sınıf"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Sokak tespit konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sokak öğeleri için özelleşmiş parametreler
        self.sign_cascade = cv2.CascadeClassifier()
        
        # Text detector'ı güvenli yükle
        self.text_detector = None
        text_model_path = config.get('text_model_path')
        if text_model_path:
            try:
                import os
                if os.path.exists(text_model_path):
                    self.text_detector = cv2.dnn.readNet(text_model_path)
                    self.logger.info(f"EAST text detector yüklendi: {text_model_path}")
                else:
                    self.logger.warning(f"Text detection model bulunamadı: {text_model_path} - Text detection devre dışı")
            except Exception as e:
                self.logger.warning(f"Text detector yüklenemedi: {e} - Text detection devre dışı")
        
        # Renk filtreleri (tabelalar için)
        self.sign_colors = {
            'red': ([0, 50, 50], [10, 255, 255]),     # Kırmızı tabelalar
            'blue': ([100, 50, 50], [130, 255, 255]), # Mavi tabelalar
            'green': ([40, 50, 50], [80, 255, 255]),  # Yeşil tabelalar
            'yellow': ([20, 50, 50], [30, 255, 255])  # Sarı tabelalar
        }
        
        self.logger.info("StreetDetector başlatıldı")
    
    def detect_street_elements(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Sokak öğelerini tespit et
        
        Args:
            frame: Giriş görüntüsü
            
        Returns:
            Tespit edilen sokak öğelerinin listesi
        """
        detections = []
        
        try:
            # Trafik tabelalarını tespit et
            signs = self._detect_traffic_signs(frame)
            detections.extend(signs)
            
            # Reklam panolarını tespit et
            billboards = self._detect_billboards(frame)
            detections.extend(billboards)
            
            # Sokak tabelalarını tespit et
            street_signs = self._detect_street_signs(frame)
            detections.extend(street_signs)
            
            # Yön işaretlerini tespit et
            direction_signs = self._detect_direction_signs(frame)
            detections.extend(direction_signs)
            
        except Exception as e:
            self.logger.error(f"Sokak öğeleri tespit hatası: {e}")
        
        return detections
    
    def _detect_traffic_signs(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Trafik tabelalarını tespit et"""
        detections = []
        
        # HSV renk uzayına çevir
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Her renk için tespit
        for color_name, (lower, upper) in self.sign_colors.items():
            # Renk maskesi
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Morfolojik işlemler
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Konturları bul
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum alan
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Şekil analizi (dairesel veya kare tabelalar)
                    aspect_ratio = w / h
                    if 0.7 <= aspect_ratio <= 1.3:  # Yaklaşık kare/daire
                        detections.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.8,  # Sabit güven skoru
                            'label': f'trafik_tabelası_{color_name}',
                            'source': 'street_detector',
                            'type': 'traffic_sign'
                        })
        
        return detections
    
    def _detect_billboards(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Reklam panolarını tespit et"""
        detections = []
        
        # Kenar tespiti
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Çizgi tespiti (HoughLines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Dikdörtgen şekiller ara
            rectangles = self._find_rectangles_from_lines(lines)
            
            for rect in rectangles:
                x, y, w, h = rect
                
                # Reklam panosu olabilecek boyutları filtrele
                if w > 200 and h > 100 and w/h > 1.5:  # Geniş dikdörtgen
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.7,
                        'label': 'reklam_panosu',
                        'source': 'street_detector',
                        'type': 'billboard'
                    })
        
        return detections
    
    def _detect_street_signs(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Sokak tabelalarını tespit et"""
        detections = []
        
        if self.text_detector is None:
            return detections
        
        try:
            # EAST text detection kullan
            blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), 
                                       (123.68, 116.78, 103.94), swapRB=True, crop=False)
            self.text_detector.setInput(blob)
            scores, geometry = self.text_detector.forward(['feature_fusion/Conv_7/Sigmoid', 
                                                          'feature_fusion/concat_3'])
            
            # Text bölgelerini çıkar
            rectangles, confidences = self._decode_predictions(scores, geometry)
            
            # NMS uygula
            indices = cv2.dnn.NMSBoxes(rectangles, confidences, 0.5, 0.4)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = rectangles[i]
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': confidences[i],
                        'label': 'sokak_tabelası',
                        'source': 'street_detector',
                        'type': 'street_sign'
                    })
                    
        except Exception as e:
            self.logger.error(f"Sokak tabelası tespit hatası: {e}")
        
        return detections
    
    def _detect_direction_signs(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Yön işaretlerini tespit et"""
        detections = []
        
        # Mavi ve yeşil yön tabelaları için özel algoritma
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Mavi yön tabelaları
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum alan
                x, y, w, h = cv2.boundingRect(contour)
                
                # Yön tabelası şekli kontrolü (dikdörtgen)
                if w/h > 1.5 and w > 150:
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.75,
                        'label': 'yön_tabelası',
                        'source': 'street_detector',
                        'type': 'direction_sign'
                    })
        
        return detections
    
    def _find_rectangles_from_lines(self, lines: np.ndarray) -> List[tuple]:
        """Çizgilerden dikdörtgenleri bul"""
        rectangles = []
        
        # Basit dikdörtgen bulma algoritması
        # Bu kısım daha gelişmiş hale getirilebilir
        
        return rectangles
    
    def _decode_predictions(self, scores: np.ndarray, geometry: np.ndarray, 
                          scoreThresh: float = 0.5):
        """EAST model çıktılarını çöz"""
        rectangles = []
        confidences = []
        
        # EAST model çıktı formatına göre işle
        # Bu kısım EAST model implementasyonuna bağlı
        
        return rectangles, confidences 