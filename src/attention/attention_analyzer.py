"""
Dikkat analizi sınıfı
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple


class AttentionAnalyzer:
    """Dikkat analizi ve bakış-nesne ilişkisi sınıfı"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Dikkat analizi konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analiz parametreleri
        self.attention_radius = config.get('attention_radius', 50)
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.temporal_smoothing = config.get('temporal_smoothing', 0.7)
        
        # Geçmiş veriler
        self.previous_attention_map = None
        self.attention_history = []
        
        self.logger.info("AttentionAnalyzer başlatıldı")
    
    def analyze_attention_objects(self, gaze_data: Dict[str, Any], 
                                 detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bakış ve nesne tespitleri arasındaki ilişkiyi analiz et
        
        Args:
            gaze_data: Bakış verileri
            detections: Tespit edilen nesneler
            
        Returns:
            Dikkat analizi sonuçları
        """
        if not gaze_data or not gaze_data.get('gaze_point'):
            return {
                'focused_objects': [],
                'attention_distribution': {},
                'interest_level': 0.0
            }
        
        gaze_point = gaze_data['gaze_point']
        
        # Bakış noktasına yakın nesneleri bul
        focused_objects = self._find_objects_in_gaze(gaze_point, detections)
        
        # Dikkat dağılımını hesapla
        attention_distribution = self._calculate_attention_distribution(
            gaze_point, detections
        )
        
        # İlgi seviyesini hesapla
        interest_level = self._calculate_interest_level(focused_objects)
        
        # Sonuçları derle
        result = {
            'gaze_point': gaze_point,
            'focused_objects': focused_objects,
            'attention_distribution': attention_distribution,
            'interest_level': interest_level,
            'gaze_stability': self._calculate_gaze_stability(gaze_point)
        }
        
        # Geçmişe ekle
        self._update_attention_history(result)
        
        return result
    
    def _find_objects_in_gaze(self, gaze_point: Tuple[int, int], 
                             detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Bakış noktasındaki nesneleri bul"""
        focused_objects = []
        gx, gy = gaze_point
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Nesne merkezi
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Bakış noktasından uzaklık
            distance = np.sqrt((gx - center_x)**2 + (gy - center_y)**2)
            
            # Nesne içinde mi kontrolü
            if x1 <= gx <= x2 and y1 <= gy <= y2:
                attention_score = 1.0
            elif distance <= self.attention_radius:
                # Yakınlık bazlı skor
                attention_score = 1.0 - (distance / self.attention_radius)
            else:
                continue
            
            focused_obj = detection.copy()
            focused_obj['attention_score'] = attention_score
            focused_obj['distance_to_gaze'] = distance
            focused_objects.append(focused_obj)
        
        # Dikkat skoruna göre sırala
        focused_objects.sort(key=lambda x: x['attention_score'], reverse=True)
        
        return focused_objects
    
    def _calculate_attention_distribution(self, gaze_point: Tuple[int, int], 
                                        detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """Nesne kategorilerine göre dikkat dağılımı"""
        distribution = {}
        total_attention = 0.0
        
        gx, gy = gaze_point
        
        for detection in detections:
            category = detection.get('type', 'unknown')
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Nesne merkezinden uzaklık
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            distance = np.sqrt((gx - center_x)**2 + (gy - center_y)**2)
            
            # Dikkat ağırlığı
            if distance <= self.attention_radius:
                weight = 1.0 - (distance / self.attention_radius)
                if category not in distribution:
                    distribution[category] = 0.0
                distribution[category] += weight
                total_attention += weight
        
        # Normalize et
        if total_attention > 0:
            for category in distribution:
                distribution[category] /= total_attention
        
        return distribution
    
    def _calculate_interest_level(self, focused_objects: List[Dict[str, Any]]) -> float:
        """İlgi seviyesini hesapla"""
        if not focused_objects:
            return 0.0
        
        # En yüksek dikkat skoru
        max_attention = focused_objects[0]['attention_score']
        
        # Odaklanılan nesne sayısı
        object_count = len(focused_objects)
        
        # Nesne türü çeşitliliği
        unique_types = len(set(obj.get('type', 'unknown') for obj in focused_objects))
        
        # Kombine ilgi seviyesi
        interest_level = (max_attention * 0.5 + 
                         min(object_count / 5.0, 1.0) * 0.3 + 
                         min(unique_types / 3.0, 1.0) * 0.2)
        
        return min(interest_level, 1.0)
    
    def _calculate_gaze_stability(self, current_gaze: Tuple[int, int]) -> float:
        """Bakış kararlılığını hesapla"""
        if len(self.attention_history) < 2:
            return 0.5  # Varsayılan
        
        # Son birkaç bakış noktasını al
        recent_gazes = [h.get('gaze_point') for h in self.attention_history[-5:] 
                       if h.get('gaze_point')]
        
        if len(recent_gazes) < 2:
            return 0.5
        
        # Bakış noktaları arasındaki ortalama mesafe
        distances = []
        for i in range(1, len(recent_gazes)):
            prev_gaze = recent_gazes[i-1]
            curr_gaze = recent_gazes[i]
            distance = np.sqrt((curr_gaze[0] - prev_gaze[0])**2 + 
                             (curr_gaze[1] - prev_gaze[1])**2)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # Kararlılık skoru (düşük mesafe = yüksek kararlılık)
        stability = max(0.0, 1.0 - avg_distance / 100.0)
        
        return stability
    
    def _update_attention_history(self, attention_result: Dict[str, Any]) -> None:
        """Dikkat geçmişini güncelle"""
        self.attention_history.append(attention_result)
        
        # Geçmiş boyutunu sınırla
        max_history = self.config.get('max_history_length', 30)
        if len(self.attention_history) > max_history:
            self.attention_history = self.attention_history[-max_history:]
    
    def get_attention_summary(self, time_window: int = 10) -> Dict[str, Any]:
        """Belirli zaman penceresi için dikkat özeti"""
        if len(self.attention_history) < time_window:
            recent_history = self.attention_history
        else:
            recent_history = self.attention_history[-time_window:]
        
        if not recent_history:
            return {}
        
        # Toplam dikkat dağılımı
        total_distribution = {}
        total_interest = 0.0
        
        for entry in recent_history:
            # Dikkat dağılımını topla
            for category, value in entry.get('attention_distribution', {}).items():
                if category not in total_distribution:
                    total_distribution[category] = 0.0
                total_distribution[category] += value
            
            # İlgi seviyesini topla
            total_interest += entry.get('interest_level', 0.0)
        
        # Ortalama hesapla
        avg_distribution = {k: v / len(recent_history) 
                          for k, v in total_distribution.items()}
        avg_interest = total_interest / len(recent_history)
        
        return {
            'average_attention_distribution': avg_distribution,
            'average_interest_level': avg_interest,
            'time_window': time_window,
            'sample_count': len(recent_history)
        } 