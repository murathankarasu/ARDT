"""
Dikkat analizi sınıfı - Gelişmiş tracking ile
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque


class ObjectAttentionTracker:
    """Objelere dikkat süresi takip eden sınıf"""
    
    def __init__(self, max_history_seconds: int = 60):
        self.max_history_seconds = max_history_seconds
        self.object_attention_history = defaultdict(list)  # object_id -> [(timestamp, attention_score), ...]
        self.active_objects = {}  # object_id -> {'first_seen': timestamp, 'total_attention': float, 'last_seen': timestamp}
        self.logger = logging.getLogger(__name__)
    
    def update_attention(self, focused_objects: List[Dict[str, Any]]):
        """Dikkat edilen objeleri güncelle"""
        current_time = time.time()
        
        for obj in focused_objects:
            object_id = self._generate_object_id(obj)
            attention_score = obj.get('attention_score', 0.0)
            
            # Obje historysini güncelle
            self.object_attention_history[object_id].append((current_time, attention_score))
            
            # Aktif objeleri güncelle
            if object_id not in self.active_objects:
                self.active_objects[object_id] = {
                    'first_seen': current_time,
                    'total_attention': 0.0,
                    'last_seen': current_time,
                    'object_data': obj
                }
            
            self.active_objects[object_id]['last_seen'] = current_time
            self.active_objects[object_id]['total_attention'] += attention_score
        
        # Eski kayıtları temizle
        self._cleanup_old_records(current_time)
    
    def get_object_attention_duration(self, object_id: str) -> float:
        """Objeye bakış süresi (saniye)"""
        if object_id not in self.active_objects:
            return 0.0
        
        obj_data = self.active_objects[object_id]
        return obj_data['last_seen'] - obj_data['first_seen']
    
    def get_object_total_attention_score(self, object_id: str) -> float:
        """Objenin toplam dikkat skoru"""
        if object_id not in self.active_objects:
            return 0.0
        return self.active_objects[object_id]['total_attention']
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """Dikkat özeti"""
        current_time = time.time()
        summary = {
            'total_active_objects': len(self.active_objects),
            'objects': []
        }
        
        for object_id, obj_data in self.active_objects.items():
            duration = current_time - obj_data['first_seen']
            summary['objects'].append({
                'object_id': object_id,
                'object_class': obj_data['object_data'].get('label', 'unknown'),
                'duration_seconds': duration,
                'total_attention_score': obj_data['total_attention'],
                'last_seen_seconds_ago': current_time - obj_data['last_seen']
            })
        
        # En çok ilgi çeken objeye göre sırala
        summary['objects'].sort(key=lambda x: x['total_attention_score'], reverse=True)
        return summary
    
    def _generate_object_id(self, obj: Dict[str, Any]) -> str:
        """Obje için unique ID oluştur"""
        bbox = obj.get('bbox', [0, 0, 0, 0])
        label = obj.get('label', 'unknown')
        # Bounding box merkezi ve sınıf ismi ile ID oluştur
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        return f"{label}_{center_x}_{center_y}"
    
    def _cleanup_old_records(self, current_time: float):
        """Eski kayıtları temizle"""
        cutoff_time = current_time - self.max_history_seconds
        
        # History temizleme
        for object_id in list(self.object_attention_history.keys()):
            history = self.object_attention_history[object_id]
            # Eski kayıtları filtrele
            filtered_history = [(t, score) for t, score in history if t > cutoff_time]
            if filtered_history:
                self.object_attention_history[object_id] = filtered_history
            else:
                del self.object_attention_history[object_id]
        
        # Aktif objeler temizleme (5 saniye görmediğimiz objeleri kaldır)
        inactive_cutoff = current_time - 5.0
        for object_id in list(self.active_objects.keys()):
            if self.active_objects[object_id]['last_seen'] < inactive_cutoff:
                del self.active_objects[object_id]


class AttentionAnalyzer:
    """Dikkat analizi sınıfı"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Dikkat analizi konfigürasyonu
        """
        self.config = config
        self.attention_radius = config.get('attention_radius', 50)
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.temporal_smoothing = config.get('temporal_smoothing', 0.7)
        self.max_history_length = config.get('max_history_length', 30)
        
        # Geçmiş dikkat verileri
        self.attention_history = deque(maxlen=self.max_history_length)
        
        # Obje dikkat tracker
        self.object_tracker = ObjectAttentionTracker()
        
        self.logger = logging.getLogger(__name__)
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
                'interest_level': 0.0,
                'object_attention_summary': self.object_tracker.get_attention_summary()
            }
        
        gaze_point = gaze_data['gaze_point']
        
        # Bakış noktasına yakın nesneleri bul
        focused_objects = self._find_objects_in_gaze(gaze_point, detections)
        
        # Obje dikkat tracker'ı güncelle
        self.object_tracker.update_attention(focused_objects)
        
        # Interaction event'leri tespit et (yeni objeler için focus_start)
        current_objects = set(obj.get('object_id', 'unknown') for obj in focused_objects)
        if hasattr(self, '_previous_focused_objects'):
            previous_objects = set(obj.get('object_id', 'unknown') for obj in self._previous_focused_objects)
            
            # Yeni focus başlangıçları
            new_focuses = current_objects - previous_objects
            ended_focuses = previous_objects - current_objects
            
            # Event detection results
            interaction_events = []
            for obj_id in new_focuses:
                obj = next((o for o in focused_objects if o.get('object_id') == obj_id), None)
                if obj:
                    interaction_events.append({
                        'event_type': 'focus_start',
                        'object_data': obj,
                        'timestamp': time.time()
                    })
            
            for obj_id in ended_focuses:
                obj = next((o for o in self._previous_focused_objects if o.get('object_id') == obj_id), None)
                if obj:
                    interaction_events.append({
                        'event_type': 'focus_end',
                        'object_data': obj,
                        'timestamp': time.time()
                    })
        else:
            interaction_events = []
        
        # Previous state'i güncelle
        self._previous_focused_objects = focused_objects.copy()
        
        # Dikkat dağılımını hesapla
        attention_distribution = self._calculate_attention_distribution(
            gaze_point, detections
        )
        
        # İlgi seviyesini hesapla
        interest_level = self._calculate_interest_level(focused_objects)
        
        # Gelişmiş analiz sonuçları
        result = {
            'gaze_point': gaze_point,
            'focused_objects': focused_objects,
            'attention_distribution': attention_distribution,
            'interest_level': interest_level,
            'gaze_stability': self._calculate_gaze_stability(gaze_point),
            'object_attention_summary': self.object_tracker.get_attention_summary(),
            'top_attended_objects': self._get_top_attended_objects(),
            'interaction_events': interaction_events  # Yeni: interaction events
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
            
            # Dikkat süresini ekle
            object_id = self.object_tracker._generate_object_id(detection)
            focused_obj['attention_duration'] = self.object_tracker.get_object_attention_duration(object_id)
            focused_obj['total_attention_score'] = self.object_tracker.get_object_total_attention_score(object_id)
            
            focused_objects.append(focused_obj)
        
        # Dikkat skoruna göre sırala
        focused_objects.sort(key=lambda x: x['attention_score'], reverse=True)
        
        return focused_objects
    
    def _get_top_attended_objects(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """En çok dikkat çeken objeleri al"""
        summary = self.object_tracker.get_attention_summary()
        return summary['objects'][:top_n]
    
    def _calculate_attention_distribution(self, gaze_point: Tuple[int, int], 
                                        detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """Nesne kategorilerine göre dikkat dağılımı"""
        distribution = {}
        total_attention = 0.0
        
        gx, gy = gaze_point
        
        for detection in detections:
            category = detection.get('label', 'unknown')  # 'type' yerine 'label' kullan
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
        unique_types = len(set(obj.get('label', 'unknown') for obj in focused_objects))
        
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
        recent_gazes = [h.get('gaze_point') for h in self.attention_history 
                       if h.get('gaze_point')][-5:]
        
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
        attention_result['timestamp'] = time.time()
        self.attention_history.append(attention_result)
    
    def get_attention_summary(self, time_window: int = 10) -> Dict[str, Any]:
        """Belirli zaman penceresi için dikkat özeti"""
        if len(self.attention_history) < time_window:
            recent_history = list(self.attention_history)
        else:
            recent_history = list(self.attention_history)[-time_window:]
        
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
            'sample_count': len(recent_history),
            'object_attention_tracker': self.object_tracker.get_attention_summary()
        } 