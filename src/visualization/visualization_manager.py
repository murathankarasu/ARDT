"""
Gelişmiş Görselleştirme Yöneticisi - 3D Gaze ve Çakışma Görselleştirmesi
"""

import cv2
import numpy as np
import math
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from attention import GazeCollisionEvent


class VisualizationManager:
    """Gelişmiş görselleştirme sistemi"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Görselleştirme konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Görselleştirme ayarları
        self.show_3d_gaze = config.get('show_3d_gaze', True)
        self.show_collisions = config.get('show_collisions', True)
        self.show_gaze_trail = config.get('show_gaze_trail', True)
        self.show_depth_info = config.get('show_depth_info', True)
        
        # Animasyon parametreleri
        self.animation_enabled = config.get('animation_enabled', True)
        self.pulse_speed = config.get('pulse_speed', 3.0)
        self.fade_duration = config.get('fade_duration', 2.0)
        
        # Renk paleti
        self.color_scheme = {
            'gaze_active': (0, 255, 255),      # Cyan - aktif gaze
            'gaze_fading': (0, 200, 200),      # Darker cyan - fading
            'collision': (0, 0, 255),          # Red - çakışma
            'collision_trail': (255, 0, 0),    # Blue - çakışma izi
            'depth_indicator': (255, 255, 0),  # Yellow - derinlik
            'connection_line': (0, 255, 0),    # Green - bağlantı çizgisi
            'info_text': (255, 255, 255),      # White - bilgi metni
            'warning_text': (0, 255, 255),     # Cyan - uyarı metni
            'success_text': (0, 255, 0)        # Green - başarı metni
        }
        
        # Gaze trail
        self.gaze_trail = []
        self.max_trail_length = 20
        
        # Çakışma efektleri
        self.collision_effects = []
        
        self.logger.info("VisualizationManager başlatıldı")
    
    def render_frame(self, frame: np.ndarray, gaze_3d_data: Dict[str, Any], 
                    detections: List[Dict[str, Any]], collision_events: List[GazeCollisionEvent],
                    attention_analysis: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Kompleks frame rendering
        
        Args:
            frame: Temel frame
            gaze_3d_data: 3D gaze verisi
            detections: Tespit edilen objeler
            collision_events: Çakışma olayları
            attention_analysis: Dikkat analizi (opsiyonel)
            
        Returns:
            Render edilmiş frame
        """
        result_frame = frame.copy()
        
        # 1. Temel obje tespitlerini çiz
        self._draw_object_detections(result_frame, detections)
        
        # 2. 3D Gaze objelerini çiz
        if self.show_3d_gaze:
            self._draw_3d_gaze_objects(result_frame, gaze_3d_data)
        
        # 3. Gaze trail çiz
        if self.show_gaze_trail:
            self._draw_gaze_trail(result_frame, gaze_3d_data)
        
        # 4. Çakışma efektlerini çiz
        if self.show_collisions:
            self._draw_collision_effects(result_frame, collision_events)
        
        # 5. Derinlik bilgilerini çiz
        if self.show_depth_info:
            self._draw_depth_information(result_frame, gaze_3d_data)
        
        # 6. Bilgi overlay'lerini çiz
        self._draw_info_overlays(result_frame, gaze_3d_data, collision_events, attention_analysis)
        
        # 7. Animasyonları güncelle
        if self.animation_enabled:
            self._update_animations()
        
        return result_frame
    
    def _draw_object_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """Gelişmiş obje tespiti çizimi"""
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0.0)
            category = detection.get('category', 'other')
            
            # Kategori bazlı renk seçimi
            color = self._get_category_color(category)
            
            # Ana bbox - çakışma durumuna göre kalınlık
            thickness = 4 if self._is_object_in_collision(detection) else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Çakışma durumunda ekstra efekt
            if self._is_object_in_collision(detection):
                # Pulsing effect
                pulse_size = int(5 * (math.sin(time.time() * 4) + 1))
                cv2.rectangle(frame, (x1-pulse_size, y1-pulse_size), 
                            (x2+pulse_size, y2+pulse_size), 
                            self.color_scheme['collision'], 2)
            
            # Label ve confidence
            label_text = f"{label}: {confidence:.2f}"
            
            # AI classification varsa ekle
            if 'ai_class' in detection:
                label_text += f" | AI: {detection['ai_class']}"
            
            # Label background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-25), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(frame, label_text, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_3d_gaze_objects(self, frame: np.ndarray, gaze_3d_data: Dict[str, Any]):
        """3D gaze objelerini çiz"""
        gaze_objects = gaze_3d_data.get('gaze_objects', [])
        
        for gaze_obj in gaze_objects:
            # 2D projeksiyonu al - OpenCV için integer tuple'a çevir
            proj_2d = tuple(map(int, gaze_obj.get_2d_projection(frame.shape)))
            
            # Yaş ve derinlik bazlı görselleştirme
            age = time.time() - gaze_obj.creation_time
            alpha = max(0.1, 1.0 - age / 5.0)  # 5 saniyede fade
            
            # Çember çiz - pulse efekti ile
            current_size = int(gaze_obj.size)
            
            # Dış çerçeve
            cv2.circle(frame, proj_2d, current_size + 5, 
                      self.color_scheme['gaze_active'], 3)
            
            # Ana çember
            cv2.circle(frame, proj_2d, current_size, 
                      gaze_obj.color, -1)
            
            # İç highlight
            cv2.circle(frame, proj_2d, current_size // 2, 
                      (255, 255, 255), -1)
            
            # Derinlik göstergesi
            depth_text = f"{gaze_obj.position[2]:.0f}"
            cv2.putText(frame, depth_text, 
                       (proj_2d[0] - 10, proj_2d[1] - current_size - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.color_scheme['depth_indicator'], 2)
            
            # 3D koordinat bilgisi
            coord_text = f"({gaze_obj.position[0]:.0f}, {gaze_obj.position[1]:.0f}, {gaze_obj.position[2]:.0f})"
            cv2.putText(frame, coord_text, 
                       (proj_2d[0] - 30, proj_2d[1] + current_size + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       self.color_scheme['info_text'], 1)
    
    def _draw_gaze_trail(self, frame: np.ndarray, gaze_3d_data: Dict[str, Any]):
        """Gaze trail çizimi"""
        gaze_3d_vector = gaze_3d_data.get('gaze_3d_vector')
        
        if gaze_3d_vector:
            # Trail'e yeni nokta ekle
            current_time = time.time()
            trail_point = {
                'position': gaze_3d_vector[:2],  # Sadece 2D
                'timestamp': current_time,
                'depth': gaze_3d_vector[2]
            }
            
            self.gaze_trail.append(trail_point)
            
            # Eski noktaları temizle
            self.gaze_trail = [p for p in self.gaze_trail 
                             if current_time - p['timestamp'] < 3.0]
            
            # Trail çiz
            if len(self.gaze_trail) > 1:
                for i in range(1, len(self.gaze_trail)):
                    prev_point = self.gaze_trail[i-1]
                    curr_point = self.gaze_trail[i]
                    
                    # Yaş bazlı alpha
                    age = current_time - prev_point['timestamp']
                    alpha = max(0.1, 1.0 - age / 3.0)
                    
                    # Derinlik bazlı renk
                    depth_factor = min(1.0, prev_point['depth'] / 500.0)
                    color_intensity = int(255 * alpha * depth_factor)
                    
                    # Çizgi çiz - koordinatları integer'a çevir
                    prev_pos = tuple(map(int, prev_point['position']))
                    curr_pos = tuple(map(int, curr_point['position']))
                    cv2.line(frame, prev_pos, curr_pos,
                            (color_intensity, color_intensity, 255), 
                            max(1, int(3 * alpha)))
    
    def _draw_collision_effects(self, frame: np.ndarray, collision_events: List[GazeCollisionEvent]):
        """Çakışma efektlerini çiz"""
        for event in collision_events:
            if not event.is_active:
                continue
                
            # Çakışma noktası - OpenCV için integer tuple'a çevir
            gaze_point = tuple(map(int, event.gaze_point))
            
            # Pulsing collision indicator
            pulse_factor = (math.sin(time.time() * 6) + 1) / 2
            pulse_size = int(20 + pulse_factor * 15)
            
            # Dış çerçeve - red
            cv2.circle(frame, gaze_point, pulse_size + 5, 
                      self.color_scheme['collision'], 4)
            
            # İç çerçeve - brighter red
            cv2.circle(frame, gaze_point, pulse_size, 
                      (0, 100, 255), 3)
            
            # Merkez nokta
            cv2.circle(frame, gaze_point, 5, (255, 255, 255), -1)
            
            # Çakışma bilgisi
            obj_label = event.object_data.get('label', 'Unknown')
            duration_text = f"🎯 {obj_label} - {event.duration:.1f}s"
            depth_text = f"Derinlik: {event.collision_depth:.1f}"
            
            # Bilgi background
            text_y = gaze_point[1] - 40
            cv2.rectangle(frame, (gaze_point[0] - 60, text_y - 25), 
                         (gaze_point[0] + 150, text_y + 10), 
                         (0, 0, 0), -1)
            
            # Bilgi metni
            cv2.putText(frame, duration_text, 
                       (gaze_point[0] - 55, text_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.color_scheme['collision'], 2)
            
            cv2.putText(frame, depth_text, 
                       (gaze_point[0] - 55, text_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       self.color_scheme['warning_text'], 1)
            
            # Çakışan objeye bağlantı çizgisi
            obj_bbox = event.object_data.get('bbox', [])
            if len(obj_bbox) == 4:
                obj_center = (int((obj_bbox[0] + obj_bbox[2]) // 2), 
                            int((obj_bbox[1] + obj_bbox[3]) // 2))
                
                # Animasyonlu bağlantı çizgisi
                line_alpha = (math.sin(time.time() * 4) + 1) / 2
                line_color = tuple(int(c * line_alpha) for c in self.color_scheme['connection_line'])
                
                cv2.line(frame, gaze_point, obj_center, line_color, 3)
    
    def _draw_depth_information(self, frame: np.ndarray, gaze_3d_data: Dict[str, Any]):
        """Derinlik bilgilerini çiz"""
        depth_estimate = gaze_3d_data.get('depth_estimate', 0.0)
        
        if depth_estimate > 0:
            # Derinlik bar
            bar_x = 20
            bar_y = 100
            bar_width = 200
            bar_height = 20
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Derinlik değeri (0-1000 arası normalize)
            depth_normalized = min(1.0, depth_estimate / 1000.0)
            fill_width = int(bar_width * depth_normalized)
            
            # Derinlik rengini hesapla (yakın: yeşil, uzak: kırmızı)
            if depth_normalized < 0.3:
                bar_color = (0, 255, 0)  # Yakın - yeşil
            elif depth_normalized < 0.7:
                bar_color = (0, 255, 255)  # Orta - sarı
            else:
                bar_color = (0, 0, 255)  # Uzak - kırmızı
            
            # Derinlik bar'ı çiz
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + fill_width, bar_y + bar_height),
                         bar_color, -1)
            
            # Derinlik metni
            depth_text = f"Derinlik: {depth_estimate:.1f}m"
            cv2.putText(frame, depth_text, (bar_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.color_scheme['depth_indicator'], 2)
    
    def _draw_info_overlays(self, frame: np.ndarray, gaze_3d_data: Dict[str, Any], 
                           collision_events: List[GazeCollisionEvent],
                           attention_analysis: Optional[Dict[str, Any]]):
        """Bilgi overlay'lerini çiz"""
        h, w = frame.shape[:2]
        
        # Üst bilgi paneli
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Başlık
        cv2.putText(frame, "🎯 3D Gaze Tracking & Collision Detection", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   self.color_scheme['info_text'], 2)
        
        # İstatistikler
        stats_y = 45
        
        # Aktif gaze objesi sayısı
        gaze_count = len(gaze_3d_data.get('gaze_objects', []))
        cv2.putText(frame, f"Aktif 3D Gaze: {gaze_count}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   self.color_scheme['success_text'], 1)
        
        # Aktif çakışma sayısı
        active_collisions = len([e for e in collision_events if e.is_active])
        collision_color = self.color_scheme['collision'] if active_collisions > 0 else self.color_scheme['info_text']
        cv2.putText(frame, f"Aktif Çakışma: {active_collisions}", 
                   (200, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   collision_color, 1)
        
        # Derinlik bilgisi
        depth = gaze_3d_data.get('depth_estimate', 0.0)
        cv2.putText(frame, f"Ortalama Derinlik: {depth:.1f}m", 
                   (400, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   self.color_scheme['depth_indicator'], 1)
        
        # Attention analysis bilgisi
        if attention_analysis:
            interest_level = attention_analysis.get('interest_level', 0.0)
            cv2.putText(frame, f"İlgi Seviyesi: {interest_level*100:.1f}%", 
                       (10, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.color_scheme['warning_text'], 1)
        
        # Kontrol bilgileri
        controls_y = stats_y + 50
        cv2.putText(frame, "Kontroller: 'q'=Çıkış, 's'=Screenshot, 'r'=Reset, 'c'=Çakışma Temizle", 
                   (10, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   self.color_scheme['info_text'], 1)
    
    def _get_category_color(self, category: str) -> Tuple[int, int, int]:
        """Kategori bazlı renk döndür"""
        color_map = {
            'traffic_sign': (0, 255, 0),
            'billboard': (255, 0, 0),
            'street_sign': (0, 255, 255),
            'shelf': (128, 0, 128),
            'product': (255, 128, 0),
            'furniture': (255, 100, 200),
            'electronics': (100, 100, 255),
            'other': (255, 255, 255)
        }
        return color_map.get(category, (255, 255, 255))
    
    def _is_object_in_collision(self, detection: Dict[str, Any]) -> bool:
        """Objenin çakışma durumunu kontrol et"""
        # Bu collision_events ile karşılaştırılması gerekecek
        # Şimdilik basit implementasyon
        return False
    
    def _update_animations(self):
        """Animasyonları güncelle"""
        # Çakışma efektlerini güncelle
        current_time = time.time()
        
        # Eski efektleri temizle
        self.collision_effects = [
            effect for effect in self.collision_effects
            if current_time - effect.get('timestamp', 0) < 3.0
        ]
    
    def reset_trail(self):
        """Gaze trail'i temizle"""
        self.gaze_trail.clear()
        self.logger.info("Gaze trail temizlendi")
    
    def reset_collision_effects(self):
        """Çakışma efektlerini temizle"""
        self.collision_effects.clear()
        self.logger.info("Çakışma efektleri temizlendi")
    
    def toggle_3d_gaze(self):
        """3D gaze görüntüsünü aç/kapat"""
        self.show_3d_gaze = not self.show_3d_gaze
        self.logger.info(f"3D Gaze görüntü: {'Açık' if self.show_3d_gaze else 'Kapalı'}")
    
    def toggle_collisions(self):
        """Çakışma görüntüsünü aç/kapat"""
        self.show_collisions = not self.show_collisions
        self.logger.info(f"Çakışma görüntü: {'Açık' if self.show_collisions else 'Kapalı'}")
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """Görselleştirme istatistikleri"""
        return {
            'gaze_trail_length': len(self.gaze_trail),
            'collision_effects_count': len(self.collision_effects),
            'show_3d_gaze': self.show_3d_gaze,
            'show_collisions': self.show_collisions,
            'show_gaze_trail': self.show_gaze_trail,
            'animation_enabled': self.animation_enabled
        } 