"""
Adaptif Kamera Sistemi - Düşük Donanım Optimized
"""

import cv2
import numpy as np
import logging
import threading
import time
import psutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SceneType(Enum):
    """Sahne türleri"""
    STREET = "street"
    STORE = "store"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    NIGHT = "night"
    DAY = "day"


class PerformanceMode(Enum):
    """Performans modları"""
    POWER_SAVE = "power_save"      # Minimum CPU/GPU kullanımı
    BALANCED = "balanced"          # Dengeli performans
    PERFORMANCE = "performance"    # Maksimum kalite
    ADAPTIVE = "adaptive"          # Otomatik ayarlama


class QualityPreset(Enum):
    """Kalite ön ayarları"""
    LOW = "low"          # 720p, 15 FPS
    MEDIUM = "medium"    # 1080p, 30 FPS  
    HIGH = "high"        # 1440p, 30 FPS
    ULTRA = "ultra"      # 4K, 60 FPS


@dataclass
class CameraSettings:
    """Kamera ayarları"""
    brightness: float = 0.5
    contrast: float = 1.0
    saturation: float = 1.0
    exposure: float = -1  # Auto exposure
    white_balance: float = -1  # Auto white balance
    fps: int = 20  # Daha stabil FPS (hızlı hareket sorununu önler)
    resolution: Tuple[int, int] = (1280, 720)
    focus: float = -1  # Auto focus
    
    # Yeni optimizasyon ayarları
    quality_preset: QualityPreset = QualityPreset.MEDIUM
    performance_mode: PerformanceMode = PerformanceMode.ADAPTIVE
    enable_post_processing: bool = True
    buffer_frames: int = 1  # Frame buffer boyutu
    
    # Stabilizasyon ayarları (yanıp sönme önleme) - Ultra stabilize
    brightness_smoothing: float = 0.8  # Yüksek yumuşatma faktörü
    contrast_smoothing: float = 0.8    # Yüksek yumuşatma faktörü
    transition_speed: float = 0.1      # Çok yavaş geçiş (hızlı değişimleri önler)
    min_change_threshold: float = 0.2  # Daha büyük eşik (küçük değişimleri yok say)
    manual_exposure_enabled: bool = False  # Manuel exposure modu
    manual_wb_enabled: bool = False    # Manuel white balance modu


@dataclass
class SystemResources:
    """Sistem kaynak durumu"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    temperature: float = 0.0


class AdaptiveCamera:
    """Adaptif kamera sistemi - Optimize edilmiş"""
    
    # Kalite presetleri
    QUALITY_PRESETS = {
        QualityPreset.LOW: {
            'resolution': (1280, 720),
            'fps': 10,  # Çok yavaş - stabil
            'post_processing': False,
            'buffer_frames': 1
        },
        QualityPreset.MEDIUM: {
            'resolution': (1920, 1080),
            'fps': 15,  # Stabil FPS
            'post_processing': True,
            'buffer_frames': 2
        },
        QualityPreset.HIGH: {
            'resolution': (2560, 1440),
            'fps': 20,  # Stabil FPS
            'post_processing': True,
            'buffer_frames': 3
        },
        QualityPreset.ULTRA: {
            'resolution': (3840, 2160),
            'fps': 25,  # Çok düşük FPS (stabilite için)
            'post_processing': True,
            'buffer_frames': 4
        }
    }
    
    def __init__(self, camera_id: int = 0, config: dict = None):
        """
        Args:
            camera_id: Kamera ID
            config: Konfigürasyon
        """
        self.camera_id = camera_id
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Kamera nesnesi
        self.cap = None
        
        # Adaptif ayarlar
        self.current_settings = CameraSettings()
        self.target_settings = CameraSettings()
        
        # Sahne analizi
        self.current_scene = SceneType.DAY
        self.scene_confidence = 0.0
        
        # Sistem kaynak izleme
        self.system_resources = SystemResources()
        self.resource_monitor_interval = 2.0  # 2 saniye
        
        # İstatistikler
        self.frame_stats = {
            'brightness_history': [],
            'contrast_history': [],
            'motion_history': [],
            'quality_history': [],
            'processing_time_history': []
        }
        
        # Threading
        self.adaptive_thread = None
        self.resource_thread = None
        self.running = False
        
        # Performans izleme
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0
        self.target_fps = 30
        
        # Otomatik optimizasyon
        self.auto_quality_adjust = True
        self.last_quality_adjustment = time.time()
        self.performance_degradation_count = 0
        
        # Stabilizasyon ayarlarını konfigürasyondan oku
        self._load_stabilization_config()
        
        # Başlangıç kalite presetini belirle
        self._determine_initial_quality()
        self._initialize_camera()
        
    def _load_stabilization_config(self):
        """Stabilizasyon ayarlarını konfigürasyondan yükle"""
        try:
            stabilization_config = self.config.get('stabilization', {})
            
            # Stabilizasyon ayarlarını current_settings'e uygula
            self.current_settings.brightness_smoothing = stabilization_config.get('brightness_smoothing', 0.5)
            self.current_settings.contrast_smoothing = stabilization_config.get('contrast_smoothing', 0.5)
            self.current_settings.transition_speed = stabilization_config.get('transition_speed', 0.5)
            self.current_settings.min_change_threshold = stabilization_config.get('min_change_threshold', 0.05)
            
            # Manuel exposure ve white balance ayarları
            self.current_settings.manual_exposure_enabled = not self.config.get('auto_exposure', True)
            self.current_settings.manual_wb_enabled = not self.config.get('auto_white_balance', True)
            
            if self.current_settings.manual_exposure_enabled:
                self.current_settings.exposure = self.config.get('manual_exposure', 0.4)
                
            if self.current_settings.manual_wb_enabled:
                self.current_settings.white_balance = self.config.get('manual_white_balance', 4600)
            
            # FPS ve çözünürlük ayarları
            resolution_config = self.config.get('resolution', {})
            if resolution_config:
                width = resolution_config.get('width', 1280)
                height = resolution_config.get('height', 720)
                self.current_settings.resolution = (width, height)
            
            self.current_settings.fps = self.config.get('fps', 30)
            self.current_settings.buffer_frames = self.config.get('buffer_size', 1)
            
            self.logger.info("Stabilizasyon ayarları yüklendi")
            self.logger.info(f"Parlaklık yumuşatma: {self.current_settings.brightness_smoothing}")
            self.logger.info(f"Geçiş hızı: {self.current_settings.transition_speed}")
            self.logger.info(f"Manuel exposure: {self.current_settings.manual_exposure_enabled}")
            
        except Exception as e:
            self.logger.error(f"Stabilizasyon ayarları yükleme hatası: {e}")
        
    def _determine_initial_quality(self):
        """Sistem performansına göre başlangıç kalitesini belirle"""
        # Sistem özelliklerini kontrol et
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        cpu_count = psutil.cpu_count()
        
        # Basit heuristik ile kalite presetini belirle
        if total_memory < 4 or cpu_count < 4:
            self.current_settings.quality_preset = QualityPreset.LOW
            self.logger.info("Düşük sistem kaynakları tespit edildi - LOW kalite modu")
        elif total_memory < 8 or cpu_count < 8:
            self.current_settings.quality_preset = QualityPreset.MEDIUM
            self.logger.info("Orta sistem kaynakları tespit edildi - MEDIUM kalite modu")
        elif total_memory < 16:
            self.current_settings.quality_preset = QualityPreset.HIGH
            self.logger.info("Yüksek sistem kaynakları tespit edildi - HIGH kalite modu")
        else:
            self.current_settings.quality_preset = QualityPreset.ULTRA
            self.logger.info("Çok yüksek sistem kaynakları tespit edildi - ULTRA kalite modu")
        
        # Preset ayarlarını uygula
        self._apply_quality_preset(self.current_settings.quality_preset)
        
    def _apply_quality_preset(self, preset: QualityPreset):
        """Kalite presetini uygula"""
        preset_config = self.QUALITY_PRESETS[preset]
        
        self.current_settings.resolution = preset_config['resolution']
        self.current_settings.fps = preset_config['fps']
        self.current_settings.enable_post_processing = preset_config['post_processing']
        self.current_settings.buffer_frames = preset_config['buffer_frames']
        self.target_fps = preset_config['fps']
        
        self.logger.info(f"Kalite preset uygulandı: {preset.value} - {preset_config['resolution']}@{preset_config['fps']}fps")
        
    def _initialize_camera(self):
        """Kamerayı başlat"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Kamera {self.camera_id} açılamadı")
            
            # Buffer boyutunu minimize et (düşük latency için)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.current_settings.buffer_frames)
            
            # Başlangıç ayarları
            self._apply_settings(self.current_settings)
            
            # Adaptif thread başlat
            self.running = True
            self.adaptive_thread = threading.Thread(target=self._adaptive_loop)
            self.adaptive_thread.daemon = True
            self.adaptive_thread.start()
            
            # Kaynak izleme thread başlat
            self.resource_thread = threading.Thread(target=self._resource_monitor_loop)
            self.resource_thread.daemon = True
            self.resource_thread.start()
            
            self.logger.info(f"Optimize edilmiş adaptif kamera sistemi başlatıldı: {self.camera_id}")
            
        except Exception as e:
            self.logger.error(f"Kamera başlatma hatası: {e}")
            raise
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Stabilize edilmiş frame al"""
        if self.cap is None:
            return None
        
        start_time = time.time()
        
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        # Renk formatını kontrol et (siyah beyaz sorununu çözer)
        if len(frame.shape) == 2:  # Grayscale ise BGR'ye çevir
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # FPS sayacı
        self._update_fps_counter()
        
        # Frame stabilizasyonu (hızlı hareket sorununu çözer)
        frame = self._stabilize_frame(frame)
        
        # Frame kalitesini analiz et (sadece adaptive modda ve daha az sıklıkta)
        if (self.current_settings.performance_mode == PerformanceMode.ADAPTIVE and 
            self.fps_counter % 30 == 0):  # Her 30 frame'de bir analiz et
            self._analyze_frame_quality(frame)
        
        # Post-processing (eğer etkinse ve daha yumuşak)
        if self.current_settings.enable_post_processing:
            frame = self._post_process_frame_smooth(frame)
        
        # İşlem süresini kaydet
        processing_time = time.time() - start_time
        self.frame_stats['processing_time_history'].append(processing_time)
        
        return frame
    
    def _stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame stabilizasyonu - hızlı hareket ve titreme önleme"""
        try:
            # Parlaklık stabilizasyonu
            if hasattr(self, 'prev_brightness'):
                current_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                
                # Ani parlaklık değişimlerini yumuşat
                brightness_diff = abs(current_brightness - self.prev_brightness)
                if brightness_diff > 30:  # Büyük değişimse yumuşat
                    alpha = 0.7  # Geçmiş frame ağırlığı
                    frame = cv2.addWeighted(frame, 1-alpha, self.prev_frame if hasattr(self, 'prev_frame') else frame, alpha, 0)
                
                self.prev_brightness = current_brightness * 0.7 + self.prev_brightness * 0.3
            else:
                self.prev_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
            # Önceki frame'i sakla
            self.prev_frame = frame.copy()
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"Frame stabilizasyon hatası: {e}")
            return frame
    
    def _post_process_frame_smooth(self, frame: np.ndarray) -> np.ndarray:
        """Yumuşak post-processing - siyah beyaz sorununu önler"""
        try:
            # Çok hafif CLAHE (aşırı kontrast değişimini önler)
            if self.fps_counter % 10 == 0:  # Her 10 frame'de bir uygula
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))  # Çok hafif
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"Post-processing hatası: {e}")
            return frame
    
    def _resource_monitor_loop(self):
        """Sistem kaynaklarını izleme döngüsü"""
        while self.running:
            try:
                # CPU kullanımı
                self.system_resources.cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Memory kullanımı
                memory = psutil.virtual_memory()
                self.system_resources.memory_percent = memory.percent
                
                # GPU memory (opsiyonel)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.system_resources.gpu_memory_percent = gpus[0].memoryUtil * 100
                except:
                    pass
                
                # Performans degradation kontrolü
                if self.auto_quality_adjust:
                    self._check_performance_degradation()
                
                time.sleep(self.resource_monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Kaynak izleme hatası: {e}")
                time.sleep(5)
    
    def _check_performance_degradation(self):
        """Performans degradation kontrolü"""
        current_time = time.time()
        
        # En az 5 saniye bekle
        if current_time - self.last_quality_adjustment < 5:
            return
        
        # CPU kullanımı çok yüksekse
        if self.system_resources.cpu_percent > 85:
            self._downgrade_quality()
            return
        
        # Memory kullanımı çok yüksekse
        if self.system_resources.memory_percent > 90:
            self._downgrade_quality()
            return
        
        # FPS hedefe ulaşamıyorsa
        if self.actual_fps < self.target_fps * 0.8:
            self.performance_degradation_count += 1
            if self.performance_degradation_count >= 3:
                self._downgrade_quality()
                self.performance_degradation_count = 0
        else:
            self.performance_degradation_count = 0
            # Performans iyiyse kaliteyi artırabilir miyiz?
            self._try_upgrade_quality()
    
    def _downgrade_quality(self):
        """Kaliteyi düşür"""
        current_preset = self.current_settings.quality_preset
        
        if current_preset == QualityPreset.ULTRA:
            new_preset = QualityPreset.HIGH
        elif current_preset == QualityPreset.HIGH:
            new_preset = QualityPreset.MEDIUM
        elif current_preset == QualityPreset.MEDIUM:
            new_preset = QualityPreset.LOW
        else:
            # Zaten en düşük kalitede
            # Post-processing'i kapat
            self.current_settings.enable_post_processing = False
            self.logger.warning("Post-processing kapatıldı - performans optimizasyonu")
            return
        
        self._apply_quality_preset(new_preset)
        self.current_settings.quality_preset = new_preset
        self._apply_settings(self.current_settings)
        self.last_quality_adjustment = time.time()
        
        self.logger.warning(f"Kalite düşürüldü: {new_preset.value} - performans optimizasyonu")
    
    def _try_upgrade_quality(self):
        """Kaliteyi artırmayı dene"""
        # Sadece sistem kaynakları uygunsa
        if (self.system_resources.cpu_percent < 60 and 
            self.system_resources.memory_percent < 70 and
            self.actual_fps >= self.target_fps * 0.95):
            
            current_preset = self.current_settings.quality_preset
            
            if current_preset == QualityPreset.LOW:
                new_preset = QualityPreset.MEDIUM
            elif current_preset == QualityPreset.MEDIUM:
                new_preset = QualityPreset.HIGH
            elif current_preset == QualityPreset.HIGH:
                new_preset = QualityPreset.ULTRA
            else:
                return  # Zaten en yüksek kalite
            
            self._apply_quality_preset(new_preset)
            self.current_settings.quality_preset = new_preset
            self._apply_settings(self.current_settings)
            self.last_quality_adjustment = time.time()
            
            self.logger.info(f"Kalite artırıldı: {new_preset.value} - yeterli sistem kaynağı")
    
    def _adaptive_loop(self):
        """Adaptif ayarlama döngüsü"""
        while self.running:
            try:
                # Sahne analizi (sadece adaptive modda)
                if self.current_settings.performance_mode == PerformanceMode.ADAPTIVE:
                    scene_type = self._analyze_current_scene()
                    
                    # Ayarları güncelle
                    if scene_type != self.current_scene:
                        self.current_scene = scene_type
                        self.target_settings = self._get_optimal_settings(scene_type)
                        self._smooth_transition_to_target()
                
                # İstatistikleri temizle
                self._cleanup_stats()
                
                time.sleep(3.0)  # 3 saniye aralıklarla kontrol (daha stabilize)
                
            except Exception as e:
                self.logger.error(f"Adaptif döngü hatası: {e}")
                time.sleep(2)
    
    def _analyze_current_scene(self) -> SceneType:
        """Mevcut sahneyi analiz et"""
        if not self.frame_stats['brightness_history']:
            return self.current_scene
        
        # Son frame'lerin ortalama parlaklığı
        avg_brightness = np.mean(self.frame_stats['brightness_history'][-5:])
        
        # Kontrast analizi
        avg_contrast = np.mean(self.frame_stats['contrast_history'][-5:])
        
        # Hareket analizi
        avg_motion = np.mean(self.frame_stats['motion_history'][-3:]) if self.frame_stats['motion_history'] else 0
        
        # Sahne sınıflandırması (optimize edilmiş)
        if avg_brightness < 80:
            scene_type = SceneType.NIGHT
        elif avg_brightness > 180:
            scene_type = SceneType.DAY
        else:
            # Hareket ve kontrast analizi
            if avg_motion > 0.3:  # Yüksek hareket = sokak
                scene_type = SceneType.STREET
            elif avg_contrast > 1.2:  # Yüksek kontrast = kapalı alan
                scene_type = SceneType.INDOOR
            else:
                scene_type = SceneType.STORE
        
        return scene_type
    
    def _get_optimal_settings(self, scene_type: SceneType) -> CameraSettings:
        """Sahne türüne göre optimal ayarları hesapla"""
        settings = CameraSettings()
        
        # Mevcut kalite presetini koru
        settings.quality_preset = self.current_settings.quality_preset
        settings.performance_mode = self.current_settings.performance_mode
        
        if scene_type == SceneType.NIGHT:
            # Gece modu - daha az FPS, daha fazla pozlama
            settings.brightness = 0.7
            settings.contrast = 1.3
            settings.saturation = 0.9
            settings.exposure = 0.3
            # FPS'i %25 düşür
            settings.fps = max(15, int(self.current_settings.fps * 0.75))
            
        elif scene_type == SceneType.DAY:
            # Gündüz modu - standart ayarlar
            settings.brightness = 0.5
            settings.contrast = 1.0
            settings.saturation = 1.1
            settings.exposure = -0.3
            settings.fps = self.current_settings.fps
            
        elif scene_type == SceneType.STREET:
            # Sokak modu - hızlı hareket, yüksek FPS
            settings.brightness = 0.6
            settings.contrast = 1.1
            settings.saturation = 1.0
            settings.exposure = -0.2
            # FPS'i %10 artır (maksimum sınırına kadar)
            settings.fps = min(60, int(self.current_settings.fps * 1.1))
            
        elif scene_type == SceneType.STORE:
            # Mağaza modu - detay odaklı, orta FPS
            settings.brightness = 0.5
            settings.contrast = 1.2
            settings.saturation = 1.2
            settings.exposure = 0.0
            settings.fps = max(20, int(self.current_settings.fps * 0.9))
            
        elif scene_type == SceneType.INDOOR:
            # Kapalı alan - dengeli ayarlar
            settings.brightness = 0.6
            settings.contrast = 1.1
            settings.saturation = 1.0
            settings.exposure = 0.1
            settings.fps = self.current_settings.fps
        
        # Diğer ayarları koru
        settings.resolution = self.current_settings.resolution
        settings.enable_post_processing = self.current_settings.enable_post_processing
        settings.buffer_frames = self.current_settings.buffer_frames
        
        return settings
    
    def _smooth_transition_to_target(self):
        """Hedef ayarlara yumuşak geçiş - Stabilizasyon ile yanıp sönme önleme"""
        try:
            # Stabilizasyon ayarlarını kullan
            transition_speed = self.current_settings.transition_speed
            min_threshold = self.current_settings.min_change_threshold
            brightness_smoothing = self.current_settings.brightness_smoothing
            contrast_smoothing = self.current_settings.contrast_smoothing
            
            # Parlaklık yumuşak geçiş - eğer değişim yeterince büyükse
            brightness_diff = abs(self.target_settings.brightness - self.current_settings.brightness)
            if brightness_diff > min_threshold:
                alpha_brightness = transition_speed * brightness_smoothing
                self.current_settings.brightness += alpha_brightness * (
                    self.target_settings.brightness - self.current_settings.brightness
                )
            
            # Kontrast yumuşak geçiş
            contrast_diff = abs(self.target_settings.contrast - self.current_settings.contrast) 
            if contrast_diff > min_threshold:
                alpha_contrast = transition_speed * contrast_smoothing
                self.current_settings.contrast += alpha_contrast * (
                    self.target_settings.contrast - self.current_settings.contrast
                )
            
            # Saturasyon yumuşak geçiş
            saturation_diff = abs(self.target_settings.saturation - self.current_settings.saturation)
            if saturation_diff > min_threshold:
                alpha_saturation = transition_speed * 0.8  # Saturasyon için daha yavaş
                self.current_settings.saturation += alpha_saturation * (
                    self.target_settings.saturation - self.current_settings.saturation
                )
            
            # Exposure yumuşak geçiş (manuel modda)
            if not self.current_settings.manual_exposure_enabled:
                exposure_diff = abs(self.target_settings.exposure - self.current_settings.exposure)
                if exposure_diff > min_threshold:
                    alpha_exposure = transition_speed * 0.3  # Exposure için çok yavaş
                    self.current_settings.exposure += alpha_exposure * (
                        self.target_settings.exposure - self.current_settings.exposure
                    )
            
            # FPS değişimi (sadece büyük farklar için)
            fps_diff = abs(self.target_settings.fps - self.current_settings.fps)
            if fps_diff >= 5:  # En az 5 FPS fark
                self.current_settings.fps = self.target_settings.fps
            
            # Ayarları uygula
            self._apply_settings(self.current_settings)
            
            self.logger.debug(f"Yumuşak geçiş: B={self.current_settings.brightness:.2f}, "
                            f"C={self.current_settings.contrast:.2f}, "
                            f"S={self.current_settings.saturation:.2f}")
            
        except Exception as e:
            self.logger.error(f"Yumuşak geçiş hatası: {e}")
            # Hata durumunda ayarları direkt uygula
            self._apply_settings(self.current_settings)
    
    def _apply_settings(self, settings: CameraSettings):
        """Ayarları kameraya uygula - Stabilizasyon desteği ile"""
        if self.cap is None:
            return
        
        try:
            # Temel kamera özellikleri
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, settings.brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, settings.contrast)
            self.cap.set(cv2.CAP_PROP_SATURATION, settings.saturation)
            
            # Exposure kontrolü - manuel mod destekli
            if settings.manual_exposure_enabled:
                # Manuel exposure modu - yanıp sönme önleme için sabit değer
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manuel mod
                self.cap.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
                self.logger.debug(f"Manuel exposure uygulandı: {settings.exposure}")
            else:
                # Otomatik exposure modu
                if settings.exposure >= 0:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
                else:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto mode
            
            # White Balance kontrolü - manuel mod destekli
            if settings.manual_wb_enabled:
                # Manuel white balance - yanıp sönme önleme için sabit değer
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Manuel mod
                if settings.white_balance > 0:
                    self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, settings.white_balance)
                self.logger.debug(f"Manuel white balance uygulandı: {settings.white_balance}")
            else:
                # Otomatik white balance
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Auto mode
            
            # FPS ayarla - büyük buffer ile smooth görüntü
            self.cap.set(cv2.CAP_PROP_FPS, settings.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, settings.buffer_frames)
            
            # Çözünürlük ayarla
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1])
            
            # Ek stabilizasyon ayarları
            self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0)  # Backlight compensation kapalı
            
            self.logger.debug(f"Kamera ayarları uygulandı: {settings.resolution}@{settings.fps}fps, "
                            f"B={settings.brightness:.2f}, C={settings.contrast:.2f}")
            
        except Exception as e:
            self.logger.error(f"Ayar uygulama hatası: {e}")
            # Temel ayarları deneme
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, settings.fps)
            except Exception as e2:
                self.logger.error(f"Temel ayar hatası: {e2}")
    
    def _analyze_frame_quality(self, frame: np.ndarray):
        """Frame kalitesini analiz et"""
        try:
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Parlaklık
            brightness = np.mean(gray)
            self.frame_stats['brightness_history'].append(brightness)
            
            # Kontrast (standart sapma)
            contrast = np.std(gray)
            self.frame_stats['contrast_history'].append(contrast)
            
            # Hareket tespiti
            if len(self.frame_stats['brightness_history']) > 1:
                motion = self._calculate_motion(gray)
                self.frame_stats['motion_history'].append(motion)
            
            # Kalite skoru
            quality = self._calculate_quality_score(frame)
            self.frame_stats['quality_history'].append(quality)
            
        except Exception as e:
            self.logger.error(f"Frame analiz hatası: {e}")
    
    def _calculate_motion(self, current_gray: np.ndarray) -> float:
        """Hareket miktarını hesapla"""
        if not hasattr(self, 'previous_gray'):
            self.previous_gray = current_gray
            return 0.0
        
        # Frame farkı
        diff = cv2.absdiff(self.previous_gray, current_gray)
        motion_score = np.mean(diff) / 255.0
        
        self.previous_gray = current_gray
        return motion_score
    
    def _calculate_quality_score(self, frame: np.ndarray) -> float:
        """Frame kalite skoru"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Laplacian varyansı (blur tespiti)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalizasyon
        quality_score = min(laplacian_var / 1000.0, 1.0)
        
        return quality_score
    
    def _post_process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame post-processing"""
        try:
            # Adaptif histogram eşitleme
            if self.current_scene in [SceneType.NIGHT, SceneType.INDOOR]:
                frame = self._apply_clahe(frame)
            
            # Gürültü azaltma
            if self.current_scene == SceneType.NIGHT:
                frame = cv2.bilateralFilter(frame, 5, 50, 50)
            
            # Keskinlik artırma
            if self.current_scene in [SceneType.STORE, SceneType.INDOOR]:
                frame = self._apply_sharpening(frame)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Post-processing hatası: {e}")
            return frame
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _apply_sharpening(self, frame: np.ndarray) -> np.ndarray:
        """Keskinlik artırma"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        return cv2.addWeighted(frame, 0.7, sharpened, 0.3, 0)
    
    def _update_fps_counter(self):
        """FPS sayacını güncelle"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.actual_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _cleanup_stats(self):
        """İstatistikleri temizle"""
        max_history = 50
        
        for key in self.frame_stats:
            if len(self.frame_stats[key]) > max_history:
                self.frame_stats[key] = self.frame_stats[key][-max_history:]
    
    def get_camera_status(self) -> Dict[str, any]:
        """Kamera durumu bilgisi"""
        return {
            'current_scene': self.current_scene.value,
            'scene_confidence': self.scene_confidence,
            'actual_fps': self.actual_fps,
            'settings': {
                'brightness': self.current_settings.brightness,
                'contrast': self.current_settings.contrast,
                'saturation': self.current_settings.saturation,
                'exposure': self.current_settings.exposure
            },
            'quality_metrics': {
                'avg_brightness': np.mean(self.frame_stats['brightness_history'][-10:]) if self.frame_stats['brightness_history'] else 0,
                'avg_contrast': np.mean(self.frame_stats['contrast_history'][-10:]) if self.frame_stats['contrast_history'] else 0,
                'avg_quality': np.mean(self.frame_stats['quality_history'][-10:]) if self.frame_stats['quality_history'] else 0
            }
        }
    
    def release(self):
        """Kaynakları serbest bırak"""
        self.running = False
        
        if self.adaptive_thread and self.adaptive_thread.is_alive():
            self.adaptive_thread.join(timeout=2)
        
        if self.resource_thread and self.resource_thread.is_alive():
            self.resource_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        self.logger.info("Optimize edilmiş adaptif kamera sistemi kapatıldı") 

    def set_performance_mode(self, mode: PerformanceMode):
        """Performans modunu ayarla"""
        self.current_settings.performance_mode = mode
        
        if mode == PerformanceMode.POWER_SAVE:
            # Güç tasarrufu modu
            self._apply_quality_preset(QualityPreset.LOW)
            self.current_settings.enable_post_processing = False
            self.auto_quality_adjust = False
            
        elif mode == PerformanceMode.BALANCED:
            # Dengeli mod
            self._apply_quality_preset(QualityPreset.MEDIUM)
            self.current_settings.enable_post_processing = True
            self.auto_quality_adjust = False
            
        elif mode == PerformanceMode.PERFORMANCE:
            # Performans modu
            self._apply_quality_preset(QualityPreset.HIGH)
            self.current_settings.enable_post_processing = True
            self.auto_quality_adjust = False
            
        elif mode == PerformanceMode.ADAPTIVE:
            # Adaptif mod - otomatik ayarlama
            self.auto_quality_adjust = True
        
        self._apply_settings(self.current_settings)
        self.logger.info(f"Performans modu değiştirildi: {mode.value}")
    
    def get_performance_metrics(self) -> Dict[str, any]:
        """Performans metrikleri"""
        avg_processing_time = 0
        if self.frame_stats['processing_time_history']:
            avg_processing_time = np.mean(self.frame_stats['processing_time_history'][-10:])
        
        return {
            'actual_fps': self.actual_fps,
            'target_fps': self.target_fps,
            'fps_efficiency': (self.actual_fps / self.target_fps * 100) if self.target_fps > 0 else 0,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'current_quality': self.current_settings.quality_preset.value,
            'performance_mode': self.current_settings.performance_mode.value,
            'system_resources': {
                'cpu_percent': self.system_resources.cpu_percent,
                'memory_percent': self.system_resources.memory_percent,
                'gpu_memory_percent': self.system_resources.gpu_memory_percent
            },
            'camera_settings': {
                'resolution': f"{self.current_settings.resolution[0]}x{self.current_settings.resolution[1]}",
                'fps': self.current_settings.fps,
                'post_processing': self.current_settings.enable_post_processing
            }
        } 