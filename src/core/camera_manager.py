"""
Kamera yönetimi sınıfı
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List


def detect_available_cameras(max_test: int = 10) -> List[int]:
    """
    Mevcut kameraları tespit et
    
    Args:
        max_test: Test edilecek maksimum kamera sayısı
        
    Returns:
        Kullanılabilir kamera indekslerinin listesi
    """
    available_cameras = []
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                logging.info(f"Kamera {i} bulundu ve test edildi")
            cap.release()
        else:
            # macOS'ta farklı backend'ler deneyelim
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2, cv2.CAP_DSHOW]
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            available_cameras.append(i)
                            logging.info(f"Kamera {i} bulundu (backend: {backend})")
                            cap.release()
                            break
                    cap.release()
                except Exception:
                    continue
    
    return available_cameras


class CameraManager:
    """Kamera işlemlerini yöneten sınıf"""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720)):
        """
        Args:
            camera_id: Kamera indeksi
            resolution: Çözünürlük (width, height)
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.logger = logging.getLogger(__name__)
        
        # Önce mevcut kameraları tespit et
        available = detect_available_cameras()
        if not available:
            self.logger.error("Hiç kamera bulunamadı! Kamera izinlerini kontrol edin.")
            raise RuntimeError("Kamera bulunamadı")
        
        # Eğer istenen kamera mevcut değilse, ilk mevcut kamerayı kullan
        if camera_id not in available:
            self.logger.warning(f"Kamera {camera_id} bulunamadı. Mevcut kameralar: {available}")
            self.camera_id = available[0]
            self.logger.info(f"Kamera {self.camera_id} kullanılacak")
        
        self._initialize_camera()
    
    def _initialize_camera(self) -> None:
        """Kamerayı başlat"""
        try:
            # macOS için AVFoundation backend'i deneyelim
            import platform
            if platform.system() == "Darwin":  # macOS
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                # Farklı backend'ler deneyelim
                backends = [cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_GSTREAMER]
                for backend in backends:
                    try:
                        self.cap = cv2.VideoCapture(self.camera_id, backend)
                        if self.cap.isOpened():
                            break
                    except Exception:
                        continue
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Kamera {self.camera_id} açılamadı")
            
            # Çözünürlük ayarla
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # FPS ayarla
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Kamera test et
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Kamera {self.camera_id} frame üretemiyor")
            
            self.logger.info(f"Kamera {self.camera_id} başarıyla başlatıldı - Çözünürlük: {frame.shape}")
            
        except Exception as e:
            self.logger.error(f"Kamera başlatma hatası: {e}")
            if self.cap:
                self.cap.release()
            raise
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Kameradan bir frame al
        
        Returns:
            Frame verisi veya None
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            self.logger.warning("Frame alınamadı")
            return None
        
        return frame
    
    def get_camera_info(self) -> dict:
        """Kamera bilgilerini al"""
        if self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION)
        }
    
    def set_camera_property(self, prop: int, value: float) -> bool:
        """Kamera özelliği ayarla"""
        if self.cap is None:
            return False
        
        return self.cap.set(prop, value)
    
    def release(self) -> None:
        """Kamera kaynaklarını serbest bırak"""
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Kamera kaynakları serbest bırakıldı")
    
    def __del__(self):
        """Destructor"""
        self.release() 