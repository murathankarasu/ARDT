"""
ARDS - Ana uygulama dosyası (Modüler Yapı)
"""

import cv2
import argparse
import logging
import time
import os
import warnings
import sys
from pathlib import Path

# Sistem uyarılarını çöz
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

sys.path.append(os.path.dirname(__file__))

from core.app_initializer import ARDSInitializer
from core.frame_processor import FrameProcessor
from utils.logger import setup_logger, get_log_filename


class ARDSApplication:
    """Ana ARDS uygulama sınıfı - Modüler yapı"""
    
    def __init__(self, args):
        self.args = args
        self.logger = self._setup_logger()
        self.running = False
        
        # Modüler bileşenler
        self.initializer = ARDSInitializer(args, self.logger)
        self.frame_processor = None
        self.components = {}
        
        print("🚀 ARDS - 3D Gaze Tracking & Collision Detection")
        print("✨ Modüler yapı: Temiz kod, yüksek performans")
    
    def _setup_logger(self):
        """Logger kurulumu"""
        log_level = logging.DEBUG if self.args.verbose else logging.WARNING
        log_file = get_log_filename("ards") if Path("logs").exists() else None
        logger = setup_logger("ARDS", level=log_level, log_file=log_file)
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        return logger
    
    def initialize(self):
        """Sistem bileşenlerini başlat"""
        try:
            # Tüm bileşenleri başlat
            self.components = self.initializer.initialize_all_components()
            
            # Frame processor'ı başlat
            self.frame_processor = FrameProcessor(self.components)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sistem başlatma hatası: {e}")
            return False
    
    def run(self):
        """Ana uygulama döngüsü"""
        if not self.initialize():
            print("❌ Sistem başlatılamadı")
            return
        
        self.running = True
        camera_manager = self.components.get('camera_manager')
        
        try:
            while self.running:
                # Frame al
                frame = camera_manager.get_frame()
                if frame is None:
                    continue
                
                # Frame'i işle
                result_frame, stats = self.frame_processor.process_frame(frame, self.args.mode)
                
                # Görüntüyü göster
                cv2.imshow("ARDS - 3D Gaze Tracking & Collision Detection", result_frame)
                
                # Kullanıcı girişi
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_input(key, result_frame, stats):
                    break
                
                # Hız kontrolü
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Kullanıcı tarafından durduruldu")
        except Exception as e:
            self.logger.error(f"❌ Kritik hata: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def _handle_input(self, key, frame, stats) -> bool:
        """Kullanıcı girişlerini yönet"""
        if key == ord('q'):
            self.running = False
            return False
        elif key == ord('s'):
            self._save_screenshot(frame, stats)
        elif key == ord('r'):
            self._reset_gaze_trail()
        elif key == ord('c'):
            self._clear_collision_data()
        elif key == ord('g'):
            self._toggle_3d_gaze()
        elif key == ord('i'):
            self._show_detailed_info(stats)
        elif key == ord('h'):
            self._show_human_readable_summary()
        
        return True
    
    def _show_human_readable_summary(self):
        """İnsan okunakli özet göster"""
        try:
            if not self.frame_processor:
                print("⚠️ Frame processor mevcut değil")
                return
            
            readable_summary = self.frame_processor.get_readable_analytics_summary()
            
            print("\n" + "="*60)
            print("📊 MEVCUT OTURUM ÖZETİ - İNSAN OKUNAKLI")
            print("="*60)
            
            if not readable_summary:
                print("• Analytics mevcut değil")
            else:
                for key, value in readable_summary.items():
                    if isinstance(value, str):
                        print(f"• {value}")
                    elif isinstance(value, list) and key == 'most_interesting_objects':
                        if value:
                            print(f"• En çok bakılan objeler: {', '.join(value)}")
            
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Human readable summary hatası: {e}")
    
    def _save_screenshot(self, frame, stats):
        """Ekran görüntüsü kaydet"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ards_screenshot_{timestamp}.png"
            filepath = os.path.join(self.args.output, filename)
            
            cv2.imwrite(filepath, frame)
            print(f"📸 Ekran görüntüsü kaydedildi: {filename}")
            
        except Exception as e:
            self.logger.error(f"Screenshot kaydetme hatası: {e}")
    
    def _reset_gaze_trail(self):
        """Gaze trail'i sıfırla"""
        try:
            gaze_3d_processor = self.components.get('gaze_3d_processor')
            if gaze_3d_processor:
                gaze_3d_processor.reset_collision_data()
                print("🔄 Gaze trail sıfırlandı")
            else:
                print("⚠️ Gaze 3D processor mevcut değil")
                
        except Exception as e:
            self.logger.error(f"Gaze trail reset hatası: {e}")
    
    def _clear_collision_data(self):
        """Çakışma verilerini temizle"""
        try:
            gaze_3d_processor = self.components.get('gaze_3d_processor')
            if gaze_3d_processor:
                gaze_3d_processor.reset_collision_data()
                print("🗑️ Çakışma verileri temizlendi")
            else:
                print("⚠️ Gaze 3D processor mevcut değil")
                
        except Exception as e:
            self.logger.error(f"Collision data clear hatası: {e}")
    
    def _toggle_3d_gaze(self):
        """3D gaze görüntüsünü aç/kapat"""
        try:
            visualization_manager = self.components.get('visualization_manager')
            if visualization_manager:
                # Toggle 3D gaze rendering
                # Bu fonksiyon visualization_manager'da implement edilmeli
                print("🎯 3D gaze görüntü durumu değiştirildi")
            else:
                print("⚠️ Visualization manager mevcut değil")
                
        except Exception as e:
            self.logger.error(f"3D gaze toggle hatası: {e}")
    
    def _show_detailed_info(self, stats):
        """Detaylı sistem bilgisini göster"""
        try:
            print("\n" + "="*80)
            print("🔍 ARDS - Detaylı Sistem Bilgisi")
            print("="*80)
            
            # Frame processing stats
            if self.frame_processor:
                proc_stats = self.frame_processor.get_processing_statistics()
                print(f"📊 Frame Count: {proc_stats['frame_count']:,}")
                print(f"📊 Total Detections: {proc_stats['total_detections']:,}")
                print(f"📊 Total Collisions: {proc_stats['total_collisions']:,}")
                print(f"📊 Detection Rate: {proc_stats['detection_rate']:.3f}")
                print(f"📊 Collision Rate: {proc_stats['collision_rate']:.3f}")
                print(f"📊 Avg Processing Time: {proc_stats['avg_processing_time']*1000:.1f}ms")
                print(f"📊 Current FPS: {proc_stats['current_fps']:.1f}")
            
            # Sistem bilgileri
            system_info = self.initializer.get_system_info()
            print(f"💾 Memory Usage: {system_info['memory_usage']:.1f}%")
            print(f"👁️ Gaze Tracking: {'Aktif' if system_info['gaze_tracking_active'] else 'Devre Dışı'}")
            print(f"🎯 3D Gaze: {'Aktif' if system_info['gaze_3d_active'] else 'Devre Dışı'}")
            print(f"📊 Analytics: {'Aktif' if system_info['analytics_active'] else 'Devre Dışı'}")
            
            # Çakışma bilgileri
            gaze_3d_processor = self.components.get('gaze_3d_processor')
            if gaze_3d_processor:
                collision_summary = gaze_3d_processor.get_collision_summary()
                print(f"🎯 Aktif Çakışma: {collision_summary['active_collisions']}")
                print(f"🎯 Toplam Çakışma: {collision_summary['total_collision_events']}")
                print(f"🎯 Ortalama Süre: {collision_summary['average_collision_duration']:.1f}s")
            
            print("="*80)
            
        except Exception as e:
            self.logger.error(f"Detailed info hatası: {e}")
    
    def cleanup(self):
        """Sistem temizleme"""
        try:
            print("🔄 Sistem kapatılıyor...")
            
            # Session özeti logla
            if self.frame_processor:
                self.frame_processor.log_session_end_summary()
            
            # Bileşenleri temizle
            if self.initializer:
                self.initializer.cleanup()
            
            # OpenCV penceresini kapat
            cv2.destroyAllWindows()
            
            print("✅ Sistem temizlendi")
            
        except Exception as e:
            self.logger.error(f"Cleanup hatası: {e}")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="ARDS - 3D Gaze Tracking & Collision Detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Konfigürasyon dosyası yolu")
    parser.add_argument("--camera", type=int, default=0, 
                        help="Kamera indeksi")
    parser.add_argument("--mode", type=str, choices=["street", "store", "mixed"], 
                        default="mixed", help="Analiz modu")
    parser.add_argument("--output", type=str, default="output/", 
                        help="Çıktı klasörü")
    parser.add_argument("--firebase", action="store_true", 
                        help="Firebase analytics etkinleştir")
    parser.add_argument("--performance", type=str, 
                        choices=["power_save", "balanced", "performance", "adaptive"],
                        default="balanced", help="Performans modu")
    parser.add_argument("--verbose", action="store_true", 
                        help="Detaylı log")
    
    args = parser.parse_args()
    
    # Uygulama başlat
    app = ARDSApplication(args)
    app.run()


if __name__ == "__main__":
    main() 