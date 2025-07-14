"""
ARDS - Ana uygulama dosyası (Düşük Donanım Optimize)
"""

import cv2
import argparse
import logging
import psutil
import torch
import time
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(__file__))

from core.adaptive_camera import AdaptiveCamera, PerformanceMode
from detection.object_detector import ObjectDetector
from detection.ai_classifier import AIClassifier, ModelComplexity
from attention.gaze_tracker import GazeTracker
from attention.attention_analyzer import AttentionAnalyzer
from analytics.firestore_manager import FirestoreAnalytics
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger, get_log_filename
from utils.system_monitor import SystemMonitor, PerformanceLevel


def main():
    """Ana uygulama fonksiyonu - optimize edilmiş"""
    
    # Argument parser
    parser = argparse.ArgumentParser(description="ARDS - AI Tabanlı Mikro Düzey Yer Tanıma Sistemi (Optimize)")
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
    parser.add_argument("--gpu", action="store_true", 
                        help="GPU kullanımını zorla")
    parser.add_argument("--performance", type=str, 
                        choices=["power_save", "balanced", "performance", "adaptive"],
                        default="adaptive", help="Performans modu")
    parser.add_argument("--verbose", action="store_true", 
                        help="Detaylı log")
    args = parser.parse_args()
    
    # Logger kurulumu
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = get_log_filename("ards") if Path("logs").exists() else None
    logger = setup_logger("ARDS", level=log_level, log_file=log_file)
    
    # Sistem bilgileri
    logger.info("="*60)
    logger.info("🚀 ARDS - AI Tabanlı Mikro Düzey Yer Tanıma Sistemi (Optimize)")
    logger.info("="*60)
    
    # Sistem monitoring başlat
    system_monitor = SystemMonitor(monitoring_interval=1.0)
    system_monitor.start_monitoring()
    
    # Donanım kontrolü
    show_system_info(logger, system_monitor)
    
    try:
        # Konfigürasyon yükleme
        config = ConfigLoader(args.config)
        logger.info(f"Konfigürasyon yüklendi: {args.config}")
        
        # Çıktı klasörünü oluştur
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Analytics sistemi (isteğe bağlı)
        analytics = None
        if args.firebase:
            try:
                firebase_config = config.get('firebase', {})
                analytics = FirestoreAnalytics(firebase_config)
                logger.info("📊 Firestore Analytics etkinleştirildi")
            except Exception as e:
                logger.warning(f"Analytics başlatılamadı: {e}")
        
        # Bileşenleri başlat
        logger.info("🔧 Sistem bileşenleri başlatılıyor...")
        
        # Adaptif Kamera
        camera_manager = AdaptiveCamera(
            camera_id=args.camera, 
            config=config.camera_config
        )
        
        # Performans modunu ayarla
        performance_mode = getattr(PerformanceMode, args.performance.upper())
        camera_manager.set_performance_mode(performance_mode)
        
        # AI Sınıflandırıcı
        ai_classifier = AIClassifier(config.get('ai_classifier', {}))
        ai_classifier.optimize_for_performance()
        
        # Nesne Tespit Sistemi
        object_detector = ObjectDetector(config.detection_config)
        
        # Bakış Takip Sistemi (adaptif)
        gaze_tracker = None
        if system_monitor.get_current_metrics().memory_percent < 70:
            gaze_tracker = GazeTracker(config.attention_config)
            logger.info("👁️ Gaze tracking etkinleştirildi")
        else:
            logger.warning("⚠️ Gaze tracking devre dışı - düşük memory")
        
        # Dikkat Analizi
        attention_analyzer = None
        if gaze_tracker:
            attention_analyzer = AttentionAnalyzer(config.get('attention_analysis', {}))
        
        # Performans callback'leri
        setup_performance_callbacks(system_monitor, camera_manager, ai_classifier, logger)
        
        logger.info(f"✅ Sistem hazır - Mod: {args.mode}, Performans: {args.performance}")
        logger.info("📹 Kamera başlatıldı - Çıkış için 'q' tuşuna basın")
        logger.info("🔧 Kontroller: 's'=screenshot, 'i'=info, 'p'=performance toggle")
        
        # Performans izleme
        frame_count = 0
        total_detections = 0
        skip_frame_count = 0
        last_fps_check = time.time()
        target_fps = camera_manager.target_fps
        
        # Adaptive processing parametreleri
        processing_intensity = 1.0  # 0.1 - 1.0 arası
        frame_skip_threshold = 0.7  # FPS efficiency threshold
        
        # Ana işleme döngüsü
        while True:
            start_time = time.time()
            
            # Kameradan görüntü al
            frame = camera_manager.get_frame()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Kamera durumu
            camera_status = camera_manager.get_camera_status()
            current_scene = camera_status['current_scene']
            actual_fps = camera_status['actual_fps']
            
            # Adaptive processing - performansa göre işlem seviyesini ayarla
            should_process_frame = decide_frame_processing(
                frame_count, processing_intensity, actual_fps, target_fps
            )
            
            if not should_process_frame:
                skip_frame_count += 1
                # Basit görselleştirme göster
                simple_frame = add_simple_overlay(frame, camera_status, 
                                                f"SKIP - Frame {frame_count}")
                cv2.imshow("ARDS - Optimize Mod", simple_frame)
                
                # Çıkış kontrolü
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # Nesne tespiti (adaptif seviye ve gaze durumuna göre)
            detections = detect_objects_adaptive(
                object_detector, frame, args.mode, processing_intensity, 
                gaze_available=(gaze_tracker is not None)
            )
            
            # AI tabanlı sınıflandırma (performansa göre)
            enhanced_detections = []
            if processing_intensity > 0.5:  # Sadece yüksek processing'de AI kullan
                for detection in detections:
                    # AI ile daha detaylı sınıflandırma
                    ai_result = ai_classifier.classify_region(
                        frame, detection['bbox'], context=current_scene
                    )
                    
                    # Sonuçları birleştir - güvenli key erişimi
                    enhanced_detection = detection.copy()
                    enhanced_detection.update({
                        'ai_class': ai_result.get('class', 'unknown'),
                        'ai_confidence': ai_result.get('confidence', 0.0),
                        'ai_method': ai_result.get('method', 'unknown'),
                        'scene_context': current_scene
                    })
                    enhanced_detections.append(enhanced_detection)
                    
                    # Analytics'e kaydet
                    if analytics:
                        analytics.log_detection(enhanced_detection, camera_status['settings'])
            else:
                # Sadece temel tespiti kullan
                enhanced_detections = detections
            
            total_detections += len(enhanced_detections)
            
            # Dikkat analizi (sadece yüksek performansta)
            attention_analysis = None
            if gaze_tracker and processing_intensity > 0.7:
                attention_data = gaze_tracker.analyze(frame)
                if attention_data and attention_analyzer:
                    # Nesne-dikkat ilişkisi analizi
                    attention_analysis = attention_analyzer.analyze_attention_objects(
                        attention_data, enhanced_detections
                    )
                    
                    # Analytics'e kaydet
                    if analytics:
                        analytics.log_attention(attention_analysis)
            
            # Sonuçları görselleştir
            result_frame = visualize_optimized_results(
                frame, enhanced_detections, attention_analysis, camera_status,
                processing_intensity
            )
            
            # Performans bilgilerini göster
            if frame_count % 30 == 0:  # Her 30 frame'de bir güncelle
                show_enhanced_performance_overlay(result_frame, {
                    'fps': actual_fps,
                    'target_fps': target_fps,
                    'scene': current_scene,
                    'detections': len(enhanced_detections),
                    'total_detections': total_detections,
                    'frame_count': frame_count,
                    'skip_count': skip_frame_count,
                    'processing_intensity': processing_intensity,
                    'system_metrics': system_monitor.get_current_metrics(),
                    'ai_performance': ai_classifier.get_performance_stats(),
                    'camera_performance': camera_manager.get_performance_metrics(),
                    'analytics': analytics.get_session_summary() if analytics else None
                })
            
            # Görüntüyü göster
            cv2.imshow("ARDS - Optimize Mod", result_frame)
            
            # Çıkış kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Screenshot
                screenshot_path = output_path / f"screenshot_{frame_count}.jpg"
                cv2.imwrite(str(screenshot_path), result_frame)
                logger.info(f"📸 Screenshot kaydedildi: {screenshot_path}")
            elif key == ord('i'):  # Info
                show_detailed_info(logger, frame_count, enhanced_detections, 
                                 current_scene, system_monitor, ai_classifier, camera_manager)
            elif key == ord('p'):  # Performance toggle
                processing_intensity = toggle_processing_intensity(processing_intensity)
                logger.info(f"🔧 Processing intensity: {processing_intensity:.1f}")
            
            # FPS ve processing intensity kontrolü
            processing_intensity = adjust_processing_intensity(
                actual_fps, target_fps, processing_intensity, 
                system_monitor.get_current_metrics()
            )
            
            # Frame işlem süresini logla
            frame_time = time.time() - start_time
            if frame_time > 1.0 / target_fps * 1.5:  # Frame süresi çok uzunsa
                logger.debug(f"⚠️ Yavaş frame: {frame_time:.3f}s")
                
    except KeyboardInterrupt:
        logger.info("🛑 Kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"❌ Kritik hata: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Temizlik
        logger.info("🧹 Sistem kapatılıyor...")
        
        # System monitoring durdur
        system_monitor.stop_monitoring()
        
        # Performans raporunu göster
        show_final_performance_report(logger, system_monitor, ai_classifier, camera_manager)
        
        if 'camera_manager' in locals():
            camera_manager.release()
        
        if 'analytics' in locals() and analytics:
            analytics.close()
        
        cv2.destroyAllWindows()
        logger.info("✅ ARDS sistemi kapatıldı")


def setup_performance_callbacks(system_monitor: SystemMonitor, camera_manager, ai_classifier, logger):
    """Performans callback'lerini ayarla"""
    
    def on_critical_performance(metrics, level):
        logger.warning(f"🔥 Kritik performans seviyesi: CPU %{metrics.cpu_percent:.1f}, RAM %{metrics.memory_percent:.1f}")
        # Acil optimizasyonlar
        camera_manager.set_performance_mode(PerformanceMode.POWER_SAVE)
    
    def on_high_performance(metrics, level):
        logger.warning(f"⚠️ Yüksek kaynak kullanımı: CPU %{metrics.cpu_percent:.1f}")
        # Orta seviye optimizasyonlar
        camera_manager.set_performance_mode(PerformanceMode.BALANCED)
    
    def on_low_performance(metrics, level):
        # Performans artırma fırsatı
        camera_manager.set_performance_mode(PerformanceMode.ADAPTIVE)
    
    system_monitor.add_performance_callback(PerformanceLevel.CRITICAL, on_critical_performance)
    system_monitor.add_performance_callback(PerformanceLevel.HIGH, on_high_performance)
    system_monitor.add_performance_callback(PerformanceLevel.LOW, on_low_performance)


def decide_frame_processing(frame_count: int, intensity: float, actual_fps: float, target_fps: float) -> bool:
    """Frame işlenip işlenmeyeceğine karar ver"""
    # FPS efficiency kontrolü
    fps_efficiency = actual_fps / target_fps if target_fps > 0 else 1.0
    
    # Düşük performansta frame atlama
    if fps_efficiency < 0.7 and intensity < 0.8:
        # Her 2. frame'i atla
        return frame_count % 2 == 0
    elif fps_efficiency < 0.5:
        # Her 3. frame'i atla
        return frame_count % 3 == 0
    
    return True


def detect_objects_adaptive(detector, frame, mode: str, intensity: float, gaze_available: bool = True):
    """Adaptif nesne tespiti - gaze durumuna göre"""
    # Log mesajı (sporadik)
    import random
    if random.randint(1, 90) == 1:  # Her 90 frame'de bir log
        if not gaze_available:
            logging.getLogger(__name__).info("🔍 Genel tespit modu: Tüm görünür nesneler tespit ediliyor (gaze yok)")
        else:
            logging.getLogger(__name__).info(f"🔍 Özelleşmiş tespit modu: {mode.upper()} (gaze aktif)")
    
    # Intensity'e göre farklı stratejiler
    if intensity < 0.3:
        # Çok hafif - sadece temel tespit
        detector.confidence_threshold = 0.7
        return detector.detect(frame, mode, gaze_available=gaze_available)
    elif intensity < 0.7:
        # Orta seviye
        detector.confidence_threshold = 0.6
        return detector.detect(frame, mode, gaze_available=gaze_available)
    else:
        # Tam performans
        detector.confidence_threshold = 0.4
        return detector.detect(frame, mode, gaze_available=gaze_available)


def adjust_processing_intensity(actual_fps: float, target_fps: float, 
                              current_intensity: float, system_metrics) -> float:
    """Processing intensity'yi dinamik ayarla"""
    # FPS efficiency
    fps_efficiency = actual_fps / target_fps if target_fps > 0 else 1.0
    
    # Sistem kaynaklarının durumu
    cpu_stress = system_metrics.cpu_percent > 80
    memory_stress = system_metrics.memory_percent > 85
    
    new_intensity = current_intensity
    
    # FPS düşükse intensity azalt
    if fps_efficiency < 0.7:
        new_intensity = max(0.1, current_intensity - 0.1)
    elif fps_efficiency > 0.9 and not cpu_stress and not memory_stress:
        # FPS iyi ve sistem rahatsa intensity artır
        new_intensity = min(1.0, current_intensity + 0.05)
    
    # Sistem stres altındaysa hızlı düşüş
    if cpu_stress or memory_stress:
        new_intensity = max(0.1, new_intensity - 0.2)
    
    return new_intensity


def toggle_processing_intensity(current: float) -> float:
    """Processing intensity'yi manuel toggle et"""
    if current <= 0.3:
        return 0.7
    elif current <= 0.7:
        return 1.0
    else:
        return 0.3


def show_system_info(logger, system_monitor):
    """Sistem bilgilerini göster"""
    # CPU bilgisi
    cpu_count = psutil.cpu_count()
    cpu_usage = psutil.cpu_percent()
    logger.info(f"💻 CPU: {cpu_count} çekirdek, %{cpu_usage} kullanım")
    
    # RAM bilgisi
    memory = psutil.virtual_memory()
    logger.info(f"🧠 RAM: {memory.total // (1024**3)}GB toplam, %{memory.percent} kullanım")
    
    # GPU bilgisi
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory}GB VRAM)")
    else:
        logger.warning("⚠️ GPU bulunamadı - CPU modunda çalışıyor")
    
    # Sistem performansı
    current_metrics = system_monitor.get_current_metrics()
    logger.info(f"💾 Sistem: CPU %{current_metrics.cpu_percent:.1f}, RAM %{current_metrics.memory_percent:.1f}")


def visualize_optimized_results(frame, detections, attention_analysis, camera_status, processing_intensity):
    """Gelişmiş sonuçları görselleştir"""
    result_frame = frame.copy()
    
    # Genel mod kontrolü
    general_mode = any(det.get('mode') == 'general' for det in detections)
    
    # Tespit türüne göre renk paleti - genel mod için genişletilmiş
    color_map = {
        # Geleneksel kategoriler
        'traffic_sign': (0, 255, 0),      # Yeşil
        'billboard': (255, 0, 0),         # Kırmızı
        'street_sign': (0, 255, 255),     # Sarı
        'direction_sign': (255, 255, 0),  # Cyan
        'shelf': (128, 0, 128),           # Mor
        'product_category': (255, 128, 0), # Turuncu
        'price_tag': (0, 128, 255),       # Açık mavi
        'shopping_cart': (128, 255, 0),   # Açık yeşil
        
        # Genel nesne kategorileri
        'furniture': (255, 100, 200),     # Pembe
        'plants': (50, 200, 50),          # Koyu yeşil
        'electronics': (100, 100, 255),   # Mavi
        'kitchen': (255, 200, 100),       # Sarımsı
        'living': (200, 100, 255),        # Mor
        'outdoor': (100, 255, 255),       # Açık mavi
        'animals': (255, 150, 100),       # Turuncu
        'accessories': (200, 200, 200),   # Gri
        'other': (255, 255, 255)          # Beyaz
    }
    
    # Nesne tespitlerini çiz
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Ana tespit bilgileri
        main_label = detection.get('label', 'unknown')
        main_confidence = detection.get('confidence', 0.0)
        category = detection.get('category', 'other')
        mode = detection.get('mode', 'mixed')
        
        # AI sınıflandırma
        ai_class = detection.get('ai_class', '')
        ai_confidence = detection.get('ai_confidence', 0.0)
        
        # Renk seç - önce kategori, sonra tip
        color = color_map.get(category, color_map.get(detection.get('type', 'unknown'), (255, 255, 255)))
        
        # Ana kutu - genel modda daha kalın çerçeve
        thickness = 3 if general_mode else 2
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Etiket metni - genel mod için daha detaylı
        if general_mode:
            label_lines = [
                f"{main_label}: {main_confidence:.2f}",
                f"Kategori: {category}"
            ]
            
            # Ek bilgiler (genel modda)
            if 'size_category' in detection:
                label_lines.append(f"Boyut: {detection['size_category']}")
            if 'screen_region' in detection:
                label_lines.append(f"Konum: {detection['screen_region']}")
            
            if ai_class and ai_confidence > 0:
                label_lines.append(f"AI: {ai_class} ({ai_confidence:.2f})")
        else:
            # Normal mod
            label_lines = [
                f"{main_label}: {main_confidence:.2f}"
            ]
            if ai_class and ai_confidence > 0:
                label_lines.append(f"AI: {ai_class} ({ai_confidence:.2f})")
        
        # Etiket arka planı
        label_height = len(label_lines) * 18 + 10
        label_width = max(200, len(max(label_lines, key=len)) * 8)
        
        cv2.rectangle(result_frame, (x1, y1 - label_height), 
                     (x1 + label_width, y1), color, -1)
        
        # Etiket metinleri
        for i, line in enumerate(label_lines):
            cv2.putText(result_frame, line, (x1 + 5, y1 - label_height + 15 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Dikkat verilerini çiz
    if attention_analysis:
        gaze_point = attention_analysis.get('gaze_point')
        if gaze_point:
            # Bakış noktası
            cv2.circle(result_frame, gaze_point, 15, (0, 0, 255), 3)
            cv2.circle(result_frame, gaze_point, 5, (255, 255, 255), -1)
            
            # İlgi seviyesi göstergesi
            interest_level = attention_analysis.get('interest_level', 0.0)
            radius = int(50 + interest_level * 100)
            cv2.circle(result_frame, gaze_point, radius, (0, 0, 255), 1)
            
            # Dikkat dağılımı metni
            attention_text = f"İlgi: %{interest_level*100:.0f}"
            cv2.putText(result_frame, attention_text, 
                       (gaze_point[0] + 20, gaze_point[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Kamera durumu
    scene_type = camera_status.get('current_scene', 'unknown')
    fps = camera_status.get('actual_fps', 0)
    
    # Durum çubuğu - mod bilgisi ekle
    mode_text = "GENEL MOD" if general_mode else "ÖZELLEŞMIŞ MOD"
    mode_color = (100, 255, 100) if general_mode else (255, 255, 100)
    
    status_text = f"🔍 {mode_text} | Sahne: {scene_type} | FPS: {fps:.1f}"
    cv2.putText(result_frame, status_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    
    # Genel mod için ek bilgi
    if general_mode:
        info_text = f"🎯 {len(detections)} nesne tespit edildi (tüm kategoriler)"
        cv2.putText(result_frame, info_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
    return result_frame


def show_enhanced_performance_overlay(frame, stats):
    """Performans bilgilerini overlay olarak göster"""
    h, w = frame.shape[:2]
    
    # Overlay alanı
    overlay_height = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 300, 0), (w, overlay_height), (0, 0, 0), -1)
    
    # Transparanlık
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Performans metrikler
    metrics = [
        f"FPS: {stats['fps']:.1f}",
        f"Sahne: {stats['scene']}",
        f"Tespit: {stats['detections']}",
        f"Toplam: {stats['total_detections']}",
        f"Frame: {stats['frame_count']}"
    ]
    
    # Analytics bilgisi
    if stats['analytics']:
        analytics_info = stats['analytics']
        metrics.append(f"Oturum: {analytics_info['duration']:.0f}s")
    
    # Metrikleri yaz
    for i, metric in enumerate(metrics):
        cv2.putText(frame, metric, (w - 290, 20 + i * 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def show_detailed_info(logger, frame_count, enhanced_detections, 
                       current_scene, system_monitor, ai_classifier, camera_manager):
    """Detaylı bilgi göster"""
    logger.info(f"📊 Frame: {frame_count}, Tespit: {len(enhanced_detections)}, Sahne: {current_scene}")
    logger.info(f"💾 Sistem: CPU %{system_monitor.get_current_metrics().cpu_percent:.1f}, RAM %{system_monitor.get_current_metrics().memory_percent:.1f}")
    logger.info(f"🎮 Kamera: Target FPS: {camera_manager.target_fps:.1f}, Actual FPS: {camera_manager.get_performance_metrics()['actual_fps']:.1f}")
    logger.info(f"🧠 AI Sınıflandırıcı: Model: {ai_classifier.model_complexity}, Performans: {ai_classifier.get_performance_stats()}")


def show_final_performance_report(logger, system_monitor, ai_classifier, camera_manager):
    """Son performans raporunu göster"""
    logger.info("🎯 Performans Raporu:")
    logger.info(f"💾 Sistem: CPU %{system_monitor.get_current_metrics().cpu_percent:.1f}, RAM %{system_monitor.get_current_metrics().memory_percent:.1f}")
    logger.info(f"🎮 Kamera: Target FPS: {camera_manager.target_fps:.1f}, Actual FPS: {camera_manager.get_performance_metrics()['actual_fps']:.1f}")
    logger.info(f"🧠 AI Sınıflandırıcı: Model: {ai_classifier.model_complexity}, Performans: {ai_classifier.get_performance_stats()}")


def add_simple_overlay(frame, camera_status, message):
    """Basit bir overlay ekler"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.putText(overlay, message, (w // 2 - len(message) * 5, h // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return overlay


if __name__ == "__main__":
    main() 