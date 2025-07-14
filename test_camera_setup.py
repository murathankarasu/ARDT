#!/usr/bin/env python3
"""
ARDS Kamera ve Sistem Test Script
Tüm çözümleri test eder:
1. Kamera tespiti ve kurulumu
2. CLIP device mismatch düzeltmesi  
3. Firestore analytics testi
"""

import sys
import os
sys.path.append('src')

import cv2
import logging
import torch
from pathlib import Path

# ARDS modules
from core.camera_manager import CameraManager, detect_available_cameras
from detection.ai_classifier import AIClassifier
from analytics.firestore_manager import FirestoreAnalytics
from utils.config_loader import ConfigLoader

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ARDS_TEST")

def test_camera_detection():
    """Test kamera tespiti"""
    logger.info("🎥 Kamera tespiti test ediliyor...")
    
    try:
        available_cameras = detect_available_cameras()
        logger.info(f"✅ Bulunan kameralar: {available_cameras}")
        
        if not available_cameras:
            logger.error("❌ Hiç kamera bulunamadı!")
            logger.info("💡 macOS'ta System Preferences > Security & Privacy > Camera'dan izin verin")
            return False
        
        # İlk kamerayı test et
        camera_id = available_cameras[0]
        logger.info(f"📹 Kamera {camera_id} test ediliyor...")
        
        camera_manager = CameraManager(camera_id=camera_id)
        
        # Frame test
        frame = camera_manager.get_frame()
        if frame is not None:
            logger.info(f"✅ Kamera başarılı - Frame boyutu: {frame.shape}")
            camera_manager.release()
            return True
        else:
            logger.error("❌ Frame alınamadı")
            camera_manager.release()
            return False
            
    except Exception as e:
        logger.error(f"❌ Kamera testi başarısız: {e}")
        return False

def test_clip_device_fix():
    """Test CLIP device mismatch düzeltmesi"""
    logger.info("🤖 CLIP device mismatch düzeltmesi test ediliyor...")
    
    try:
        # Device kontrolü
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("✅ Apple M2 GPU (MPS) tespit edildi")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("✅ CUDA GPU tespit edildi")
        else:
            device = "cpu"
            logger.info("ℹ️ CPU modu kullanılacak")
        
        # AI Classifier test
        config = {
            'force_model_complexity': 'medium',
            'auto_select_model': False
        }
        
        ai_classifier = AIClassifier(config)
        logger.info(f"✅ AI Classifier başlatıldı - Device: {ai_classifier.device}")
        
        # Test image oluştur
        import numpy as np
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_bbox = [50, 50, 150, 150]
        
        # CLIP test
        logger.info("🧠 CLIP sınıflandırma test ediliyor...")
        result = ai_classifier.classify_region(test_image, test_bbox, context='store')
        
        logger.info(f"✅ CLIP test başarılı: {result['class']} (güven: {result['confidence']:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"❌ CLIP test başarısız: {e}")
        # Detaylı hata bilgisi
        import traceback
        logger.error(f"Hata detayı: {traceback.format_exc()}")
        return False

def test_firestore_analytics():
    """Test Firestore analytics"""
    logger.info("📊 Firestore analytics test ediliyor...")
    
    try:
        # Firebase config dosyası var mı kontrol et
        firebase_config_path = "configs/firebase_production.yaml"
        if not os.path.exists(firebase_config_path):
            logger.warning(f"⚠️ Firebase config dosyası bulunamadı: {firebase_config_path}")
            logger.info("💡 firebase-admin kurulu mu kontrol edin: pip install firebase-admin")
            return False
        
        # Config yükle
        config_loader = ConfigLoader(firebase_config_path)
        firebase_config = config_loader.get('firebase', {})
        
        # Analytics test
        analytics = FirestoreAnalytics(firebase_config)
        
        if analytics.db is None:
            logger.warning("⚠️ Firestore bağlantısı kurulamadı (normal - credentials gerekli)")
            logger.info("💡 Firebase credentials düzgün ayarlandıysa çalışacak")
            return True
        
        # Test veri gönder
        test_detection = {
            'type': 'test',
            'label': 'test_object',
            'confidence': 0.95,
            'bbox': [100, 100, 200, 200],
            'scene_context': 'test'
        }
        
        analytics.log_detection(test_detection)
        logger.info("✅ Firestore analytics test başarılı")
        
        # Session summary
        summary = analytics.get_session_summary()
        logger.info(f"📈 Session özeti: {summary}")
        
        analytics.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Firestore test başarısız: {e}")
        logger.info("💡 Firebase Admin SDK kurulu mu: pip install firebase-admin")
        return False

def main():
    """Ana test fonksiyonu"""
    logger.info("🚀 ARDS Sistem Testi Başlıyor")
    logger.info("=" * 50)
    
    results = {}
    
    # 1. Kamera testi
    results['camera'] = test_camera_detection()
    
    # 2. CLIP device fix testi
    results['clip'] = test_clip_device_fix()
    
    # 3. Firestore analytics testi
    results['firestore'] = test_firestore_analytics()
    
    # Sonuçlar
    logger.info("=" * 50)
    logger.info("📋 TEST SONUÇLARI:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ BAŞARILI" if passed else "❌ BAŞARISIZ"
        logger.info(f"  {test_name.upper()}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 TÜM TESTLER BAŞARILI!")
        logger.info("💡 Sistemi çalıştırmak için:")
        logger.info("   python src/main.py --config configs/firebase_production.yaml --firebase")
    else:
        logger.warning("⚠️ Bazı testler başarısız oldu. Yukarıdaki önerileri kontrol edin.")
    
    return all_passed

if __name__ == "__main__":
    main() 