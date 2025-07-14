#!/usr/bin/env python3
"""
ARDS için gerekli AI modellerini indiren optimize edilmiş script
Kaynak dostu sürüm - 50GB yerine ~1GB indirme
"""

import os
import sys
import requests
import urllib.request
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoProcessor
from timm import create_model
import argparse


def download_file(url: str, destination: str, description: str = ""):
    """Dosya indirme fonksiyonu"""
    print(f"📥 İndiriliyor: {description}")
    
    try:
        # Dosya boyutunu al
        response = requests.head(url)
        total_size = int(response.headers.get('content-length', 0))
        
        # Progress bar ile indir
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ İndirildi: {destination}")
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False


def download_huggingface_model(model_name: str, cache_dir: str):
    """HuggingFace modeli indir"""
    try:
        print(f"🤗 HuggingFace modeli indiriliyor: {model_name}")
        
        # Model ve processor'ı indir
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        print(f"✅ HuggingFace modeli indirildi: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace model hatası ({model_name}): {e}")
        return False


def download_timm_model(model_name: str, cache_dir: str):
    """TIMM modeli indir"""
    try:
        print(f"⏰ TIMM modeli indiriliyor: {model_name}")
        
        # Modeli oluştur ve cache'le
        model = create_model(model_name, pretrained=True)
        
        # Model ağırlıklarını kaydet
        model_path = Path(cache_dir) / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        
        print(f"✅ TIMM modeli indirildi: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ TIMM model hatası ({model_name}): {e}")
        return False


def check_system_requirements():
    """Sistem gereksinimlerini kontrol et"""
    print("🔍 Sistem gereksinimleri kontrol ediliyor...")
    
    # Python sürümü
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ gerekli")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU: {gpu_name}")
        else:
            print("⚠️ GPU bulunamadı - CPU modunda çalışacak")
            
    except ImportError:
        print("❌ PyTorch bulunamadı")
        return False
    
    # Disk alanı
    import shutil
    free_space = shutil.disk_usage(".").free / (1024**3)  # GB
    required_space = 2  # GB (azaltıldı)
    
    if free_space < required_space:
        print(f"❌ Yetersiz disk alanı: {free_space:.1f}GB (gerekli: {required_space}GB)")
        return False
    
    print(f"✅ Disk alanı: {free_space:.1f}GB")
    
    return True


def get_model_configs():
    """Model konfigürasyonlarını döndür"""
    return {
        "ultra": {
            "name": "Ultra Hafif (Düşük Sistem)",
            "description": "Çok düşük donanım için - sadece ~50MB",
            "yolo_models": ["yolov8n"],
            "huggingface_models": [],
            "timm_models": []
        },
        "minimal": {
            "name": "Minimal (Çok Hafif)",
            "description": "Sadece temel modeller - ~200MB",
            "yolo_models": ["yolov8n"],
            "huggingface_models": [],
            "timm_models": ["efficientnet_lite0"]
        },
        "light": {
            "name": "Hafif (Önerilen)",
            "description": "Dengeli performans - ~800MB",
            "yolo_models": ["yolov8n", "yolov8s"],
            "huggingface_models": ["openai/clip-vit-base-patch32"],
            "timm_models": ["efficientnet_b0"]
        },
        "medium": {
            "name": "Orta Model (3GB)",
            "description": "MEDIUM komplexity - kalite-performans dengesi ~3GB",
            "yolo_models": ["yolov8s"],
            "huggingface_models": ["openai/clip-vit-base-patch32"],
            "timm_models": ["efficientnet_b2", "resnet34"]
        },
        "standard": {
            "name": "Standart",
            "description": "Tam özellik seti - ~1.5GB",
            "yolo_models": ["yolov8n", "yolov8s", "yolov8m"],
            "huggingface_models": ["openai/clip-vit-base-patch32"],
            "timm_models": ["efficientnet_b2", "resnet50"]
        },
        "full": {
            "name": "Tam (Büyük Sistemler)",
            "description": "Tüm modeller dahil - ~3GB",
            "yolo_models": ["yolov8n", "yolov8s", "yolov8m"],
            "huggingface_models": ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"],
            "timm_models": ["efficientnet_b2", "resnet50", "vit_base_patch16_224"]
        }
    }


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="ARDS Model İndirici")
    parser.add_argument(
        "--mode", 
        choices=["ultra", "minimal", "light", "medium", "standard", "full"], 
        default="light",
        help="İndirme modu (varsayılan: light)"
    )
    parser.add_argument(
        "--list-modes", 
        action="store_true",
        help="Mevcut modları listele"
    )
    
    args = parser.parse_args()
    
    print("🚀 ARDS - Optimize Model İndirici")
    print("=" * 60)
    
    # Modları listele
    if args.list_modes:
        configs = get_model_configs()
        print("\n📋 Mevcut İndirme Modları:")
        print("-" * 40)
        for mode, config in configs.items():
            print(f"🔹 {mode}: {config['name']}")
            print(f"   {config['description']}")
        print("\nÖrnek kullanım:")
        print("python scripts/install_models.py --mode minimal")
        return 0
    
    # Sistem kontrolü
    if not check_system_requirements():
        print("❌ Sistem gereksinimleri karşılanmıyor")
        return 1
    
    # Model konfigürasyonu
    configs = get_model_configs()
    selected_config = configs[args.mode]
    
    print(f"\n🎯 Seçilen mod: {selected_config['name']}")
    print(f"📖 Açıklama: {selected_config['description']}")
    
    # Kullanıcı onayı
    response = input(f"\n{selected_config['name']} modunu indirmek istiyor musunuz? (y/N): ")
    if response.lower() not in ['y', 'yes', 'evet', 'e']:
        print("❌ İşlem iptal edildi")
        return 1
    
    # Model klasörlerini oluştur
    models_dir = Path("models")
    pretrained_dir = models_dir / "pretrained"
    custom_dir = models_dir / "custom"
    cache_dir = models_dir / "cache"
    
    for directory in [pretrained_dir, custom_dir, cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # YOLO model URL'leri (v8.3.0 güncel sürüm)
    yolo_urls = {
        "yolov8n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "yolov8s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "yolov8m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt"
    }
    
    success_count = 0
    total_models = (
        len(selected_config["yolo_models"]) + 
        len(selected_config["huggingface_models"]) + 
        len(selected_config["timm_models"])
    )
    
    # YOLO modellerini indir
    print("\n📦 YOLO Modelleri")
    print("-" * 30)
    for model_name in selected_config["yolo_models"]:
        destination = pretrained_dir / f"{model_name}.pt"
        
        if destination.exists():
            print(f"⚠️ Zaten mevcut: {model_name}")
            success_count += 1
            continue
        
        url = yolo_urls[model_name]
        if download_file(url, str(destination), f"{model_name} (~{get_model_size(model_name)})"):
            success_count += 1
    
    # HuggingFace modellerini indir
    if selected_config["huggingface_models"]:
        print("\n🤗 HuggingFace Modelleri")
        print("-" * 30)
        for model_name in selected_config["huggingface_models"]:
            if download_huggingface_model(model_name, str(cache_dir)):
                success_count += 1
    
    # TIMM modellerini indir
    print("\n⏰ TIMM Modelleri")
    print("-" * 30)
    for model_name in selected_config["timm_models"]:
        if download_timm_model(model_name, str(cache_dir)):
            success_count += 1
    
    # Sonuç
    print("\n" + "=" * 60)
    print(f"📊 Tamamlandı: {success_count}/{total_models} model")
    
    if success_count == total_models:
        print("🎉 Tüm modeller başarıyla indirildi!")
        print("\n🚀 Kullanım:")
        print("python src/main.py --mode mixed")
        
        # Konfigürasyon önerisi
        if args.mode == "minimal":
            print("\n💡 Minimal mod kullanıyorsunuz, mobile.yaml config dosyasını kullanın:")
            print("python src/main.py --config configs/mobile.yaml")
        elif args.mode == "light":
            print("\n💡 Hafif mod kullanıyorsunız, default ayarlar optimal çalışacak")
    else:
        print("⚠️ Bazı modeller indirilemedi")
        print("📡 İnternet bağlantınızı kontrol edin")
        print("💾 Disk alanınızı kontrol edin")
        return 1
    
    # Model dosya boyutları
    print(f"\n📁 Model klasörü boyutu:")
    total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
    print(f"Toplam: {total_size / (1024**3):.2f} GB")
    
    return 0


def get_model_size(model_name):
    """Model boyutlarını tahmin et"""
    sizes = {
        "yolov8n": "6MB",
        "yolov8s": "22MB", 
        "yolov8m": "52MB",
        "efficientnet_lite0": "20MB",
        "efficientnet_b0": "21MB",
        "efficientnet_b2": "36MB",
        "resnet50": "98MB"
    }
    return sizes.get(model_name, "bilinmiyor")


if __name__ == "__main__":
    sys.exit(main()) 