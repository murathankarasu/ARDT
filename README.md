# ARDS - AI-Powered Micro-Level Location Recognition & Attention Detection System

<div align="center">

![ARDS Logo](docs/images/logo.png)

**Gelişmiş AI teknolojileri ile mikro düzeyde yer tanımlama ve dikkat analizi**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Firebase](https://img.shields.io/badge/Firebase-Enabled-orange.svg)](https://firebase.google.com/)

</div>

## 🎯 Proje Hedefi

ARDS, kamera görüntülerinde mikro düzeyde yer tanımlama ve dikkat analizi yapan gelişmiş bir AI sistemidir. Sokak tabelalarından mağaza reyonlarına, reklam panolarından fiyat etiketlerine kadar geniş bir yelpazede nesneleri tespit eder ve insanların bakış yönlerini analiz ederek hangi nesnelere odaklandıklarını tahmin eder.

## ✨ Öne Çıkan Özellikler

### 🔍 Gelişmiş AI Sınıflandırıcı
- **CLIP Model**: Görüntü-metin anlayışı ile context-aware sınıflandırma
- **EfficientNet**: Yüksek doğrulukta görüntü sınıflandırma
- **ResNet**: Genel nesne tanıma
- **Ensemble Learning**: Çoklu model sonuçlarını birleştirme

### 📹 Adaptif Kamera Sistemi
- **Otomatik Sahne Tespiti**: Gece/gündüz, sokak/mağaza otomatik tanıma
- **Dinamik Optimizasyon**: Sahne türüne göre kamera ayarlarını otomatik optimize etme
- **Gerçek Zamanlı Kalite Analizi**: Frame kalitesini sürekli izleme
- **Otonom Araç Mantığı**: Tesla tarzı adaptif kamera optimizasyonu

### 👁️ Gelişmiş Dikkat Analizi
- **Göz Takibi**: MediaPipe ile hassas bakış noktası tespiti
- **Nesne-Dikkat İlişkisi**: Hangi nesnelere bakıldığını analiz etme
- **İlgi Seviyesi**: Dikkat yoğunluğunu ölçme
- **Temporal Analysis**: Zaman içindeki dikkat dağılımı

### 📊 Firebase Analytics
- **Gerçek Zamanlı Veri**: Tüm tespitler ve dikkat verileri Firestore'a kayıt
- **Batch Processing**: Verimli veri işleme
- **Session Management**: Oturum bazlı analitik
- **Comprehensive Metrics**: Detaylı performans metrikleri

## 🚀 Hızlı Başlangıç

### Otomatik Kurulum (Önerilen)

```bash
git clone https://github.com/username/ARDS.git
cd ARDS
./scripts/quick_start.sh
```

### Manuel Kurulum

```bash
# 1. Repository klonla
git clone https://github.com/username/ARDS.git
cd ARDS

# 2. Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. AI modellerini indir
python scripts/install_models.py

# 5. Çalıştır!
python src/main.py --mode mixed
```

## 💻 Sistem Gereksinimleri

### Minimum
- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600X
- **RAM**: 8GB DDR4
- **GPU**: NVIDIA GTX 1050 Ti (4GB) / AMD RX 570
- **Python**: 3.8+

### Önerilen
- **CPU**: Intel i7-10700K / AMD Ryzen 7 3700X
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA RTX 3060 (8GB) / AMD RX 6600 XT
- **Python**: 3.10+

### Production
- **CPU**: Intel i9-12900K / AMD Ryzen 9 5900X
- **RAM**: 32GB+ DDR4
- **GPU**: NVIDIA RTX 4070+ (12GB+ VRAM)

## 🎮 Kullanım Örnekleri

### Temel Kullanım

```bash
# Karma mod - sokak ve mağaza analizi (otomatik optimizasyon)
python src/main.py --mode mixed

# Sadece sokak analizi
python src/main.py --mode street

# Sadece mağaza analizi  
python src/main.py --mode store

# Firebase analytics ile
python src/main.py --mode mixed --firebase

# GPU zorla kullan
python src/main.py --mode mixed --gpu

# Detaylı log
python src/main.py --mode mixed --verbose
```

### 🔧 Performans Optimizasyonu (YENİ!)

```bash
# Otomatik performans ayarlama (varsayılan)
python src/main.py --performance adaptive

# Güç tasarrufu modu (düşük donanım)
python src/main.py --performance power_save

# Dengeli performans
python src/main.py --performance balanced

# Maksimum kalite (yüksek donanım)
python src/main.py --performance performance

# Mobil/Embedded cihazlar için
python src/main.py --config configs/mobile.yaml --performance power_save

# 4GB RAM altı sistemler için
python src/main.py --performance power_save --mode mixed
```

### 📱 Düşük Donanım Desteği

ARDS artık **2GB RAM** ve **2 CPU çekirdeği** olan cihazlarda bile çalışır!

**Otomatik Optimizasyon Özellikleri:**
- 🧠 Sistem kaynaklarına göre otomatik model seçimi
- 📊 Gerçek zamanlı performans izleme
- ⚡ Adaptif frame processing
- 🔋 Pil tasarrufu modu
- 📱 Mobil cihaz desteği

### Çalışma Sırasında Kontroller

- **q**: Uygulamayı kapat
- **s**: Screenshot al  
- **i**: Detaylı sistem bilgisi göster
- **p**: Performans modu değiştir (manuel toggle)

## 🏗️ Proje Yapısı

```
ARDS/
├── src/                      # Ana kaynak kod
│   ├── core/                # Temel sistem bileşenleri
│   │   ├── adaptive_camera.py    # Adaptif kamera sistemi
│   │   └── image_processor.py    # Görüntü işleme
│   ├── detection/           # Nesne tespit sistemi
│   │   ├── object_detector.py    # Ana tespit motoru
│   │   ├── ai_classifier.py     # AI sınıflandırıcı
│   │   ├── street_detector.py   # Sokak öğeleri
│   │   └── store_detector.py    # Mağaza öğeleri
│   ├── attention/           # Dikkat analizi
│   │   ├── gaze_tracker.py      # Bakış takibi
│   │   └── attention_analyzer.py # Dikkat analizi
│   ├── analytics/           # Veri analizi
│   │   └── firestore_manager.py # Firebase entegrasyonu
│   ├── utils/               # Yardımcı fonksiyonlar
│   └── main.py             # Ana uygulama
├── models/                  # AI modelleri
│   ├── pretrained/         # Önceden eğitilmiş modeller
│   ├── custom/             # Özel modeller
│   └── cache/              # Model cache
├── configs/                # Konfigürasyon dosyaları
├── scripts/                # Yardımcı script'ler
├── docs/                   # Dokumentasyon
└── requirements.txt        # Python bağımlılıkları
```

## 🔧 Konfigürasyon

### Ana Konfigürasyon (`configs/default.yaml`)

```yaml
# Adaptif kamera ayarları
camera:
  adaptive_mode: true
  scene_analysis_interval: 0.5
  quality_threshold: 0.7

# AI sınıflandırıcı
ai_classifier:
  device: "auto"  # auto/cuda/cpu
  clip_model: "openai/clip-vit-base-patch32"
  ensemble_weights:
    clip: 0.5
    efficientnet: 0.3
    resnet: 0.2

# Firebase analytics
firebase:
  enabled: false
  credentials_path: "path/to/service-account.json"
  project_id: "your-project-id"

# Performans optimizasyonu
performance:
  use_gpu: true
  max_fps: 30
  worker_threads: 4
```

## 📊 Firebase Analytics Kurulumu

1. **Firebase Console'da proje oluşturun**
2. **Firestore Database etkinleştirin**
3. **Service Account anahtarı oluşturun**
4. **Konfigürasyonu güncelleyin**

Detaylı kurulum için: [Firebase Setup Guide](docs/FIREBASE_SETUP.md)

## 🎯 Uygulama Alanları

### 🏪 Retail & E-commerce
- Müşteri davranış analizi
- Raf optimizasyonu
- Ürün yerleştirme analizi
- Dikkat haritaları

### 🚗 Trafik & Ulaşım
- Trafik levhası tespiti
- Yön tabelası analizi
- Sürücü dikkat analizi
- Güvenlik kameraları

### 🏢 Güvenlik & Gözetim
- Şüpheli davranış tespiti
- Dikkat analizi
- Nesne takibi
- Olay kaydı

### 📱 Mobil Uygulamalar
- Artırılmış gerçeklik
- Görsel arama
- Akıllı asistan
- Accessibility uygulamaları

## 🧠 AI Modelleri ve Teknolojiler

### Nesne Tespit
- **YOLOv8**: Hızlı ve doğru nesne tespiti
- **Custom Detectors**: Sokak ve mağaza özelleştirmeleri

### Görüntü Sınıflandırma
- **CLIP**: Görüntü-metin anlayışı
- **EfficientNet**: Yüksek doğruluk
- **ResNet**: Genel sınıflandırma

### Dikkat Analizi
- **MediaPipe**: Yüz ve göz landmark tespiti
- **Custom Algorithms**: Bakış yönü tahmini

## 📈 Performans Metrikleri

| Özellik | Değer |
|---------|-------|
| **FPS** | 30+ (GPU) / 15+ (CPU) |
| **Tespit Doğruluğu** | %92+ |
| **Dikkat Doğruluğu** | %85+ |
| **Bellek Kullanımı** | 2-8GB |
| **GPU Kullanımı** | 3-6GB VRAM |

## 📚 Dokümantasyon

- [Gelişmiş Kullanım Kılavuzu](docs/ADVANCED_USAGE.md)
- [API Dokümantasyonu](docs/API.md)
- [Model Eğitimi](docs/TRAINING.md)
- [Deployment Rehberi](docs/DEPLOYMENT.md)

## 🤝 Katkıda Bulunma

Katkılarınızı memnuniyetle karşılıyoruz! Lütfen şunları kontrol edin:

1. [Contributing Guidelines](CONTRIBUTING.md)
2. [Code of Conduct](CODE_OF_CONDUCT.md)
3. [Development Setup](docs/DEVELOPMENT.md)

## 📝 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## 🆘 Destek ve İletişim

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/username/ARDS/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/username/ARDS/discussions)
- **📧 Email**: support@ards.com
- **💬 Discord**: [ARDS Community](https://discord.gg/ards)

## 🙏 Teşekkürler

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://github.com/google/mediapipe)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!**

Made with ❤️ by ARDS Team

</div> 