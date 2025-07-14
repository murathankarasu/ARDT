# ARDS - Gelişmiş Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### 1. Sistem Gereksinimleri

#### Minimum Gereksinimler
- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600X
- **RAM**: 8GB DDR4
- **GPU**: NVIDIA GTX 1050 Ti (4GB) / AMD RX 570
- **Depolama**: 10GB boş alan
- **Python**: 3.8+

#### Önerilen Gereksinimler
- **CPU**: Intel i7-10700K / AMD Ryzen 7 3700X
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA RTX 3060 (8GB) / AMD RX 6600 XT
- **Depolama**: 20GB SSD
- **Python**: 3.10+

#### Production Ortamı
- **CPU**: Intel i9-12900K / AMD Ryzen 9 5900X
- **RAM**: 32GB+ DDR4
- **GPU**: NVIDIA RTX 4070+ (12GB+ VRAM)
- **Depolama**: 50GB+ NVMe SSD

### 2. Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/username/ARDS.git
cd ARDS

# Hızlı kurulum scripti
./scripts/quick_start.sh

# Manuel kurulum
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# AI modellerini indirin
python scripts/install_models.py
```

## 🎯 Kullanım Modları

### Temel Kullanım

```bash
# Karma mod (sokak + mağaza)
python src/main.py --mode mixed

# Sadece sokak analizi
python src/main.py --mode street

# Sadece mağaza analizi
python src/main.py --mode store
```

### Gelişmiş Özellikler

```bash
# Firebase Analytics ile
python src/main.py --mode mixed --firebase

# GPU zorla kullan
python src/main.py --mode mixed --gpu

# Detaylı log
python src/main.py --mode mixed --verbose

# Özel konfigürasyon
python src/main.py --config configs/custom.yaml
```

### Kamera Kontrolü

```bash
# Farklı kamera kullan
python src/main.py --camera 1

# Çıktı klasörü belirle
python src/main.py --output results/session1/
```

## 🎮 Interaktif Kontroller

### Çalışma Sırasında Tuşlar

- **q**: Uygulamayı kapat
- **s**: Screenshot al
- **i**: Anlık bilgi göster
- **ESC**: Çıkış

## 🔧 Konfigürasyon

### Ana Konfigürasyon (`configs/default.yaml`)

```yaml
# Kamera ayarları
camera:
  adaptive_mode: true        # Otomatik optimizasyon
  quality_threshold: 0.7     # Kalite eşiği
  
# AI Sınıflandırıcı
ai_classifier:
  device: "auto"             # auto/cuda/cpu
  clip_model: "openai/clip-vit-base-patch32"
  
# Performans
performance:
  use_gpu: true
  max_fps: 30
  worker_threads: 4
```

### Firebase Konfigürasyonu

```yaml
firebase:
  enabled: true
  credentials_path: "path/to/service-key.json"
  project_id: "your-project-id"
```

## 📊 Firebase Analytics Kurulumu

### 1. Firebase Projesi Oluşturma

1. [Firebase Console](https://console.firebase.google.com) açın
2. "Create a project" tıklayın
3. Proje adını girin
4. Firestore Database oluşturun

### 2. Servis Hesabı Anahtarı

1. Project Settings > Service Accounts
2. "Generate new private key" tıklayın
3. JSON dosyasını güvenli yere kaydedin
4. Konfigürasyonda dosya yolunu belirtin

### 3. Firestore Kuralları

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /ards_sessions/{sessionId} {
      allow read, write: if true;
      match /{document=**} {
        allow read, write: if true;
      }
    }
  }
}
```

## 🤖 AI Modelleri

### Desteklenen Modeller

#### YOLO Modelleri
- **yolov8n.pt**: Hızlı tespit (nano)
- **yolov8s.pt**: Dengeli performans (small)
- **yolov8m.pt**: Yüksek doğruluk (medium)

#### HuggingFace Modelleri
- **CLIP**: Görüntü-metin anlayışı
- **EfficientNet**: Görüntü sınıflandırma
- **ResNet**: Genel nesne tanıma

### Model İndirme

```bash
# Tüm modelleri indir
python scripts/install_models.py

# Sadece YOLO modelleri
python -c "from scripts.install_models import download_yolo_models; download_yolo_models()"
```

## 📈 Performans Optimizasyonu

### GPU Optimizasyonu

```python
# GPU bellek kullanımını sınırla
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### CPU Optimizasyonu

```bash
# Thread sayısını artır
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Kamera Optimizasyonu

```yaml
camera:
  resolution:
    width: 1280    # Düşük çözünürlük = Hızlı
    height: 720
  fps: 30          # Düşük FPS = Az CPU
```

## 🏪 Uygulama Senaryoları

### Retail Mağaza Analizi

```bash
# Mağaza modunda çalıştır
python src/main.py --mode store --firebase

# Özel mağaza konfigürasyonu
python src/main.py --config configs/retail.yaml
```

### Trafik/Sokak Analizi

```bash
# Sokak modunda çalıştır
python src/main.py --mode street --output traffic_data/

# Yüksek çözünürlük
python src/main.py --mode street --config configs/high_res.yaml
```

### Güvenlik Uygulaması

```bash
# Sürekli kayıt
python src/main.py --mode mixed --firebase --output security/
```

## 📊 Veri Analizi

### Firestore Verileri

```python
# Oturum verilerini al
import firebase_admin
from firebase_admin import firestore

db = firestore.client()
sessions = db.collection('ards_sessions').get()

for session in sessions:
    data = session.to_dict()
    print(f"Session: {data['session_id']}")
    print(f"Duration: {data['active_duration']}s")
    print(f"Detections: {data['total_detections']}")
```

### CSV Export

```python
# Analytics verilerini CSV'ye aktar
from analytics.data_exporter import export_to_csv

export_to_csv('session_123', 'output/analytics.csv')
```

## 🔍 Sorun Giderme

### Yaygın Hatalar

#### "CUDA out of memory"
```bash
# GPU bellek kullanımını azalt
python src/main.py --config configs/low_memory.yaml
```

#### "Camera not found"
```bash
# Mevcut kameraları listele
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

#### "Model not found"
```bash
# Modelleri yeniden indir
python scripts/install_models.py
```

### Debug Modu

```bash
# Detaylı log
python src/main.py --verbose

# Tek frame işle
python src/main.py --debug --single-frame
```

### Performans İzleme

```bash
# Sistem kaynaklarını izle
htop

# GPU kullanımını izle
nvidia-smi -l 1
```

## 🌐 API Kullanımı

### REST API

```python
# Flask API başlat
python src/api/rest_server.py

# Endpoint'ler
POST /api/analyze     # Görüntü analizi
GET  /api/status      # Sistem durumu
GET  /api/metrics     # Performans metrikleri
```

### WebSocket

```python
# Real-time veri akışı
python src/api/websocket_server.py

# Client bağlantısı
ws://localhost:8765/stream
```

## 📚 Daha Fazla Bilgi

- [API Dokümantasyonu](API.md)
- [Model Eğitimi](TRAINING.md)
- [Deployment](DEPLOYMENT.md)
- [Katkıda Bulunma](CONTRIBUTING.md)

## 🆘 Destek

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@ards.com 