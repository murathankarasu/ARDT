#!/bin/bash

# ARDS - Hızlı Başlangıç Scripti

echo "🚀 ARDS - AI Tabanlı Mikro Düzey Yer Tanıma Sistemi"
echo "=================================================="

# Python sürümünü kontrol et
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 bulunamadı. Lütfen Python 3.8+ yükleyin."
    exit 1
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python sürümü: $python_version"

# Virtual environment oluştur
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment oluşturuluyor..."
    python3 -m venv venv
fi

# Virtual environment'ı aktive et
echo "🔧 Virtual environment aktive ediliyor..."
source venv/bin/activate

# Bağımlılıkları yükle
echo "📚 Bağımlılıklar yükleniyor..."
pip install --upgrade pip
pip install -r requirements.txt

# Gerekli klasörleri oluştur
echo "📁 Klasörler oluşturuluyor..."
mkdir -p logs output models/pretrained models/custom

# Modelleri indir
echo "🤖 AI modelleri indiriliyor..."
python scripts/install_models.py

# Konfigürasyon kontrolü
if [ ! -f "configs/default.yaml" ]; then
    echo "❌ Konfigürasyon dosyası bulunamadı!"
    exit 1
fi

echo ""
echo "✅ ARDS başarıyla kuruldu!"
echo ""
echo "🎯 Kullanım örnekleri:"
echo "  # Varsayılan ayarlarla çalıştır"
echo "  python src/main.py"
echo ""
echo "  # Sadece sokak modunda çalıştır"
echo "  python src/main.py --mode street"
echo ""
echo "  # Mağaza modunda çalıştır"
echo "  python src/main.py --mode store"
echo ""
echo "  # Özel konfigürasyon ile çalıştır"
echo "  python src/main.py --config configs/custom.yaml"
echo ""
echo "🔧 Sistem hazır! Çıkış için 'q' tuşuna basın." 