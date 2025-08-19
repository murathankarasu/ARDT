Here’s your **English `README.md` version** of the ARDS project, fully translated and formatted for GitHub:

---

# ARDS - AI-Powered Micro-Level Location Recognition & Attention Detection System

<div align="center">

![ARDS Logo](docs/images/logo.png)

**Advanced AI technologies for micro-level location recognition and attention analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Firebase](https://img.shields.io/badge/Firebase-Enabled-orange.svg)](https://firebase.google.com/)

</div>

## 🎯 Project Goal

ARDS is an advanced AI system for micro-level location recognition and attention analysis in camera images. From street signs to store shelves, billboards to price tags, it detects objects and analyzes people’s gaze direction to estimate which objects they are focusing on.

## ✨ Key Features

### 🔍 Advanced AI Classifier

* **CLIP Model**: Context-aware image-text classification
* **EfficientNet**: High-accuracy image classification
* **ResNet**: General object recognition
* **Ensemble Learning**: Combine results from multiple models

### 📹 Adaptive Camera System

* **Automatic Scene Detection**: Recognizes day/night, street/store automatically
* **Dynamic Optimization**: Auto-adjusts camera settings based on scene type
* **Real-Time Quality Analysis**: Continuous frame quality monitoring
* **Autonomous Vehicle Logic**: Tesla-style adaptive camera optimization

### 👁️ Advanced Attention Analysis

* **Eye Tracking**: Precise gaze detection with MediaPipe
* **Object-Attention Relation**: Analyzes which objects are looked at
* **Interest Level**: Measures attention intensity
* **Temporal Analysis**: Tracks attention distribution over time

### 📊 Firebase Analytics

* **Real-Time Data**: All detections and attention data stored in Firestore
* **Batch Processing**: Efficient data handling
* **Session Management**: Session-based analytics
* **Comprehensive Metrics**: Detailed performance insights

## 🚀 Quick Start

### Automatic Setup (Recommended)

```bash
git clone https://github.com/username/ARDS.git
cd ARDS
./scripts/quick_start.sh
```

### Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/username/ARDS.git
cd ARDS

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download AI models
python scripts/install_models.py

# 5. Run!
python src/main.py --mode mixed
```

## 💻 System Requirements

### Minimum

* **CPU**: Intel i5-8400 / AMD Ryzen 5 2600X
* **RAM**: 8GB DDR4
* **GPU**: NVIDIA GTX 1050 Ti (4GB) / AMD RX 570
* **Python**: 3.8+

### Recommended

* **CPU**: Intel i7-10700K / AMD Ryzen 7 3700X
* **RAM**: 16GB DDR4
* **GPU**: NVIDIA RTX 3060 (8GB) / AMD RX 6600 XT
* **Python**: 3.10+

### Production

* **CPU**: Intel i9-12900K / AMD Ryzen 9 5900X
* **RAM**: 32GB+ DDR4
* **GPU**: NVIDIA RTX 4070+ (12GB+ VRAM)

## 🎮 Usage Examples

### Basic Usage

```bash
# Mixed mode - street & store analysis (auto optimization)
python src/main.py --mode mixed

# Street-only analysis
python src/main.py --mode street

# Store-only analysis
python src/main.py --mode store

# With Firebase analytics
python src/main.py --mode mixed --firebase

# Force GPU usage
python src/main.py --mode mixed --gpu

# Detailed logging
python src/main.py --mode mixed --verbose
```

### 🔧 Performance Optimization (NEW!)

```bash
# Automatic performance tuning (default)
python src/main.py --performance adaptive

# Power save mode (low-end hardware)
python src/main.py --performance power_save

# Balanced performance
python src/main.py --performance balanced

# Maximum quality (high-end hardware)
python src/main.py --performance performance

# Mobile/Embedded devices
python src/main.py --config configs/mobile.yaml --performance power_save

# Systems with <4GB RAM
python src/main.py --performance power_save --mode mixed
```

### 📱 Low-End Device Support

ARDS now runs even on devices with **2GB RAM** and **2 CPU cores**!

**Automatic Optimization Features:**

* 🧠 Auto model selection based on system resources
* 📊 Real-time performance monitoring
* ⚡ Adaptive frame processing
* 🔋 Battery saver mode
* 📱 Mobile support

### Runtime Controls

* **q**: Quit application
* **s**: Take screenshot
* **i**: Show detailed system info
* **p**: Toggle performance mode manually

## 🏗️ Project Structure

```
ARDS/
├── src/                      
│   ├── core/                
│   │   ├── adaptive_camera.py    
│   │   └── image_processor.py    
│   ├── detection/           
│   │   ├── object_detector.py    
│   │   ├── ai_classifier.py     
│   │   ├── street_detector.py   
│   │   └── store_detector.py    
│   ├── attention/           
│   │   ├── gaze_tracker.py      
│   │   └── attention_analyzer.py 
│   ├── analytics/           
│   │   └── firestore_manager.py 
│   ├── utils/               
│   └── main.py             
├── models/                  
│   ├── pretrained/         
│   ├── custom/             
│   └── cache/              
├── configs/                
├── scripts/                
├── docs/                   
└── requirements.txt        
```

## 🔧 Configuration

### Main Config (`configs/default.yaml`)

```yaml
camera:
  adaptive_mode: true
  scene_analysis_interval: 0.5
  quality_threshold: 0.7

ai_classifier:
  device: "auto"  # auto/cuda/cpu
  clip_model: "openai/clip-vit-base-patch32"
  ensemble_weights:
    clip: 0.5
    efficientnet: 0.3
    resnet: 0.2

firebase:
  enabled: false
  credentials_path: "path/to/service-account.json"
  project_id: "your-project-id"

performance:
  use_gpu: true
  max_fps: 30
  worker_threads: 4
```

## 📊 Firebase Analytics Setup

1. Create a project in **Firebase Console**
2. Enable **Firestore Database**
3. Generate a **Service Account key**
4. Update your configuration

Details: [Firebase Setup Guide](docs/FIREBASE_SETUP.md)

## 🎯 Application Areas

### 🏪 Retail & E-commerce

* Customer behavior analysis
* Shelf optimization
* Product placement analysis
* Attention heatmaps

### 🚗 Traffic & Transportation

* Traffic sign detection
* Direction sign analysis
* Driver attention monitoring
* Security cameras

### 🏢 Security & Surveillance

* Suspicious behavior detection
* Attention analysis
* Object tracking
* Event logging

### 📱 Mobile Applications

* Augmented reality
* Visual search
* Smart assistants
* Accessibility apps

## 🧠 AI Models & Technologies

### Object Detection

* **YOLOv8**: Fast and accurate object detection
* **Custom Detectors**: Specialized for street/store elements

### Image Classification

* **CLIP**: Image-text understanding
* **EfficientNet**: High accuracy
* **ResNet**: General classification

### Attention Analysis

* **MediaPipe**: Face & eye landmark detection
* **Custom Algorithms**: Gaze direction estimation

## 📈 Performance Metrics

| Feature                | Value                 |
| ---------------------- | --------------------- |
| **FPS**                | 30+ (GPU) / 15+ (CPU) |
| **Detection Accuracy** | 92%+                  |
| **Attention Accuracy** | 85%+                  |
| **Memory Usage**       | 2-8GB                 |
| **GPU Usage**          | 3-6GB VRAM            |

## 📚 Documentation

* [Advanced Usage Guide](docs/ADVANCED_USAGE.md)
* [API Documentation](docs/API.md)
* [Model Training](docs/TRAINING.md)
* [Deployment Guide](docs/DEPLOYMENT.md)

## 🤝 Contributing

We welcome contributions! Please check:

1. [Contributing Guidelines](CONTRIBUTING.md)
2. [Code of Conduct](CODE_OF_CONDUCT.md)
3. [Development Setup](docs/DEVELOPMENT.md)

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🆘 Support & Contact

* **🐛 Bug Reports**: [GitHub Issues](https://github.com/username/ARDS/issues)
* **💡 Feature Requests**: [GitHub Discussions](https://github.com/username/ARDS/discussions)
* **📧 Email**: [support@ards.com](mailto:support@ards.com)
* **💬 Discord**: [ARDS Community](https://discord.gg/ards)

## 🙏 Acknowledgements

* [OpenAI CLIP](https://github.com/openai/CLIP)
* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [MediaPipe](https://github.com/google/mediapipe)
* [PyTorch](https://pytorch.org/)
* [OpenCV](https://opencv.org/)

---

<div align="center">

**⭐ If you like this project, don’t forget to star it!**

Made with ❤️ by ARDS Team

</div>

---

Do you also want me to polish this further into a **GitHub-optimized README** (with badges, demo GIF placeholders, and a shorter TL;DR at the top), or keep it strictly as a full translation?
