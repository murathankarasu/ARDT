"""
AI Tabanlı Gelişmiş Sınıflandırıcı - Düşük Donanım Optimize
"""

import cv2
import numpy as np
import torch
import logging
import psutil
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoProcessor
from timm import create_model
import torchvision.transforms as transforms
from enum import Enum


class ModelComplexity(Enum):
    """Model karmaşıklık seviyeleri"""
    NANO = "nano"           # Çok hafif - mobil cihazlar
    LIGHT = "light"         # Hafif - düşük donanım
    MEDIUM = "medium"       # Orta - standart donanım
    HEAVY = "heavy"         # Ağır - yüksek donanım


class AIClassifier:
    """AI tabanlı gelişmiş nesne sınıflandırıcı - Optimize edilmiş"""
    
    # Model konfigürasyonları
    MODEL_CONFIGS = {
        ModelComplexity.NANO: {
            'clip_model': None,  # CLIP kullanma
            'efficientnet': 'efficientnet_lite0',
            'resnet': None,      # ResNet kullanma
            'image_size': 128,
            'batch_processing': False,
            'ensemble': False
        },
        ModelComplexity.LIGHT: {
            'clip_model': 'openai/clip-vit-small-patch32',
            'efficientnet': 'efficientnet_b0',
            'resnet': None,
            'image_size': 224,
            'batch_processing': False,
            'ensemble': True
        },
        ModelComplexity.MEDIUM: {
            'clip_model': 'openai/clip-vit-base-patch32',
            'efficientnet': 'efficientnet_b2',
            'resnet': 'resnet34',
            'image_size': 224,
            'batch_processing': True,
            'ensemble': True
        },
        ModelComplexity.HEAVY: {
            'clip_model': 'openai/clip-vit-large-patch14',
            'efficientnet': 'efficientnet_b4',
            'resnet': 'resnet50',
            'image_size': 256,
            'batch_processing': True,
            'ensemble': True
        }
    }
    
    def __init__(self, config: dict):
        """
        Args:
            config: AI sınıflandırıcı konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sınıf etiketleri - önce tanımla
        self.street_classes = [
            'trafik_lambası', 'dur_tabelası', 'hız_limiti', 'park_yasağı',
            'sola_dönüş', 'sağa_dönüş', 'ileri', 'geri_dönüş_yasağı',
            'yaya_geçidi', 'okul_bölgesi', 'hastane', 'benzin_istasyonu'
        ]
        
        self.store_classes = [
            'et_reyonu', 'süt_reyonu', 'meyve_sebze', 'ekmek_reyonu',
            'temizlik_reyonu', 'içecek_reyonu', 'donuk_gıda', 'bakliyat',
            'kasa', 'alışveriş_arabası', 'sepet', 'fiyat_etiketi'
        ]
        
        # Device seçimi - Apple Silicon M2 optimize
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.logger.info(f"AI Classifier çalışıyor: Apple M2 GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"AI Classifier çalışıyor: CUDA GPU")
        else:
            self.device = torch.device('cpu')
            self.logger.info(f"AI Classifier çalışıyor: CPU")
        
        # Sistem performansına göre model karmaşıklığını belirle
        self.model_complexity = self._determine_model_complexity()
        self.model_config = self.MODEL_CONFIGS[self.model_complexity]
        
        self.logger.info(f"Seçilen model karmaşıklığı: {self.model_complexity.value}")
        
        # Modelleri yükle
        self.models = {}
        self._load_models()
        
        # Transforms - dinamik boyut
        image_size = self.model_config['image_size']
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Fast transform (nano mod için)
        self.fast_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Performance tracking
        self.inference_times = []
        self.memory_usage = []
        
        self.logger.info(f"AIClassifier başlatıldı - {self.model_complexity.value} modu")
    
    def _determine_model_complexity(self) -> ModelComplexity:
        """Sistem performansına göre model karmaşıklığını belirle"""
        try:
            # Zorunlu model karmaşıklığı kontrolü
            force_complexity = self.config.get('force_model_complexity')
            if force_complexity:
                try:
                    complexity = ModelComplexity(force_complexity.lower())
                    self.logger.info(f"ZORLA model seçildi: {complexity.value}")
                    return complexity
                except ValueError:
                    self.logger.warning(f"Geçersiz force_model_complexity değeri: {force_complexity}")
            
            # Otomatik seçim devre dışıysa varsayılan LIGHT
            if not self.config.get('auto_select_model', True):
                complexity = ModelComplexity.LIGHT
                self.logger.info("Otomatik model seçimi devre dışı - LIGHT kullanılıyor")
                return complexity
            
            # Sistem özelliklerini kontrol et
            total_memory = psutil.virtual_memory().total / (1024**3)  # GB
            cpu_count = psutil.cpu_count()
            has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
            
            # GPU memory kontrolü
            gpu_memory = 0
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                except:
                    gpu_memory = 0
            elif torch.backends.mps.is_available():
                # M1/M2 için unified memory - toplam RAM'in yarısını varsay
                gpu_memory = total_memory * 0.5
            
            # Karmaşıklık seviyesini belirle
            if total_memory < 4 or cpu_count < 4:
                complexity = ModelComplexity.NANO
                self.logger.info("NANO model seçildi - çok düşük sistem kaynakları")
            elif total_memory < 8 or (not has_gpu and cpu_count < 8):
                complexity = ModelComplexity.LIGHT
                self.logger.info("LIGHT model seçildi - düşük sistem kaynakları")
            elif total_memory < 16 or (has_gpu and gpu_memory < 6):
                complexity = ModelComplexity.MEDIUM
                self.logger.info("MEDIUM model seçildi - orta sistem kaynakları")
            else:
                complexity = ModelComplexity.HEAVY
                self.logger.info("HEAVY model seçildi - yüksek sistem kaynakları")
            
            return complexity
            
        except Exception as e:
            self.logger.error(f"Model karmaşıklığı belirleme hatası: {e}")
            return ModelComplexity.LIGHT  # Güvenli varsayılan
    
    def _load_models(self):
        """AI modellerini yükle - optimize edilmiş"""
        try:
            # CLIP modeli (eğer destekleniyorsa)
            if self.model_config['clip_model']:
                try:
                    model_name = self.model_config['clip_model']
                    self.models['clip_model'] = AutoModel.from_pretrained(model_name)
                    self.models['clip_processor'] = AutoProcessor.from_pretrained(model_name)
                    self.models['clip_model'].to(self.device)
                    
                    # Half precision (GPU'da memory tasarrufu)
                    if self.device.type == 'cuda':
                        self.models['clip_model'].half()
                    
                    self.logger.info(f"CLIP model yüklendi: {model_name}")
                except Exception as e:
                    self.logger.warning(f"CLIP model yüklenemedi: {e}")
            
            # EfficientNet (her zaman yükle)
            if self.model_config['efficientnet']:
                try:
                    model_name = self.model_config['efficientnet']
                    self.models['efficientnet'] = create_model(
                        model_name, 
                        pretrained=True, 
                        num_classes=len(self.street_classes + self.store_classes)
                    )
                    self.models['efficientnet'].to(self.device)
                    self.models['efficientnet'].eval()
                    
                    # Half precision
                    if self.device.type == 'cuda':
                        self.models['efficientnet'].half()
                    
                    self.logger.info(f"EfficientNet model yüklendi: {model_name}")
                except Exception as e:
                    self.logger.error(f"EfficientNet model yüklenemedi: {e}")
            
            # ResNet (sadece medium/heavy modda)
            if self.model_config['resnet']:
                try:
                    model_name = self.model_config['resnet']
                    self.models['resnet'] = create_model(
                        model_name, 
                        pretrained=True, 
                        num_classes=1000
                    )
                    self.models['resnet'].to(self.device)
                    self.models['resnet'].eval()
                    
                    if self.device.type == 'cuda':
                        self.models['resnet'].half()
                    
                    self.logger.info(f"ResNet model yüklendi: {model_name}")
                except Exception as e:
                    self.logger.warning(f"ResNet model yüklenemedi: {e}")
            
            # Memory optimization
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            self.logger.info("AI modelleri başarıyla yüklendi")
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            raise
    
    def classify_region(self, image: np.ndarray, bbox: List[int], 
                       context: str = 'mixed') -> Dict[str, Any]:
        """
        Belirli bölgeyi AI ile sınıflandır - optimize edilmiş
        
        Args:
            image: Ana görüntü
            bbox: Bounding box [x1, y1, x2, y2]
            context: Bağlam ('street', 'store', 'mixed')
            
        Returns:
            Sınıflandırma sonuçları
        """
        try:
            import time
            start_time = time.time()
            
            # Bölgeyi kırp
            x1, y1, x2, y2 = bbox
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return {'class': 'unknown', 'confidence': 0.0, 'method': 'roi_error'}
            
            # Model karmaşıklığına göre farklı stratejiler
            if self.model_complexity == ModelComplexity.NANO:
                result = self._classify_nano(roi, context)
            elif self.model_complexity == ModelComplexity.LIGHT:
                result = self._classify_light(roi, context)
            elif self.model_complexity == ModelComplexity.MEDIUM:
                result = self._classify_medium(roi, context)
            else:  # HEAVY
                result = self._classify_heavy(roi, context)
            
            # Performance tracking
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Memory tracking
            if self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / (1024**2)  # MB
                self.memory_usage.append(memory_used)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sınıflandırma hatası: {e}")
            return {'class': 'unknown', 'confidence': 0.0, 'method': 'classification_error'}
    
    def _classify_nano(self, roi: np.ndarray, context: str) -> Dict[str, Any]:
        """NANO mod - sadece EfficientNet Lite"""
        try:
            # Basit ve hızlı transform
            tensor_image = self.fast_transform(roi).unsqueeze(0).to(self.device)
            
            # EfficientNet inference
            with torch.no_grad():
                outputs = self.models['efficientnet'](tensor_image)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            # Sınıf adını al
            all_classes = self.street_classes + self.store_classes
            if predicted.item() < len(all_classes):
                predicted_class = all_classes[predicted.item()]
            else:
                predicted_class = 'unknown'
            
            return {
                'class': predicted_class,
                'confidence': confidence.item(),
                'method': 'nano_efficientnet'
            }
            
        except Exception as e:
            self.logger.error(f"NANO sınıflandırma hatası: {e}")
            return {'class': 'unknown', 'confidence': 0.0, 'method': 'nano_error'}
    
    def _classify_light(self, roi: np.ndarray, context: str) -> Dict[str, Any]:
        """LIGHT mod - CLIP + EfficientNet"""
        try:
            results = {}
            
            # CLIP ile analiz (eğer mevcut)
            if 'clip_model' in self.models:
                clip_result = self._classify_with_clip(roi, context)
                results['clip'] = clip_result
            
            # EfficientNet ile analiz
            if 'efficientnet' in self.models:
                efficientnet_result = self._classify_with_efficientnet(roi, context)
                results['efficientnet'] = efficientnet_result
            
            # Basit ensemble
            if len(results) > 1:
                return self._simple_ensemble(results)
            else:
                if results:
                    result = list(results.values())[0]
                    if 'method' not in result:
                        result['method'] = 'single_model'
                    return result
                else:
                    return {'class': 'unknown', 'confidence': 0.0, 'method': 'no_results'}
                
        except Exception as e:
            self.logger.error(f"LIGHT sınıflandırma hatası: {e}")
            return {'class': 'unknown', 'confidence': 0.0, 'method': 'light_error'}
    
    def _classify_medium(self, roi: np.ndarray, context: str) -> Dict[str, Any]:
        """MEDIUM mod - CLIP + EfficientNet + ResNet"""
        try:
            results = {}
            
            # Tüm modelleri kullan
            for model_type in ['clip', 'efficientnet', 'resnet']:
                if model_type == 'clip' and 'clip_model' in self.models:
                    results['clip'] = self._classify_with_clip(roi, context)
                elif model_type == 'efficientnet' and 'efficientnet' in self.models:
                    results['efficientnet'] = self._classify_with_efficientnet(roi, context)
                elif model_type == 'resnet' and 'resnet' in self.models:
                    results['resnet'] = self._classify_with_resnet(roi)
            
            # Ensemble results
            return self._ensemble_results(results, context)
            
        except Exception as e:
            self.logger.error(f"MEDIUM sınıflandırma hatası: {e}")
            return {'class': 'unknown', 'confidence': 0.0, 'method': 'medium_error'}
    
    def _classify_heavy(self, roi: np.ndarray, context: str) -> Dict[str, Any]:
        """HEAVY mod - Tüm modeller + gelişmiş ensemble"""
        return self._classify_medium(roi, context)  # Aynı strateji
    
    def _simple_ensemble(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Basit ensemble - en yüksek güven skoru"""
        if not results:
            return {'class': 'unknown', 'confidence': 0.0}
        
        best_result = max(results.values(), key=lambda x: x.get('confidence', 0))
        best_result['method'] = 'simple_ensemble'
        return best_result
    
    def _classify_with_clip(self, image: np.ndarray, context: str) -> Dict[str, Any]:
        """CLIP modeli ile sınıflandırma"""
        try:
            # Bağlama göre metin sorgularını hazırla
            if context == 'street':
                texts = [f"a photo of {cls}" for cls in self.street_classes]
            elif context == 'store':
                texts = [f"a photo of {cls}" for cls in self.store_classes]
            else:
                texts = [f"a photo of {cls}" for cls in self.street_classes + self.store_classes]
            
            # Görüntüyü işle
            inputs = self.models['clip_processor'](
                text=texts, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # CRITICAL FIX: Input tensörları model ile aynı cihaza taşı
            inputs = {key: value.to(self.device) for key, value in inputs.items() if hasattr(value, 'to')}
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.models['clip_model'](**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # En yüksek skoru bul
            max_idx = probs.argmax().item()
            confidence = probs[0][max_idx].item()
            
            if context == 'street':
                predicted_class = self.street_classes[max_idx]
            elif context == 'store':
                predicted_class = self.store_classes[max_idx]
            else:
                all_classes = self.street_classes + self.store_classes
                predicted_class = all_classes[max_idx]
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'method': 'clip'
            }
            
        except Exception as e:
            self.logger.error(f"CLIP sınıflandırma hatası: {e}")
            return {'class': 'unknown', 'confidence': 0.0, 'method': 'clip'}
    
    def _classify_with_efficientnet(self, image: np.ndarray, context: str) -> Dict[str, Any]:
        """EfficientNet ile sınıflandırma"""
        try:
            # Görüntüyü işle
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.models['efficientnet'](tensor_image)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            # Sınıf adını al
            all_classes = self.street_classes + self.store_classes
            if predicted.item() < len(all_classes):
                predicted_class = all_classes[predicted.item()]
            else:
                predicted_class = 'unknown'
            
            return {
                'class': predicted_class,
                'confidence': confidence.item(),
                'method': 'efficientnet'
            }
            
        except Exception as e:
            self.logger.error(f"EfficientNet sınıflandırma hatası: {e}")
            return {'class': 'unknown', 'confidence': 0.0, 'method': 'efficientnet'}
    
    def _classify_with_resnet(self, image: np.ndarray) -> Dict[str, Any]:
        """ResNet ile sınıflandırma"""
        try:
            # ImageNet sınıfları için
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['resnet'](tensor_image)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            # ImageNet sınıf ID'sini genel kategoriye çevir
            predicted_class = self._map_imagenet_to_context(predicted.item())
            
            return {
                'class': predicted_class,
                'confidence': confidence.item(),
                'method': 'resnet'
            }
            
        except Exception as e:
            self.logger.error(f"ResNet sınıflandırma hatası: {e}")
            return {'class': 'unknown', 'confidence': 0.0, 'method': 'resnet'}
    
    def _map_imagenet_to_context(self, class_id: int) -> str:
        """ImageNet sınıf ID'sini kontekst sınıfına çevir"""
        # Basitleştirilmiş mapping - genişletilebilir
        if 400 <= class_id <= 500:  # Araçlar
            return 'trafik_lambası'
        elif 500 <= class_id <= 600:  # Yiyecekler
            return 'meyve_sebze'
        else:
            return 'unknown'
    
    def _ensemble_results(self, results: Dict[str, Dict], context: str) -> Dict[str, Any]:
        """Çoklu model sonuçlarını birleştir"""
        # Ağırlıklı ortalama
        weights = {
            'clip': 0.5,
            'efficientnet': 0.3,
            'resnet': 0.2
        }
        
        # Güvenilir sonuçları filtrele
        valid_results = {k: v for k, v in results.items() 
                        if v.get('confidence', 0) > 0.3}
        
        if not valid_results:
            return {'class': 'unknown', 'confidence': 0.0, 'method': 'ensemble'}
        
        # En yüksek güven skoruna sahip sonucu seç
        best_result = max(valid_results.values(), 
                         key=lambda x: x.get('confidence', 0))
        
        # Ensemble güven skoru hesapla
        ensemble_confidence = sum(
            results[method].get('confidence', 0) * weights.get(method, 0)
            for method in results.keys()
        )
        
        return {
            'class': best_result['class'],
            'confidence': ensemble_confidence,
            'method': 'ensemble',
            'individual_results': results
        } 

    def get_performance_stats(self) -> Dict[str, Any]:
        """Performans istatistikleri"""
        if not self.inference_times:
            return {'avg_inference_time_ms': 0, 'avg_memory_mb': 0}
        
        avg_inference = np.mean(self.inference_times[-100:]) * 1000  # ms
        avg_memory = np.mean(self.memory_usage[-100:]) if self.memory_usage else 0
        
        return {
            'model_complexity': self.model_complexity.value,
            'avg_inference_time_ms': avg_inference,
            'avg_memory_mb': avg_memory,
            'total_inferences': len(self.inference_times),
            'loaded_models': list(self.models.keys()),
            'device': str(self.device)
        }
    
    def optimize_for_performance(self):
        """Performans için optimize et"""
        if self.device.type == 'cuda':
            # CUDA optimizasyonları
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Memory cleanup
            torch.cuda.empty_cache()
            
            self.logger.info("CUDA optimizasyonları uygulandı")
        
        # CPU optimizasyonları
        torch.set_num_threads(min(4, psutil.cpu_count()))
        
        self.logger.info("Performans optimizasyonları uygulandı") 