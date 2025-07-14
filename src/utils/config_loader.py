"""
Konfigürasyon yükleyici sınıfı
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Konfigürasyon dosyalarını yükleyen sınıf"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Konfigürasyon dosyası yolu
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {config_path}")
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Konfigürasyon dosyasını yükle"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            
            self.logger.info(f"Konfigürasyon yüklendi: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Konfigürasyon yükleme hatası: {e}")
            raise
    
    @property
    def detection_config(self) -> Dict[str, Any]:
        """Tespit konfigürasyonu"""
        return self.config.get('detection', {})
    
    @property
    def attention_config(self) -> Dict[str, Any]:
        """Dikkat analizi konfigürasyonu"""
        return self.config.get('attention', {})
    
    @property
    def camera_config(self) -> Dict[str, Any]:
        """Kamera konfigürasyonu"""
        return self.config.get('camera', {})
    
    @property
    def processing_config(self) -> Dict[str, Any]:
        """İşleme konfigürasyonu"""
        return self.config.get('processing', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Konfigürasyon değeri al"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value 