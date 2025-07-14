"""
Logger ayarlama yardımcı fonksiyonları
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, level: int = logging.INFO, 
                log_file: str = None) -> logging.Logger:
    """
    Logger kurulumu
    
    Args:
        name: Logger ismi
        level: Log seviyesi
        log_file: Log dosyası yolu (opsiyonel)
        
    Returns:
        Konfigüre edilmiş logger
    """
    logger = logging.getLogger(name)
    
    # Eğer logger zaten konfigüre edilmişse, onu döndür
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (eğer belirtilmişse)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_log_filename(base_name: str = "ards") -> str:
    """
    Tarih bazlı log dosyası ismi oluştur
    
    Args:
        base_name: Temel dosya ismi
        
    Returns:
        Log dosyası yolu
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"logs/{base_name}_{timestamp}.log" 