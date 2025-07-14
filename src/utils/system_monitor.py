"""
Sistem Kaynak İzleme Modülü
"""

import psutil
import time
import logging
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class PerformanceLevel(Enum):
    """Performans seviyeleri"""
    CRITICAL = "critical"  # %90+ kaynak kullanımı
    HIGH = "high"         # %75-90% kaynak kullanımı
    MEDIUM = "medium"     # %50-75% kaynak kullanımı
    LOW = "low"          # %0-50% kaynak kullanımı


@dataclass
class SystemMetrics:
    """Sistem metrikleri"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    gpu_count: int = 0
    gpu_memory_used_mb: List[float] = None
    gpu_utilization: List[float] = None
    temperature: float = 0.0
    process_count: int = 0
    load_average: List[float] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.gpu_memory_used_mb is None:
            self.gpu_memory_used_mb = []
        if self.gpu_utilization is None:
            self.gpu_utilization = []
        if self.load_average is None:
            self.load_average = []


class SystemMonitor:
    """Sistem kaynak izleme sınıfı"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """
        Args:
            monitoring_interval: İzleme aralığı (saniye)
        """
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Metrics history
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1000  # Son 1000 ölçümü sakla
        
        # Callbacks
        self.performance_callbacks: Dict[PerformanceLevel, List[Callable]] = {
            level: [] for level in PerformanceLevel
        }
        
        # GPU support
        self.gpu_available = self._check_gpu_support()
        
        # Initial metrics
        self.current_metrics = self._collect_metrics()
        
        self.logger.info("SystemMonitor başlatıldı")
    
    def _check_gpu_support(self) -> bool:
        """GPU desteğini kontrol et"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.logger.info(f"{len(gpus)} GPU tespit edildi")
                return True
            else:
                self.logger.info("GPU bulunamadı")
                return False
        except ImportError:
            self.logger.warning("GPUtil modülü bulunamadı - GPU izleme devre dışı")
            return False
        except Exception as e:
            self.logger.error(f"GPU kontrol hatası: {e}")
            return False
    
    def start_monitoring(self):
        """İzlemeyi başlat"""
        if self.is_monitoring:
            self.logger.warning("İzleme zaten aktif")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Sistem izleme başlatıldı")
    
    def stop_monitoring(self):
        """İzlemeyi durdur"""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        self.logger.info("Sistem izleme durduruldu")
    
    def _monitoring_loop(self):
        """İzleme döngüsü"""
        while self.is_monitoring:
            try:
                # Metrikleri topla
                self.current_metrics = self._collect_metrics()
                
                # History'e ekle
                self.metrics_history.append(self.current_metrics)
                
                # History boyutunu kontrol et
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # Performans seviyesini kontrol et
                self._check_performance_level()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"İzleme döngüsü hatası: {e}")
                time.sleep(5)  # Hata durumunda daha uzun bekle
    
    def _collect_metrics(self) -> SystemMetrics:
        """Sistem metriklerini topla"""
        try:
            metrics = SystemMetrics()
            metrics.timestamp = time.time()
            
            # CPU metrikleri
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrikleri
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_available_gb = memory.available / (1024**3)
            
            # Disk kullanımı
            disk = psutil.disk_usage('/')
            metrics.disk_usage_percent = disk.percent
            
            # Process sayısı
            metrics.process_count = len(psutil.pids())
            
            # Load average (Unix sistemlerde)
            try:
                metrics.load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                metrics.load_average = [0.0, 0.0, 0.0]
            
            # Sıcaklık (mümkünse)
            try:
                temperatures = psutil.sensors_temperatures()
                if temperatures:
                    # CPU sıcaklığını al
                    cpu_temps = []
                    for name, entries in temperatures.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            for entry in entries:
                                if entry.current:
                                    cpu_temps.append(entry.current)
                    
                    if cpu_temps:
                        metrics.temperature = sum(cpu_temps) / len(cpu_temps)
            except (AttributeError, OSError):
                metrics.temperature = 0.0
            
            # GPU metrikleri
            if self.gpu_available:
                metrics.gpu_count, metrics.gpu_memory_used_mb, metrics.gpu_utilization = self._collect_gpu_metrics()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrik toplama hatası: {e}")
            return SystemMetrics(timestamp=time.time())
    
    def _collect_gpu_metrics(self) -> tuple:
        """GPU metriklerini topla"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            gpu_count = len(gpus)
            memory_used = []
            utilization = []
            
            for gpu in gpus:
                memory_used.append(gpu.memoryUsed)
                utilization.append(gpu.load * 100)  # % cinsinden
            
            return gpu_count, memory_used, utilization
            
        except Exception as e:
            self.logger.error(f"GPU metrik toplama hatası: {e}")
            return 0, [], []
    
    def _check_performance_level(self):
        """Performans seviyesini kontrol et ve callback'leri çağır"""
        metrics = self.current_metrics
        
        # En yüksek kaynak kullanımını bul
        max_usage = max(
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.disk_usage_percent,
            max(metrics.gpu_utilization) if metrics.gpu_utilization else 0
        )
        
        # Performans seviyesini belirle
        if max_usage >= 90:
            level = PerformanceLevel.CRITICAL
        elif max_usage >= 75:
            level = PerformanceLevel.HIGH
        elif max_usage >= 50:
            level = PerformanceLevel.MEDIUM
        else:
            level = PerformanceLevel.LOW
        
        # Callback'leri çağır
        for callback in self.performance_callbacks[level]:
            try:
                callback(metrics, level)
            except Exception as e:
                self.logger.error(f"Performance callback hatası: {e}")
    
    def add_performance_callback(self, level: PerformanceLevel, callback: Callable):
        """Performans seviyesi callback'i ekle"""
        self.performance_callbacks[level].append(callback)
        self.logger.info(f"Callback eklendi: {level.value}")
    
    def get_current_metrics(self) -> SystemMetrics:
        """Güncel metrikleri al"""
        return self.current_metrics
    
    def get_metrics_history(self, last_n: Optional[int] = None) -> List[SystemMetrics]:
        """Metrik geçmişini al"""
        if last_n is None:
            return self.metrics_history.copy()
        else:
            return self.metrics_history[-last_n:].copy()
    
    def get_average_metrics(self, duration_seconds: int = 60) -> SystemMetrics:
        """Belirli süre için ortalama metrikleri hesapla"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        # İlgili metrikleri filtrele
        relevant_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not relevant_metrics:
            return self.current_metrics
        
        # Ortalamaları hesapla
        avg_metrics = SystemMetrics()
        avg_metrics.timestamp = current_time
        
        avg_metrics.cpu_percent = sum(m.cpu_percent for m in relevant_metrics) / len(relevant_metrics)
        avg_metrics.memory_percent = sum(m.memory_percent for m in relevant_metrics) / len(relevant_metrics)
        avg_metrics.memory_available_gb = sum(m.memory_available_gb for m in relevant_metrics) / len(relevant_metrics)
        avg_metrics.disk_usage_percent = sum(m.disk_usage_percent for m in relevant_metrics) / len(relevant_metrics)
        
        if relevant_metrics[0].gpu_utilization:
            avg_metrics.gpu_utilization = [
                sum(m.gpu_utilization[i] for m in relevant_metrics if i < len(m.gpu_utilization)) / 
                len([m for m in relevant_metrics if i < len(m.gpu_utilization)])
                for i in range(len(relevant_metrics[0].gpu_utilization))
            ]
        
        return avg_metrics
    
    def is_system_under_stress(self, threshold: float = 80.0) -> bool:
        """Sistem stres altında mı kontrol et"""
        metrics = self.current_metrics
        
        return (metrics.cpu_percent > threshold or 
                metrics.memory_percent > threshold or
                (metrics.gpu_utilization and max(metrics.gpu_utilization) > threshold))
    
    def get_optimization_suggestions(self) -> List[str]:
        """Optimizasyon önerileri"""
        suggestions = []
        metrics = self.current_metrics
        
        if metrics.cpu_percent > 80:
            suggestions.append("CPU kullanımı yüksek - işlem sayısını azaltın")
        
        if metrics.memory_percent > 85:
            suggestions.append("Memory kullanımı yüksek - cache'leri temizleyin")
        
        if metrics.gpu_utilization and max(metrics.gpu_utilization) > 85:
            suggestions.append("GPU kullanımı yüksek - model karmaşıklığını azaltın")
        
        if metrics.memory_available_gb < 1.0:
            suggestions.append("Kullanılabilir RAM düşük - uygulamaları kapatın")
        
        if metrics.disk_usage_percent > 90:
            suggestions.append("Disk alanı düşük - gereksiz dosyaları silin")
        
        if metrics.temperature > 80:
            suggestions.append("CPU sıcaklığı yüksek - soğutmayı kontrol edin")
        
        return suggestions
    
    def export_metrics_summary(self) -> Dict:
        """Metrik özetini export et"""
        if not self.metrics_history:
            return {}
        
        current = self.current_metrics
        avg_60s = self.get_average_metrics(60)
        avg_300s = self.get_average_metrics(300)
        
        return {
            'current': {
                'cpu_percent': current.cpu_percent,
                'memory_percent': current.memory_percent,
                'memory_available_gb': current.memory_available_gb,
                'gpu_utilization': current.gpu_utilization,
                'temperature': current.temperature
            },
            'average_1min': {
                'cpu_percent': avg_60s.cpu_percent,
                'memory_percent': avg_60s.memory_percent,
                'gpu_utilization': avg_60s.gpu_utilization
            },
            'average_5min': {
                'cpu_percent': avg_300s.cpu_percent,
                'memory_percent': avg_300s.memory_percent,
                'gpu_utilization': avg_300s.gpu_utilization
            },
            'optimization_suggestions': self.get_optimization_suggestions(),
            'system_under_stress': self.is_system_under_stress(),
            'monitoring_duration': len(self.metrics_history) * self.monitoring_interval
        } 