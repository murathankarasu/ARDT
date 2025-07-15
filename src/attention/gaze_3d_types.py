"""
3D Gaze Veri Tipleri ve Temel Sınıflar
"""

import math
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class GazePoint3D:
    """3D Gaze nokta verisi"""
    x: float
    y: float
    z: float
    confidence: float
    timestamp: float
    
    def distance_to(self, other: 'GazePoint3D') -> float:
        """Başka bir 3D noktaya uzaklık"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def to_2d(self, frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """3D noktayı 2D'ye projekte et"""
        h, w = frame_shape[:2]
        
        # Perspektif projeksiyonu
        focal_length = min(w, h)  # Basit focal length tahmini
        z_factor = focal_length / max(self.z, 1.0)
        
        x_2d = int(self.x * z_factor + w / 2)
        y_2d = int(self.y * z_factor + h / 2)
        
        # Sınırlar içinde tut
        x_2d = max(0, min(w - 1, x_2d))
        y_2d = max(0, min(h - 1, y_2d))
        
        return (x_2d, y_2d)


class GazeCollisionEvent:
    """Çakışma olayı verisi"""
    
    def __init__(self, gaze_point: Tuple[int, int], object_data: Dict[str, Any], 
                 collision_depth: float, collision_timestamp: float):
        self.gaze_point = gaze_point
        self.object_data = object_data
        self.collision_depth = collision_depth
        self.collision_timestamp = collision_timestamp
        self.duration = 0.0
        self.is_active = True
        
    def update_duration(self, current_time: float):
        """Çakışma süresini güncelle"""
        if self.is_active:
            self.duration = current_time - self.collision_timestamp
            
    def end_collision(self, current_time: float):
        """Çakışmayı sonlandır"""
        self.is_active = False
        self.duration = current_time - self.collision_timestamp 