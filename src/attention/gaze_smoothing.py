"""
Gaze Smoothing Algoritmaları
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
from .gaze_3d_types import GazePoint3D


class GazeSmoothing:
    """Gaze verilerini yumuşatma sınıfı"""
    
    def __init__(self, window_size: int = 5, smoothing_factor: float = 0.3):
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.gaze_history = deque(maxlen=window_size)
        self.smoothed_gaze = None
        
    def smooth_gaze_point(self, gaze_point: GazePoint3D) -> GazePoint3D:
        """Gaze noktasını yumuşat"""
        self.gaze_history.append(gaze_point)
        
        if len(self.gaze_history) < 2:
            self.smoothed_gaze = gaze_point
            return gaze_point
        
        # Exponential smoothing
        if self.smoothed_gaze is None:
            self.smoothed_gaze = gaze_point
        else:
            # Yumuşatma
            self.smoothed_gaze = GazePoint3D(
                x=self.smoothed_gaze.x * (1 - self.smoothing_factor) + gaze_point.x * self.smoothing_factor,
                y=self.smoothed_gaze.y * (1 - self.smoothing_factor) + gaze_point.y * self.smoothing_factor,
                z=self.smoothed_gaze.z * (1 - self.smoothing_factor) + gaze_point.z * self.smoothing_factor,
                confidence=max(gaze_point.confidence, self.smoothed_gaze.confidence * 0.9),
                timestamp=gaze_point.timestamp
            )
        
        return self.smoothed_gaze
    
    def get_stability_score(self) -> float:
        """Gaze stabilitesi skorunu hesapla"""
        if len(self.gaze_history) < 3:
            return 0.5
        
        # Son noktalar arası ortalama mesafe
        distances = []
        for i in range(1, len(self.gaze_history)):
            dist = self.gaze_history[i].distance_to(self.gaze_history[i-1])
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Stabilite skoru (düşük mesafe = yüksek stabilite)
        stability = max(0.0, 1.0 - (avg_distance / 100.0))
        return stability
    
    def reset(self):
        """Smoothing'i sıfırla"""
        self.gaze_history.clear()
        self.smoothed_gaze = None 