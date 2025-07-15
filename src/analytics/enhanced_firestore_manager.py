"""
Gelişmiş Firestore Entegrasyonu - 3D Uzaysal Bilgilerin Detaylı Kaydı
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import json
import uuid

from .firestore_manager import FirestoreAnalytics, HumanReadableEvent
from attention.true_3d_collision_detection import Collision3DEvent
from attention.enhanced_head_pose import HeadPose3D, GazeVector3D
from detection.object_3d_positioning import Object3DPosition
from spatial.spatial_environment_analyzer import SpatialObject, SpatialPerson


@dataclass
class Spatial3DSession:
    """3D uzaysal session verisi"""
    session_id: str
    start_timestamp: float
    end_timestamp: Optional[float] = None
    duration: float = 0.0
    
    # Uzaysal bilgiler
    environment_bounds: Dict[str, Any] = field(default_factory=dict)
    total_objects_detected: int = 0
    total_people_detected: int = 0
    unique_object_types: List[str] = field(default_factory=list)
    
    # Çakışma istatistikleri
    total_3d_collisions: int = 0
    meaningful_collisions: int = 0
    collision_by_object_type: Dict[str, int] = field(default_factory=dict)
    
    # Gaze analizi
    total_gaze_duration: float = 0.0
    average_gaze_confidence: float = 0.0
    gaze_stability_score: float = 0.0
    
    # Spatial relationships
    object_interaction_count: int = 0
    spatial_density_score: float = 0.0
    environment_complexity_score: float = 0.0
    
    # Metadata
    camera_position: Optional[Tuple[float, float, float]] = None
    world_coordinate_system: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary formatına çevir"""
        return asdict(self)


@dataclass  
class EnhancedAttentionEvent:
    """Gelişmiş dikkat eventi"""
    event_id: str
    timestamp: float
    event_type: str  # "3d_collision_start", "3d_collision_end", "gaze_shift", "object_focus"
    
    # 3D spatial context
    person_head_position: Optional[Tuple[float, float, float]] = None
    person_gaze_direction: Optional[Tuple[float, float, float]] = None
    
    # Obje bilgileri
    object_id: Optional[str] = None
    object_label: Optional[str] = None
    object_3d_position: Optional[Tuple[float, float, float]] = None
    object_3d_dimensions: Optional[Tuple[float, float, float]] = None
    object_distance_from_person: Optional[float] = None
    
    # Çakışma bilgileri
    collision_3d_point: Optional[Tuple[float, float, float]] = None
    collision_angle: Optional[float] = None
    collision_confidence: Optional[float] = None
    penetration_depth: Optional[float] = None
    
    # Context
    nearby_objects: List[Dict[str, Any]] = field(default_factory=list)
    spatial_context: Dict[str, Any] = field(default_factory=dict)
    
    # Human readable
    human_description: str = ""
    severity_level: str = "normal"  # "low", "normal", "high", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary formatına çevir"""
        return asdict(self)


@dataclass
class SpatialEnvironmentSnapshot:
    """Uzaysal çevre snapshot'ı"""
    snapshot_id: str
    timestamp: float
    
    # Çevre bilgileri
    environment_bounds: Dict[str, Any]
    object_positions: List[Dict[str, Any]]
    people_positions: List[Dict[str, Any]]
    
    # Spatial relationships
    object_distances: Dict[str, float]
    spatial_clusters: List[Dict[str, Any]]
    density_map: Dict[str, Any]
    
    # Gaze information
    active_gaze_vectors: List[Dict[str, Any]]
    gaze_target_predictions: List[Dict[str, Any]]
    
    # Analytics
    complexity_score: float
    activity_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary formatına çevir"""
        return asdict(self)


class Enhanced3DFirestoreManager(FirestoreAnalytics):
    """Gelişmiş 3D Firestore yönetici sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # 3D specific collections
        self.spatial_sessions_collection = "spatial_3d_sessions"
        self.enhanced_attention_collection = "enhanced_attention_events"
        self.spatial_snapshots_collection = "spatial_environment_snapshots"
        self.collision_3d_collection = "collision_3d_events"
        self.gaze_3d_collection = "gaze_3d_tracking"
        
        # Session management
        self.current_session: Optional[Spatial3DSession] = None
        self.session_start_time = time.time()
        
        # Buffering
        self.event_buffer = deque(maxlen=config.get('event_buffer_size', 1000))
        self.snapshot_buffer = deque(maxlen=config.get('snapshot_buffer_size', 100))
        
        # Analytics parameters
        self.snapshot_interval = config.get('snapshot_interval', 30.0)  # 30 saniye
        self.meaningful_collision_threshold = config.get('meaningful_collision_threshold', 1.0)  # 1 saniye
        self.high_activity_threshold = config.get('high_activity_threshold', 0.7)
        
        # Counters
        self.session_counters = {
            'total_3d_events': 0,
            'total_collisions': 0,
            'total_snapshots': 0,
            'unique_objects': set(),
            'unique_people': set()
        }
        
        self.last_snapshot_time = time.time()
        
        # Initialize session
        self._initialize_new_session()
        
        self.logger.info("Enhanced3DFirestoreManager başlatıldı")
    
    def _initialize_new_session(self):
        """Yeni session başlat"""
        try:
            session_id = f"3d_session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
            self.current_session = Spatial3DSession(
                session_id=session_id,
                start_timestamp=time.time(),
                world_coordinate_system={
                    'origin': (0, 0, 0),
                    'scale': 1.0,
                    'coordinate_system': 'camera_relative'
                }
            )
            
            self.session_start_time = time.time()
            self.session_counters = {
                'total_3d_events': 0,
                'total_collisions': 0,
                'total_snapshots': 0,
                'unique_objects': set(),
                'unique_people': set()
            }
            
            self.logger.info(f"Yeni 3D session başlatıldı: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Session başlatma hatası: {e}")
    
    def log_3d_collision_event(self, collision_event: Collision3DEvent,
                              spatial_context: Dict[str, Any] = None):
        """3D çakışma eventi kaydet"""
        try:
            # Enhanced attention event oluştur
            enhanced_event = EnhancedAttentionEvent(
                event_id=f"collision_{collision_event.collision_id}_{int(time.time() * 1000)}",
                timestamp=collision_event.start_timestamp,
                event_type="3d_collision_start" if collision_event.is_active else "3d_collision_end",
                object_id=collision_event.object_id,
                object_label=collision_event.object_label,
                object_3d_position=collision_event.object_position,
                object_3d_dimensions=collision_event.object_dimensions,
                collision_3d_point=collision_event.collision_point_3d,
                collision_angle=collision_event.collision_angle,
                collision_confidence=collision_event.collision_confidence,
                penetration_depth=collision_event.penetration_depth,
                spatial_context=spatial_context or {},
                human_description=self._generate_collision_description(collision_event),
                severity_level=self._determine_collision_severity(collision_event)
            )
            
            # Firestore'a kaydet
            self.log_basic_event(self.enhanced_attention_collection, enhanced_event.to_dict())
            
            # Detailed collision event
            collision_details = {
                **collision_event.to_dict(),
                'spatial_context': spatial_context,
                'human_readable': {
                    'description': enhanced_event.human_description,
                    'severity': enhanced_event.severity_level,
                    'object_description': f"{collision_event.object_label} objesine bakış",
                    'collision_quality': self._describe_collision_quality(collision_event)
                }
            }
            
            self.log_basic_event(self.collision_3d_collection, collision_details)
            
            # Buffer'a ekle
            self.event_buffer.append(enhanced_event)
            
            # Session counters güncelle
            self.session_counters['total_3d_events'] += 1
            self.session_counters['total_collisions'] += 1
            self.session_counters['unique_objects'].add(collision_event.object_label)
            
            # Session güncelle
            if self.current_session:
                self.current_session.total_3d_collisions += 1
                
                if collision_event.duration >= self.meaningful_collision_threshold:
                    self.current_session.meaningful_collisions += 1
                
                if collision_event.object_label not in self.current_session.collision_by_object_type:
                    self.current_session.collision_by_object_type[collision_event.object_label] = 0
                self.current_session.collision_by_object_type[collision_event.object_label] += 1
            
            # Human readable log
            self.log_attention_activity(
                enhanced_event.human_description,
                duration=collision_event.duration,
                confidence=collision_event.collision_confidence,
                object_info={
                    'label': collision_event.object_label,
                    'position_3d': collision_event.object_position,
                    'distance': collision_event.collision_distance
                }
            )
            
        except Exception as e:
            self.logger.error(f"3D çakışma eventi kaydetme hatası: {e}")
    
    def log_3d_gaze_tracking(self, gaze_vector: GazeVector3D, head_pose: HeadPose3D = None,
                            spatial_context: Dict[str, Any] = None):
        """3D gaze tracking kaydet"""
        try:
            gaze_event = {
                'timestamp': gaze_vector.timestamp,
                'gaze_origin_3d': gaze_vector.origin_3d,
                'gaze_direction_3d': gaze_vector.direction_3d,
                'gaze_target_3d': gaze_vector.target_point_3d,
                'gaze_confidence': gaze_vector.confidence,
                'left_eye_3d': gaze_vector.left_eye_3d,
                'right_eye_3d': gaze_vector.right_eye_3d,
                'head_pose_3d': head_pose.to_dict() if head_pose else None,
                'spatial_context': spatial_context,
                'human_readable': self._generate_gaze_description(gaze_vector, head_pose)
            }
            
            # Firestore'a kaydet (batch olarak)
            self.log_basic_event(self.gaze_3d_collection, gaze_event)
            
            # Session güncelle
            if self.current_session:
                self.current_session.total_gaze_duration += 0.1  # Frame duration assumption
                self.current_session.average_gaze_confidence = (
                    self.current_session.average_gaze_confidence * 0.9 + gaze_vector.confidence * 0.1
                )
            
        except Exception as e:
            self.logger.error(f"3D gaze tracking kaydetme hatası: {e}")
    
    def log_spatial_environment_snapshot(self, spatial_objects: List[Object3DPosition],
                                       spatial_people: List[SpatialPerson],
                                       environment_analysis: Dict[str, Any]):
        """Uzaysal çevre snapshot'ı kaydet"""
        try:
            current_time = time.time()
            
            # Snapshot interval kontrolü
            if current_time - self.last_snapshot_time < self.snapshot_interval:
                return
            
            # Snapshot oluştur
            snapshot = SpatialEnvironmentSnapshot(
                snapshot_id=f"snapshot_{int(current_time * 1000)}_{uuid.uuid4().hex[:6]}",
                timestamp=current_time,
                environment_bounds=environment_analysis.get('environment_bounds', {}),
                object_positions=[obj.to_dict() for obj in spatial_objects],
                people_positions=[person.to_dict() for person in spatial_people],
                object_distances=environment_analysis.get('spatial_relationships', {}).get('object_distances', {}),
                spatial_clusters=environment_analysis.get('spatial_relationships', {}).get('spatial_clusters', []),
                density_map=self._calculate_spatial_density(spatial_objects, spatial_people),
                active_gaze_vectors=[],  # Gaze vectors from current frame
                gaze_target_predictions=[],  # Predicted gaze targets
                complexity_score=self._calculate_environment_complexity(spatial_objects, spatial_people),
                activity_level=self._determine_activity_level(spatial_objects, spatial_people)
            )
            
            # Firestore'a kaydet
            self.log_basic_event(self.spatial_snapshots_collection, snapshot.to_dict())
            
            # Buffer'a ekle
            self.snapshot_buffer.append(snapshot)
            
            # Session güncelle
            if self.current_session:
                self.current_session.environment_bounds = snapshot.environment_bounds
                self.current_session.total_objects_detected = len(spatial_objects)
                self.current_session.total_people_detected = len(spatial_people)
                self.current_session.unique_object_types = list(set(obj.label for obj in spatial_objects))
                self.current_session.spatial_density_score = snapshot.density_map.get('overall_density', 0.0)
                self.current_session.environment_complexity_score = snapshot.complexity_score
            
            self.last_snapshot_time = current_time
            self.session_counters['total_snapshots'] += 1
            
            # Human readable log
            self._log_environment_summary(snapshot)
            
        except Exception as e:
            self.logger.error(f"Spatial snapshot kaydetme hatası: {e}")
    
    def log_enhanced_attention_analytics(self, attention_analysis: Dict[str, Any],
                                       collision_analysis: Dict[str, Any]):
        """Gelişmiş dikkat analitikleri kaydet"""
        try:
            current_time = time.time()
            
            # Combined analytics
            enhanced_analytics = {
                'timestamp': current_time,
                'attention_analysis': attention_analysis,
                'collision_analysis': collision_analysis,
                'session_summary': self._generate_session_summary(),
                'spatial_insights': self._generate_spatial_insights(),
                'behavioral_patterns': self._analyze_behavioral_patterns(),
                'human_readable_insights': self._generate_human_readable_insights(
                    attention_analysis, collision_analysis
                )
            }
            
            # Firestore'a kaydet
            self.log_basic_event('enhanced_analytics', enhanced_analytics)
            
            # Performance insight
            self.log_performance_insight(
                f"3D Spatial Analytics: {len(self.session_counters['unique_objects'])} unique objects, "
                f"{self.session_counters['total_collisions']} collisions, "
                f"{self.session_counters['total_snapshots']} snapshots"
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced analytics kaydetme hatası: {e}")
    
    def finalize_session(self):
        """Session'ı sonlandır"""
        try:
            if not self.current_session:
                return
            
            current_time = time.time()
            
            # Session'ı tamamla
            self.current_session.end_timestamp = current_time
            self.current_session.duration = current_time - self.current_session.start_timestamp
            
            # Final statistics
            self.current_session.unique_object_types = list(self.session_counters['unique_objects'])
            
            # Session summary
            session_summary = self._generate_final_session_summary()
            
            # Firestore'a kaydet
            session_data = {
                **self.current_session.to_dict(),
                'session_summary': session_summary,
                'final_statistics': self.session_counters.copy()
            }
            
            # Set'leri list'e çevir
            session_data['final_statistics']['unique_objects'] = list(self.session_counters['unique_objects'])
            session_data['final_statistics']['unique_people'] = list(self.session_counters['unique_people'])
            
            self.log_basic_event(self.spatial_sessions_collection, session_data)
            
            # Human readable session summary
            self.log_session_summary(session_summary)
            
            self.logger.info(f"3D Session sonlandırıldı: {self.current_session.session_id}")
            
            # Yeni session için hazırlık
            self._initialize_new_session()
            
        except Exception as e:
            self.logger.error(f"Session sonlandırma hatası: {e}")
    
    def _generate_collision_description(self, collision: Collision3DEvent) -> str:
        """Çakışma için human readable açıklama"""
        try:
            distance_desc = "yakın" if collision.collision_distance < 500 else "uzak"
            confidence_desc = "yüksek" if collision.collision_confidence > 0.7 else "orta" if collision.collision_confidence > 0.4 else "düşük"
            
            duration_desc = ""
            if collision.duration > 0:
                if collision.duration >= 3.0:
                    duration_desc = f"{collision.duration:.1f} saniye boyunca uzun süre"
                elif collision.duration >= 1.0:
                    duration_desc = f"{collision.duration:.1f} saniye boyunca"
                else:
                    duration_desc = f"{collision.duration:.1f} saniye kısa süre"
            
            return f"Kişi {distance_desc} mesafedeki {collision.object_label} objesine {duration_desc} " \
                   f"bakıyor (güven: {confidence_desc}, açı: {collision.collision_angle:.1f}°)"
            
        except Exception as e:
            self.logger.error(f"Çakışma açıklaması oluşturma hatası: {e}")
            return f"Çakışma: {collision.object_label}"
    
    def _determine_collision_severity(self, collision: Collision3DEvent) -> str:
        """Çakışma şiddeti belirle"""
        try:
            if collision.collision_confidence > 0.8 and collision.duration > 3.0:
                return "high"
            elif collision.collision_confidence > 0.6 and collision.duration > 1.0:
                return "normal"
            elif collision.collision_confidence > 0.4:
                return "low"
            else:
                return "minimal"
                
        except Exception as e:
            self.logger.error(f"Çakışma şiddeti belirleme hatası: {e}")
            return "normal"
    
    def _describe_collision_quality(self, collision: Collision3DEvent) -> str:
        """Çakışma kalitesi açıkla"""
        try:
            quality_factors = []
            
            if collision.collision_confidence > 0.7:
                quality_factors.append("yüksek güvenilirlik")
            
            if collision.collision_angle < 30:
                quality_factors.append("doğrudan bakış")
            elif collision.collision_angle < 60:
                quality_factors.append("açılı bakış")
            else:
                quality_factors.append("çevresel bakış")
            
            if collision.penetration_depth > 0:
                quality_factors.append("derinlemesine odaklanma")
            
            return ", ".join(quality_factors) if quality_factors else "standart kalite"
            
        except Exception as e:
            self.logger.error(f"Çakışma kalitesi açıklama hatası: {e}")
            return "standart kalite"
    
    def _generate_gaze_description(self, gaze_vector: GazeVector3D, head_pose: HeadPose3D = None) -> str:
        """Gaze için human readable açıklama"""
        try:
            confidence_desc = "yüksek" if gaze_vector.confidence > 0.7 else "orta" if gaze_vector.confidence > 0.4 else "düşük"
            
            direction_desc = ""
            if head_pose:
                yaw = head_pose.rotation_angles[1]  # Yaw angle
                if abs(yaw) < 0.2:
                    direction_desc = "düz ileri"
                elif yaw > 0.2:
                    direction_desc = "sağa doğru"
                else:
                    direction_desc = "sola doğru"
            
            return f"Kişi {direction_desc} bakıyor (güven: {confidence_desc})"
            
        except Exception as e:
            self.logger.error(f"Gaze açıklaması oluşturma hatası: {e}")
            return "Gaze tracking"
    
    def _calculate_spatial_density(self, objects: List[Object3DPosition], 
                                 people: List[SpatialPerson]) -> Dict[str, Any]:
        """Uzaysal yoğunluk hesapla"""
        try:
            total_entities = len(objects) + len(people)
            
            if total_entities == 0:
                return {'overall_density': 0.0, 'object_density': 0.0, 'people_density': 0.0}
            
            # Basit yoğunluk hesaplaması
            # Gelecekte daha sophisticated hesaplama yapılabilir
            
            return {
                'overall_density': min(1.0, total_entities / 10.0),  # 10 entity = max density
                'object_density': min(1.0, len(objects) / 8.0),
                'people_density': min(1.0, len(people) / 3.0),
                'entity_distribution': {
                    'objects': len(objects),
                    'people': len(people)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Uzaysal yoğunluk hesaplama hatası: {e}")
            return {'overall_density': 0.0}
    
    def _calculate_environment_complexity(self, objects: List[Object3DPosition], 
                                        people: List[SpatialPerson]) -> float:
        """Çevre karmaşıklığı hesapla"""
        try:
            complexity = 0.0
            
            # Nesne çeşitliliği
            unique_object_types = set(obj.label for obj in objects)
            complexity += len(unique_object_types) * 0.1
            
            # Insan sayısı
            complexity += len(people) * 0.3
            
            # Nesne sayısı
            complexity += len(objects) * 0.05
            
            # Spatial distribution
            if len(objects) > 1:
                # Basit dağılım hesaplaması
                complexity += 0.2
            
            return min(1.0, complexity)
            
        except Exception as e:
            self.logger.error(f"Çevre karmaşıklığı hesaplama hatası: {e}")
            return 0.0
    
    def _determine_activity_level(self, objects: List[Object3DPosition], 
                                people: List[SpatialPerson]) -> str:
        """Aktivite seviyesi belirle"""
        try:
            total_entities = len(objects) + len(people)
            
            if total_entities == 0:
                return "empty"
            elif total_entities <= 2:
                return "low"
            elif total_entities <= 5:
                return "moderate"
            elif total_entities <= 10:
                return "high"
            else:
                return "very_high"
                
        except Exception as e:
            self.logger.error(f"Aktivite seviyesi belirleme hatası: {e}")
            return "unknown"
    
    def _log_environment_summary(self, snapshot: SpatialEnvironmentSnapshot):
        """Çevre özeti kaydet"""
        try:
            summary = f"Çevre durumu: {len(snapshot.object_positions)} nesne, " \
                     f"{len(snapshot.people_positions)} kişi, " \
                     f"aktivite: {snapshot.activity_level}, " \
                     f"karmaşıklık: {snapshot.complexity_score:.2f}"
            
            self.log_attention_activity(summary, duration=0, confidence=0.8)
            
        except Exception as e:
            self.logger.error(f"Çevre özeti kaydetme hatası: {e}")
    
    def _generate_session_summary(self) -> Dict[str, Any]:
        """Session özeti oluştur"""
        try:
            current_time = time.time()
            session_duration = current_time - self.session_start_time
            
            return {
                'session_duration': session_duration,
                'total_3d_events': self.session_counters['total_3d_events'],
                'total_collisions': self.session_counters['total_collisions'],
                'total_snapshots': self.session_counters['total_snapshots'],
                'unique_objects_count': len(self.session_counters['unique_objects']),
                'unique_people_count': len(self.session_counters['unique_people']),
                'events_per_minute': self.session_counters['total_3d_events'] / (session_duration / 60) if session_duration > 0 else 0,
                'collision_rate': self.session_counters['total_collisions'] / (session_duration / 60) if session_duration > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Session özeti oluşturma hatası: {e}")
            return {}
    
    def _generate_spatial_insights(self) -> Dict[str, Any]:
        """Uzaysal içgörüler oluştur"""
        try:
            insights = {
                'most_common_objects': [],
                'spatial_patterns': [],
                'interaction_hotspots': [],
                'attention_distribution': {}
            }
            
            # En yaygın objeler
            if self.session_counters['unique_objects']:
                insights['most_common_objects'] = list(self.session_counters['unique_objects'])[:5]
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Uzaysal içgörü oluşturma hatası: {e}")
            return {}
    
    def _analyze_behavioral_patterns(self) -> Dict[str, Any]:
        """Davranış desenlerini analiz et"""
        try:
            patterns = {
                'attention_patterns': [],
                'gaze_stability': 0.0,
                'object_preference': {},
                'interaction_frequency': 0.0
            }
            
            # Basit pattern analizi
            if self.current_session:
                patterns['gaze_stability'] = self.current_session.gaze_stability_score
                patterns['object_preference'] = self.current_session.collision_by_object_type
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Davranış deseni analizi hatası: {e}")
            return {}
    
    def _generate_human_readable_insights(self, attention_analysis: Dict[str, Any], 
                                        collision_analysis: Dict[str, Any]) -> List[str]:
        """İnsan tarafından okunabilir içgörüler"""
        try:
            insights = []
            
            # Attention insights
            if attention_analysis.get('most_attended_objects'):
                most_attended = attention_analysis['most_attended_objects'][0]
                insights.append(f"En çok dikkat çeken nesne: {most_attended[0]} ({most_attended[1]:.1f}s)")
            
            # Collision insights
            if collision_analysis.get('total_meaningful_collisions', 0) > 0:
                insights.append(f"Toplam anlamlı çakışma: {collision_analysis['total_meaningful_collisions']}")
            
            # Session insights
            if self.current_session:
                unique_objects = len(self.current_session.unique_object_types)
                insights.append(f"Tespit edilen farklı nesne türü: {unique_objects}")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"İnsan tarafından okunabilir içgörü oluşturma hatası: {e}")
            return []
    
    def _generate_final_session_summary(self) -> Dict[str, Any]:
        """Final session özeti"""
        try:
            if not self.current_session:
                return {}
            
            return {
                'session_id': self.current_session.session_id,
                'total_duration': self.current_session.duration,
                'meaningful_collisions': self.current_session.meaningful_collisions,
                'unique_objects': len(self.current_session.unique_object_types),
                'environment_complexity': self.current_session.environment_complexity_score,
                'average_gaze_confidence': self.current_session.average_gaze_confidence,
                'collision_efficiency': (
                    self.current_session.meaningful_collisions / 
                    max(1, self.current_session.total_3d_collisions)
                ),
                'human_summary': self._generate_human_session_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Final session özeti oluşturma hatası: {e}")
            return {}
    
    def _generate_human_session_summary(self) -> str:
        """İnsan tarafından okunabilir session özeti"""
        try:
            if not self.current_session:
                return "Session bilgisi mevcut değil"
            
            duration_min = self.current_session.duration / 60
            
            summary = f"Session süresi: {duration_min:.1f} dakika, " \
                     f"{self.current_session.meaningful_collisions} anlamlı çakışma, " \
                     f"{len(self.current_session.unique_object_types)} farklı nesne türü tespit edildi"
            
            if self.current_session.collision_by_object_type:
                most_interacted = max(self.current_session.collision_by_object_type.items(), key=lambda x: x[1])
                summary += f", en çok etkileşim: {most_interacted[0]} ({most_interacted[1]} kez)"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"İnsan session özeti oluşturma hatası: {e}")
            return "Session özeti oluşturulamadı"
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Session istatistikleri al"""
        try:
            return {
                'current_session': self.current_session.to_dict() if self.current_session else None,
                'session_counters': {
                    **self.session_counters,
                    'unique_objects': list(self.session_counters['unique_objects']),
                    'unique_people': list(self.session_counters['unique_people'])
                },
                'buffer_status': {
                    'event_buffer_size': len(self.event_buffer),
                    'snapshot_buffer_size': len(self.snapshot_buffer)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Session istatistikleri alma hatası: {e}")
            return {} 