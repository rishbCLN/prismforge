"""HazardMesh Feature Engineering Module.
Extracts structured numeric spatio-temporal features for hazard risk modeling.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from src.tracking.tracker import WorkerTrack


@dataclass
class HazardFeatures:
    # Worker identification & metadata (non-feature or id)
    track_id: int
    frame_id: int
    timestamp: float

    # 1. PPE features
    helmet_missing: float                  # 1.0 if missing, 0.0 otherwise
    vest_missing: float                    # 1.0 if missing, 0.0 otherwise
    ppe_violation_count: float             # 0.0, 1.0, 2.0
    ppe_violation_confidence: float        # [0.0, 1.0]

    # 2. Spatial features
    normalized_worker_machine_dist: float  # [0.0, 1.0], normalized by frame diagonal
    worker_machine_overlap: float          # [0.0, 1.0], IoU / intersection ratio
    worker_count_machine_zone: float       # count of workers within proximity threshold
    machine_proximity_severity: float      # [0.0, 1.0], 1.0 is direct contact/extreme proximity

    # 3. Temporal features
    violation_duration: float              # seconds of continuous violation
    consecutive_violating_frames: float    # frame count
    frames_since_first_detection: float    # total frames observed
    persistence_ratio: float               # [0.0, 1.0], violating frames / total frames

    # 4. Scene features
    num_workers: float                     # total active workers in scene
    num_machines: float                    # total active machines in scene
    simultaneous_violations: float         # number of concurrent violations in scene
    scene_hazard_density: float            # [0.0, 1.0], simultaneous_violations / max(1, num_workers)

    # 5. Confidence features
    mean_detection_confidence: float       # [0.0, 1.0]
    min_relevant_confidence: float         # [0.0, 1.0]
    confidence_variance: float             # variance in detection confidences

    FEATURE_NAMES = [
        "helmet_missing",
        "vest_missing",
        "ppe_violation_count",
        "ppe_violation_confidence",
        "normalized_worker_machine_dist",
        "worker_machine_overlap",
        "worker_count_machine_zone",
        "machine_proximity_severity",
        "violation_duration",
        "consecutive_violating_frames",
        "frames_since_first_detection",
        "persistence_ratio",
        "num_workers",
        "num_machines",
        "simultaneous_violations",
        "scene_hazard_density",
        "mean_detection_confidence",
        "min_relevant_confidence",
        "confidence_variance"
    ]

    def to_vector(self) -> np.ndarray:
        """Returns numeric vector ordered by FEATURE_NAMES."""
        return np.array([getattr(self, name) for name in self.FEATURE_NAMES], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureExtractor:
    """Computes structured hazard features from tracked workers and scene elements."""

    # Proximity threshold in normalized diagonal units (e.g. 0.20 = 20% of screen diagonal)
    DANGER_ZONE_THRESHOLD = 0.22

    def __init__(self, danger_threshold: float = DANGER_ZONE_THRESHOLD):
        self.danger_threshold = danger_threshold

    @staticmethod
    def calculate_normalized_distance(pos1: Tuple[float, float], pos2: Tuple[float, float], frame_w: float, frame_h: float) -> float:
        """Computes 2D Euclidean distance normalized by frame diagonal."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dist_px = math.sqrt(dx**2 + dy**2)
        diagonal = math.sqrt(frame_w**2 + frame_h**2) + 1e-5
        return float(min(1.0, dist_px / diagonal))

    @staticmethod
    def calculate_box_overlap(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """Calculates overlap intersection ratio relative to boxA area."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        return float(interArea / boxAArea)

    def extract_features(
        self,
        worker: WorkerTrack,
        all_workers: List[WorkerTrack],
        all_machines: List[Dict[str, Any]],
        frame_id: int,
        timestamp: float,
        frame_shape: Tuple[int, int]
    ) -> HazardFeatures:
        """Extracts complete hazard feature set for a single worker at timestamp."""
        frame_h, frame_w = frame_shape

        # 1. PPE features
        ppe_info = worker.associated_ppe
        h_missing = 1.0 if ppe_info.get("helmet_missing", False) else 0.0
        v_missing = 1.0 if ppe_info.get("vest_missing", False) else 0.0
        ppe_count = float(h_missing + v_missing)

        h_conf = ppe_info.get("helmet_conf", 0.9)
        v_conf = ppe_info.get("vest_conf", 0.9)
        relevant_confs = []
        if h_missing:
            relevant_confs.append(h_conf)
        if v_missing:
            relevant_confs.append(v_conf)
        ppe_viol_conf = float(np.mean(relevant_confs)) if relevant_confs else 0.0

        # 2. Spatial features
        norm_dist = 1.0
        overlap = 0.0
        proximity_severity = 0.0
        workers_in_zone = 0.0

        if all_machines:
            # Find nearest machine
            min_dist = 1.0
            best_overlap = 0.0
            for m in all_machines:
                m_center = m["center"]
                d = self.calculate_normalized_distance(worker.current_position, m_center, frame_w, frame_h)
                if d < min_dist:
                    min_dist = d
                ov = self.calculate_box_overlap(worker.bbox, m["bbox"])
                if ov > best_overlap:
                    best_overlap = ov

            norm_dist = min_dist
            overlap = best_overlap

            # Proximity severity: non-linear escalation when entering danger zone
            # When distance <= danger_threshold, severity escalates rapidly towards 1.0
            if norm_dist < self.danger_threshold:
                proximity_severity = float(1.0 - (norm_dist / self.danger_threshold)**1.5)
            else:
                proximity_severity = max(0.0, float(0.15 * (1.0 - norm_dist)))

            # Count how many workers are in this machine's zone
            for other_w in all_workers:
                for m in all_machines:
                    od = self.calculate_normalized_distance(other_w.current_position, m["center"], frame_w, frame_h)
                    if od < self.danger_threshold:
                        workers_in_zone += 1.0
                        break

        # 3. Temporal features
        duration = worker.get_violation_duration(timestamp)
        consec_frames = float(worker.consecutive_violating_frames)
        total_frames = float(worker.frames_seen)
        persistence_ratio = worker.get_persistence_ratio()

        # 4. Scene features
        n_workers = float(len(all_workers))
        n_machines = float(len(all_machines))
        
        simul_viol = 0.0
        for w in all_workers:
            w_ppe = w.associated_ppe
            w_viol = w_ppe.get("helmet_missing", False) or w_ppe.get("vest_missing", False)
            if w_viol:
                simul_viol += 1.0
            elif w.associated_machine and w.associated_machine.get("distance_px", 9999) < (self.danger_threshold * math.sqrt(frame_w**2 + frame_h**2)):
                simul_viol += 1.0

        density = float(simul_viol / max(1.0, n_workers))

        # 5. Confidence features
        confs = worker.confidence_history if worker.confidence_history else [0.85]
        mean_conf = float(np.mean(confs))
        min_conf = float(np.min(confs))
        conf_var = float(np.var(confs)) if len(confs) > 1 else 0.0

        return HazardFeatures(
            track_id=worker.track_id,
            frame_id=frame_id,
            timestamp=timestamp,
            helmet_missing=h_missing,
            vest_missing=v_missing,
            ppe_violation_count=ppe_count,
            ppe_violation_confidence=ppe_viol_conf,
            normalized_worker_machine_dist=norm_dist,
            worker_machine_overlap=overlap,
            worker_count_machine_zone=workers_in_zone,
            machine_proximity_severity=proximity_severity,
            violation_duration=duration,
            consecutive_violating_frames=consec_frames,
            frames_since_first_detection=total_frames,
            persistence_ratio=persistence_ratio,
            num_workers=n_workers,
            num_machines=n_machines,
            simultaneous_violations=simul_viol,
            scene_hazard_density=density,
            mean_detection_confidence=mean_conf,
            min_relevant_confidence=min_conf,
            confidence_variance=conf_var
        )
