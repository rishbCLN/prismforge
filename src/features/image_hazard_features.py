"""Spatial & Contextual Hazard Feature Extractor for Construction Images.
Converts raw bounding boxes of workers, PPE (helmets, vests), and heavy equipment
into structured physical/spatial feature vectors for training risk models.
"""
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ImageWorkerHazardFeature:
    image_id: str
    image_path: str
    worker_box: List[float]  # [x1, y1, x2, y2]
    # PPE features
    helmet_missing: float    # 1.0 = missing, 0.0 = worn
    vest_missing: float      # 1.0 = missing, 0.0 = worn
    ppe_violation_count: float
    # Heavy Machinery Proximity
    has_machinery_in_scene: float
    nearest_machine_dist: float  # Normalized Euclidean distance (0.0 to 1.0)
    machine_overlap_iou: float   # Direct bounding box collision
    in_critical_danger_zone: float  # Worker < 0.15 normalized distance to machine
    in_warning_proximity_zone: float  # Worker < 0.30 normalized distance to machine
    # Scene Context
    num_workers_in_scene: float
    num_machines_in_scene: float
    worker_crowding_factor: float
    worker_relative_size: float  # Box area / Image area (indicates proximity to camera)
    simultaneous_violations: float
    # Derived Hazard Targets
    ground_truth_risk_score: float
    ground_truth_severity: str  # NONE, LOW, MEDIUM, HIGH
    ground_truth_priority: str  # P0_HALT, P1_EVACUATE, P2_ALERT, P3_NORMAL

    def to_feature_vector(self) -> List[float]:
        """Returns ordered 12-dimensional numerical vector for PyTorch models."""
        return [
            self.helmet_missing,
            self.vest_missing,
            self.ppe_violation_count,
            self.has_machinery_in_scene,
            self.nearest_machine_dist,
            self.machine_overlap_iou,
            self.in_critical_danger_zone,
            self.in_warning_proximity_zone,
            self.num_workers_in_scene,
            self.num_machines_in_scene,
            self.worker_crowding_factor,
            self.worker_relative_size
        ]


class ImageHazardFeatureExtractor:
    """Extracts spatial hazard vectors from image bounding boxes."""

    @staticmethod
    def _box_center(b: List[float]) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @staticmethod
    def _box_area(b: List[float]) -> float:
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    @staticmethod
    def _box_iou(b1: List[float], b2: List[float]) -> float:
        x_left = max(b1[0], b2[0])
        y_top = max(b1[1], b2[1])
        x_right = min(b1[2], b2[2])
        y_bottom = min(b1[3], b2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        a1 = ImageHazardFeatureExtractor._box_area(b1)
        a2 = ImageHazardFeatureExtractor._box_area(b2)
        union = a1 + a2 - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _box_distance(b1: List[float], b2: List[float]) -> float:
        c1 = ImageHazardFeatureExtractor._box_center(b1)
        c2 = ImageHazardFeatureExtractor._box_center(b2)
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def extract_features_from_image(self, image_data: Dict[str, Any]) -> List[ImageWorkerHazardFeature]:
        """Processes one image record and extracts features for every detected worker."""
        objects = image_data.get("objects", [])
        image_id = image_data.get("image_id", "img")
        image_path = image_data.get("image_path", "")

        persons = [o for o in objects if o["class_name"] == "person"]
        helmets = [o for o in objects if o["class_name"] == "helmet"]
        no_helmets = [o for o in objects if o["class_name"] == "no_helmet"]
        vests = [o for o in objects if o["class_name"] == "vest"]
        no_vests = [o for o in objects if o["class_name"] == "no_vest"]
        machines = [o for o in objects if o["class_name"] == "machinery"]

        num_workers = len(persons)
        num_machines = len(machines)
        features: List[ImageWorkerHazardFeature] = []

        for p in persons:
            p_box = [p["x1"], p["y1"], p["x2"], p["y2"]]
            pw = p["x2"] - p["x1"]
            ph = p["y2"] - p["y1"]
            p_area = pw * ph

            # Head region: upper 35% of person bounding box
            head_region = [p["x1"], p["y1"], p["x2"], p["y1"] + 0.35 * ph]
            # Torso region: middle 50% of person bounding box
            torso_region = [p["x1"], p["y1"] + 0.25 * ph, p["x2"], p["y1"] + 0.75 * ph]

            # Check helmet match
            has_helmet = False
            for h in helmets:
                h_box = [h["x1"], h["y1"], h["x2"], h["y2"]]
                if self._box_iou(head_region, h_box) > 0.05 or self._box_distance(head_region, h_box) < 0.15:
                    has_helmet = True
                    break

            # Check explicit no-helmet annotations
            for nh in no_helmets:
                nh_box = [nh["x1"], nh["y1"], nh["x2"], nh["y2"]]
                if self._box_distance(head_region, nh_box) < 0.15:
                    has_helmet = False
                    break

            # Check vest match
            has_vest = False
            for v in vests:
                v_box = [v["x1"], v["y1"], v["x2"], v["y2"]]
                if self._box_iou(torso_region, v_box) > 0.05 or self._box_distance(torso_region, v_box) < 0.15:
                    has_vest = True
                    break

            for nv in no_vests:
                nv_box = [nv["x1"], nv["y1"], nv["x2"], nv["y2"]]
                if self._box_distance(torso_region, nv_box) < 0.15:
                    has_vest = False
                    break

            helmet_missing = 0.0 if has_helmet else 1.0
            vest_missing = 0.0 if has_vest else 1.0
            ppe_violations = helmet_missing + vest_missing

            # Proximity to heavy equipment
            if machines:
                min_dist = min(self._box_distance(p_box, [m["x1"], m["y1"], m["x2"], m["y2"]]) for m in machines)
                max_iou = max(self._box_iou(p_box, [m["x1"], m["y1"], m["x2"], m["y2"]]) for m in machines)
                has_mach = 1.0
            else:
                min_dist = 1.0
                max_iou = 0.0
                has_mach = 0.0

            crit_danger = 1.0 if (has_mach and min_dist < 0.18) or max_iou > 0.01 else 0.0
            warn_danger = 1.0 if (has_mach and min_dist < 0.35) else 0.0

            # Crowding
            crowding = max(0.0, float(num_workers - 1) / 10.0)

            # Ground truth risk calculation (OSHA / ANSI compliant rules)
            # Critical danger zone near machine: base 0.70
            # Warning proximity: base 0.35
            # Safe distance: base 0.05
            if crit_danger > 0:
                base_risk = 0.75 + (0.15 * (helmet_missing + vest_missing) / 2.0)
            elif warn_danger > 0:
                base_risk = 0.40 + (0.25 * (helmet_missing + vest_missing) / 2.0)
            else:
                base_risk = 0.05 + (0.25 * (helmet_missing + vest_missing) / 2.0)

            risk_score = min(1.0, max(0.0, base_risk))

            if risk_score >= 0.70:
                sev = "HIGH"
                prio = "P0_IMMEDIATE_SHUTDOWN" if max_iou > 0.0 else "P1_EVACUATE_ZONE"
            elif risk_score >= 0.40:
                sev = "MEDIUM"
                prio = "P2_SUPERVISOR_ALERT"
            elif risk_score >= 0.15:
                sev = "LOW"
                prio = "P3_LOG_INFO"
            else:
                sev = "NONE"
                prio = "P3_LOG_INFO"

            features.append(ImageWorkerHazardFeature(
                image_id=image_id,
                image_path=image_path,
                worker_box=p_box,
                helmet_missing=helmet_missing,
                vest_missing=vest_missing,
                ppe_violation_count=ppe_violations,
                has_machinery_in_scene=has_mach,
                nearest_machine_dist=min_dist,
                machine_overlap_iou=max_iou,
                in_critical_danger_zone=crit_danger,
                in_warning_proximity_zone=warn_danger,
                num_workers_in_scene=float(num_workers),
                num_machines_in_scene=float(num_machines),
                worker_crowding_factor=crowding,
                worker_relative_size=min(1.0, p_area),
                simultaneous_violations=float(ppe_violations + crit_danger),
                ground_truth_risk_score=round(risk_score, 4),
                ground_truth_severity=sev,
                ground_truth_priority=prio
            ))

        return features

    def process_manifest(self, manifest_data: Dict[str, Any]) -> List[ImageWorkerHazardFeature]:
        """Processes entire ingested manifest and returns all extracted features."""
        all_feats = []
        for img in manifest_data.get("images", []):
            feats = self.extract_features_from_image(img)
            all_feats.extend(feats)
        return all_feats
