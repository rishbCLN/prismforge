"""HazardMesh Failure Analysis and Failure-Driven Improvement Module.
Identifies prediction errors, clusters failures by meaningful physical/contextual attributes,
and tracks before/after performance across model versions.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
from src.features.feature_extractor import HazardFeatures


@dataclass
class FailureRecord:
    sample_id: str
    clip_id: str
    frame_id: int
    true_severity: str
    pred_severity: str
    risk_score: float
    features: Dict[str, float]
    penalty: float
    assigned_cluster: str


class FailureAnalyzer:
    """Clusters and diagnoses failure modes across model versions."""

    CLUSTER_DEFINITIONS = {
        "cluster_a_transient_proximity": {
            "name": "Transient Machinery Proximity",
            "description": "High proximity to machinery but brief exposure duration (< 0.5s). Models without temporal filtering trigger false alarms.",
            "target_fix": "V2 temporal persistence and consecutive-frame filtering."
        },
        "cluster_b_detector_noise": {
            "name": "Detector Confidence & Flicker",
            "description": "Spurious detection or low confidence (< 0.60) causing erratic risk escalation.",
            "target_fix": "Confidence weighting and temporal smoothing."
        },
        "cluster_c_safe_zone_ppe": {
            "name": "Isolated Safe-Zone PPE Violation",
            "description": "Missing PPE in designated safe walkway far from equipment. Naive baselines over-penalize this as high danger.",
            "target_fix": "V1 spatial context and normalized machinery proximity weighting."
        },
        "cluster_d_compound_multi_worker": {
            "name": "Compound Multi-Worker Danger Zone",
            "description": "Multiple workers in close proximity to active machinery with non-linear compounding risk.",
            "target_fix": "V3 learned PyTorch MLP with non-linear feature interaction."
        }
    }

    @classmethod
    def assign_cluster(cls, feat: Dict[str, float], y_true: str, y_pred: str) -> str:
        """Heuristically assigns a failure to a specific contextual failure cluster."""
        duration = feat.get("violation_duration", 0.0)
        proximity = feat.get("machine_proximity_severity", 0.0)
        conf = feat.get("mean_detection_confidence", 1.0)
        workers_in_zone = feat.get("worker_count_machine_zone", 0.0)
        norm_dist = feat.get("normalized_worker_machine_dist", 1.0)

        # 1. Safe zone PPE: Missing PPE but far from machinery (norm_dist > 0.35) and pred is HIGH/MEDIUM while true is LOW/NONE
        if norm_dist > 0.35 and proximity < 0.20 and (feat.get("helmet_missing", 0.0) > 0.5 or feat.get("vest_missing", 0.0) > 0.5):
            return "cluster_c_safe_zone_ppe"

        # 2. Transient proximity: High proximity but short duration (< 0.6s)
        if proximity > 0.4 and duration < 0.6 and y_pred in ("HIGH", "MEDIUM") and y_true in ("LOW", "NONE"):
            return "cluster_a_transient_proximity"

        # 3. Detector noise: Low confidence or high variance
        if conf < 0.60 or feat.get("confidence_variance", 0.0) > 0.05:
            return "cluster_b_detector_noise"

        # 4. Multi-worker compound hazard
        if workers_in_zone > 1.0 and proximity > 0.3:
            return "cluster_d_compound_multi_worker"

        # Fallback default cluster
        return "cluster_a_transient_proximity"

    @classmethod
    def analyze_model_failures(
        cls,
        samples: List[Dict[str, Any]],
        y_true: List[str],
        y_pred: List[str],
        risk_scores: List[float],
        features_list: List[HazardFeatures],
        cost_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Identifies misclassifications and clusters them into structured failure profiles."""
        failures: List[FailureRecord] = []
        cluster_counts = {k: 0 for k in cls.CLUSTER_DEFINITIONS.keys()}
        cluster_samples: Dict[str, List[Dict[str, Any]]] = {k: [] for k in cls.CLUSTER_DEFINITIONS.keys()}

        for i, (sample, yt, yp, score, feat) in enumerate(zip(samples, y_true, y_pred, risk_scores, features_list)):
            if yt != yp:
                f_dict = feat.to_dict()
                cluster_id = cls.assign_cluster(f_dict, yt, yp)
                cluster_counts[cluster_id] += 1

                # Calculate penalty
                penalty = 1.0
                if yt == "HIGH":
                    penalty = 5.0
                elif yt == "MEDIUM":
                    penalty = 3.0
                elif yt == "LOW":
                    penalty = 2.0

                record = FailureRecord(
                    sample_id=f"sample_{i}",
                    clip_id=sample.get("clip_id", f"clip_{i}"),
                    frame_id=sample.get("frame_id", i),
                    true_severity=yt,
                    pred_severity=yp,
                    risk_score=score,
                    features=f_dict,
                    penalty=penalty,
                    assigned_cluster=cluster_id
                )
                failures.append(record)
                if len(cluster_samples[cluster_id]) < 5:
                    cluster_samples[cluster_id].append(asdict(record))

        # Summarize each cluster
        total_failures = len(failures)
        clusters_summary = []
        for cid, defn in cls.CLUSTER_DEFINITIONS.items():
            count = cluster_counts[cid]
            failure_rate = round(count / max(1, total_failures), 4)
            clusters_summary.append({
                "cluster_id": cid,
                "cluster_name": defn["name"],
                "description": defn["description"],
                "target_fix": defn["target_fix"],
                "sample_count": count,
                "failure_rate_within_errors": failure_rate,
                "example_samples": cluster_samples[cid]
            })

        return {
            "total_failures": total_failures,
            "total_samples": len(y_true),
            "overall_error_rate": round(total_failures / max(1, len(y_true)), 4),
            "clusters": clusters_summary
        }

    @classmethod
    def compare_runs(cls, old_analysis: Dict[str, Any], new_analysis: Dict[str, Any], old_version: str, new_version: str) -> Dict[str, Any]:
        """Compares failure distributions between two versions to prove targeted fix."""
        old_clusters = {c["cluster_id"]: c for c in old_analysis["clusters"]}
        new_clusters = {c["cluster_id"]: c for c in new_analysis["clusters"]}

        comparison = []
        for cid, defn in cls.CLUSTER_DEFINITIONS.items():
            old_count = old_clusters.get(cid, {}).get("sample_count", 0)
            new_count = new_clusters.get(cid, {}).get("sample_count", 0)
            diff = new_count - old_count
            pct_change = round((diff / max(1, old_count)) * 100.0, 1)

            status = "IMPROVED" if diff < 0 else ("REGRESSED" if diff > 0 else "UNCHANGED")
            comparison.append({
                "cluster_id": cid,
                "name": defn["name"],
                f"{old_version}_count": old_count,
                f"{new_version}_count": new_count,
                "reduction_count": -diff,
                "percentage_change": pct_change,
                "status": status,
                "rationale": defn["target_fix"]
            })

        return {
            "comparison": comparison,
            "old_total_errors": old_analysis["total_failures"],
            "new_total_errors": new_analysis["total_failures"],
            "error_reduction": old_analysis["total_failures"] - new_analysis["total_failures"]
        }
