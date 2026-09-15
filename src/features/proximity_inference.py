"""Inference Engine for Large Vehicle Proximity Risk.

ProximityInferenceEngine takes raw detected bboxes (workers + vehicles)
and returns per-worker proximity risk with TTA and temperature scaling.

Usage:
    engine = ProximityInferenceEngine("models/proximity_reasoner.pt")
    results = engine.score_scene(
        worker_bboxes=[[cx, cy, w, h], ...],   # normalized
        vehicle_bboxes=[{"bbox_norm": [...], "class": "truck"}, ...]
    )
    for r in results:
        print(r["worker_id"], r["alert_level"], r["risk_score"])

Alert Levels:
    🟢 SAFE         — worker is in monitored but non-critical zone
    🟡 SUPERVISED   — worker should be monitored actively
    🔴 DANGER       — worker is dangerously close to vehicle
    ⚫ OUT_OF_ZONE  — worker is too far from vehicle (unmonitored)
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

from src.risk_model.large_vehicle_proximity_net import (
    LargeVehicleProximityNet, EMAModel, load_proximity_model, FEATURE_DIM
)
from src.features.proximity_features import (
    extract_proximity_feature, derive_proximity_label,
    DANGER_THRESH, SUPERVISED_THRESH, TOO_FAR_THRESH, VEHICLE_CLASS_MAP,
    PROXIMITY_CLASS_NAMES
)


# ---------------------------------------------------------------------------
# Alert level constants
# ---------------------------------------------------------------------------

ALERT_SAFE       = "SAFE"        # proximity class 0
ALERT_SUPERVISED = "SUPERVISED"  # proximity class 1
ALERT_DANGER     = "DANGER"      # proximity class 2
ALERT_OUT_OF_ZONE = "OUT_OF_ZONE"  # proximity class 3

ALERT_EMOJIS = {
    ALERT_SAFE: "🟢",
    ALERT_SUPERVISED: "🟡",
    ALERT_DANGER: "🔴",
    ALERT_OUT_OF_ZONE: "⚫",
}

# Temperature for softmax scaling (>1 → more uncertain / softer)
TEMPERATURE = 1.3


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class ProximityInferenceEngine:
    """Worker–vehicle proximity scoring engine with TTA and temperature scaling.

    Args:
        model_path: Path to saved LargeVehicleProximityNet checkpoint (.pt)
        device: 'cuda', 'cpu', or 'auto'
        tta_passes: Number of Test-Time Augmentation forward passes (default 5)
        temperature: Softmax temperature for uncertainty calibration (default 1.3)
        use_ema: Whether to load EMA weights if available
    """

    def __init__(
        self,
        model_path: str = "models/proximity_reasoner.pt",
        device: str = "auto",
        tta_passes: int = 5,
        temperature: float = TEMPERATURE,
        use_ema: bool = True,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tta_passes = tta_passes
        self.temperature = temperature

        # Try EMA weights first
        if use_ema:
            ema_path = model_path.replace(".pt", "_ema.pt")
            import os
            if os.path.isfile(ema_path):
                model_path = ema_path

        try:
            self.model = load_proximity_model(model_path, device=device, strict=False)
            self.model.eval()
            self._model_loaded = True
        except Exception as e:
            print(f"[ProximityInferenceEngine] Warning: could not load model from {model_path}: {e}")
            print("  Running in geometry-only fallback mode.")
            self.model = None
            self._model_loaded = False

    def _tta_forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run TTA: average predictions over multiple noisy forward passes."""
        scores_acc = torch.zeros(feat.size(0), device=self.device)
        class_logits_acc = torch.zeros(feat.size(0), 4, device=self.device)
        vcls_acc = torch.zeros(feat.size(0), device=self.device)

        for i in range(self.tta_passes):
            if i == 0:
                noisy_feat = feat
            else:
                noise = torch.randn_like(feat) * 0.012
                noisy_feat = (feat + noise).clamp(0.0, 1.0)
            with torch.no_grad():
                out = self.model(noisy_feat)
            scores_acc     += out["proximity_score"]
            class_logits_acc += out["proximity_class"]
            vcls_acc        += out["vehicle_class"]

        return {
            "proximity_score": scores_acc / self.tta_passes,
            "proximity_class": class_logits_acc / self.tta_passes,
            "vehicle_class": vcls_acc / self.tta_passes,
        }

    def score_pair(
        self,
        worker_bbox: List[float],
        vehicle_bbox: List[float],
        vehicle_class: str = "truck",
    ) -> Dict[str, Any]:
        """Score a single (worker, vehicle) pair.

        Returns:
            dict with keys: risk_score, proximity_class_id, proximity_class,
                            alert_level, alert_emoji, vehicle_class_pred,
                            distance, confidence, is_danger
        """
        feat = extract_proximity_feature(worker_bbox, vehicle_bbox, vehicle_class)

        if self._model_loaded:
            feat_tensor = torch.tensor([feat], dtype=torch.float32, device=self.device)
            out = self._tta_forward(feat_tensor)

            # Temperature-scaled softmax
            class_logits = out["proximity_class"][0] / self.temperature
            class_probs   = F.softmax(class_logits, dim=-1)
            pred_cls_id   = int(class_probs.argmax().item())
            confidence    = float(class_probs.max().item())
            risk_score    = float(out["proximity_score"][0].item())
            v_cls_pred    = int(out["vehicle_class"][0].sigmoid().item() > 0.5)
        else:
            # Geometry fallback
            pred_cls_id, risk_score = derive_proximity_label(worker_bbox, vehicle_bbox)
            confidence = 0.70
            v_cls_pred = VEHICLE_CLASS_MAP.get(vehicle_class.lower(), 1)

        class_name = PROXIMITY_CLASS_NAMES[pred_cls_id]
        alert_level = class_name if pred_cls_id != 3 else "OUT_OF_ZONE"

        dist = math.sqrt(
            (worker_bbox[0] - vehicle_bbox[0]) ** 2 +
            (worker_bbox[1] - vehicle_bbox[1]) ** 2
        )

        return {
            "risk_score":          round(risk_score, 4),
            "proximity_class_id":  pred_cls_id,
            "proximity_class":     class_name,
            "alert_level":         alert_level,
            "alert_emoji":         ALERT_EMOJIS.get(alert_level, "❓"),
            "vehicle_class_pred":  "tractor" if v_cls_pred == 0 else "truck",
            "distance":            round(dist, 4),
            "confidence":          round(confidence, 4),
            "is_danger":           (pred_cls_id == 2),
        }

    def score_scene(
        self,
        worker_bboxes: List[List[float]],
        vehicle_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score all workers against all vehicles in a scene.

        Args:
            worker_bboxes: List of [cx, cy, w, h] normalized bboxes
            vehicle_detections: List of {"bbox_norm": [cx,cy,w,h], "class": str}

        Returns:
            List of per-worker result dicts, each containing:
              worker_id, nearest_vehicle_id, risk_score, alert_level,
              alert_emoji, distance, is_danger, all_vehicle_scores
        """
        if not worker_bboxes or not vehicle_detections:
            return []

        results = []
        for w_id, w_bbox in enumerate(worker_bboxes):
            worker_result = {
                "worker_id": w_id,
                "worker_bbox": w_bbox,
                "all_vehicle_scores": [],
            }

            # Score against every vehicle, find nearest/most dangerous
            best_score = None
            for v_id, v_det in enumerate(vehicle_detections):
                v_bbox  = v_det.get("bbox_norm", v_det.get("bbox", [0.5, 0.5, 0.5, 0.5]))
                v_class = v_det.get("class", "truck")

                pair_result = self.score_pair(w_bbox, v_bbox, v_class)
                pair_result["vehicle_id"] = v_id
                worker_result["all_vehicle_scores"].append(pair_result)

                # Select worst-case vehicle (highest risk score)
                if best_score is None or pair_result["risk_score"] > best_score["risk_score"]:
                    best_score = pair_result

            # Summarize: use the worst-case vehicle score
            if best_score is not None:
                worker_result.update({
                    "nearest_vehicle_id": best_score["vehicle_id"],
                    "risk_score":         best_score["risk_score"],
                    "proximity_class_id": best_score["proximity_class_id"],
                    "proximity_class":    best_score["proximity_class"],
                    "alert_level":        best_score["alert_level"],
                    "alert_emoji":        best_score["alert_emoji"],
                    "distance":           best_score["distance"],
                    "confidence":         best_score["confidence"],
                    "is_danger":          best_score["is_danger"],
                    "vehicle_class":      best_score["vehicle_class_pred"],
                })
            results.append(worker_result)

        return results

    def format_scene_report(self, scene_results: List[Dict[str, Any]]) -> str:
        """Format scene results as a human-readable report string."""
        if not scene_results:
            return "No workers detected in scene."
        lines = ["=== Proximity Risk Report ==="]
        for r in scene_results:
            emoji = r.get("alert_emoji", "❓")
            lines.append(
                f"{emoji} Worker {r['worker_id']:2d}: "
                f"{r.get('alert_level', 'UNKNOWN'):12s} | "
                f"risk={r.get('risk_score', 0.0):.3f} | "
                f"dist={r.get('distance', 0.0):.3f} | "
                f"vehicle={r.get('vehicle_class', 'unknown')}"
            )
        danger_count = sum(1 for r in scene_results if r.get("is_danger"))
        lines.append(f"--- {danger_count}/{len(scene_results)} workers in DANGER zone ---")
        return "\n".join(lines)
