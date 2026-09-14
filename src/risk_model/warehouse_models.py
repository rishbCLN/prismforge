"""DamageMesh Warehouse Material Handling Risk Models (V0 to V3).
Progressive models evaluating material handling safety, damage risk, and intervention priority:
- V0: Heuristic Threshold Baseline
- V1: Spatial Context & Equipment Proximity Model
- V2: Temporal Persistence & Impact Filter
- V3: Learned Dual-Head Neural MLP with Gradient Attribution
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from src.features.warehouse_features import WarehouseKinematicFeatureExtractor


SEVERITY_CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
PRIORITY_CLASSES = ["P3_INFORMATIONAL", "P2_CORRECTIVE", "P1_IMMEDIATE"]


class WarehouseRiskOutput:
    """Standardized output container for warehouse risk predictions."""

    def __init__(
        self,
        risk_score: float,
        severity: str,
        intervention_priority: str,
        contributing_factors: List[str],
        attribution: Optional[Dict[str, float]] = None,
        model_version: str = "V0"
    ):
        self.risk_score = float(np.clip(risk_score, 0.0, 1.0))
        self.severity = severity if severity in SEVERITY_CLASSES else "LOW"
        self.intervention_priority = intervention_priority if intervention_priority in PRIORITY_CLASSES else "P3_INFORMATIONAL"
        self.contributing_factors = contributing_factors
        self.attribution = attribution or {}
        self.model_version = model_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 4),
            "severity": self.severity,
            "intervention_priority": self.intervention_priority,
            "contributing_factors": self.contributing_factors,
            "attribution": self.attribution,
            "model_version": self.model_version
        }


# ==============================================================================
# Model V0: Heuristic Threshold Baseline
# ==============================================================================
class WarehouseV0Baseline:
    """Naive static threshold-based risk model.
    Triggers alarms on instantaneous vertical/horizontal speeds without context or temporal persistence.
    """

    def __init__(self):
        self.version = "V0_Baseline"

    def predict(self, features: List[float]) -> WarehouseRiskOutput:
        # Indices:
        # 0: drop_velocity, 2: throw_horizontal_velocity, 5: drag_velocity, 7: jerk, 9: is_outside_staging
        drop_vel = features[0]
        throw_vel = features[2]
        drag_vel = features[5]
        jerk = features[7]
        outside_staging = features[9]

        factors = []
        score = 0.05

        if drop_vel > 0.25:
            score = max(score, 0.85)
            factors.append("HIGH_DROP_VELOCITY")
        if throw_vel > 0.35:
            score = max(score, 0.90)
            factors.append("HIGH_THROW_VELOCITY")
        if drag_vel > 0.20:
            score = max(score, 0.65)
            factors.append("FLOOR_DRAGGING_DETECTED")
        if jerk > 0.40:
            score = max(score, 0.70)
            factors.append("HIGH_JERK_MOTION")
        if outside_staging > 0.5:
            score = max(score, 0.50)
            factors.append("OUTSIDE_STAGING_ZONE")

        if score >= 0.80:
            severity = "CRITICAL"
            priority = "P1_IMMEDIATE"
        elif score >= 0.60:
            severity = "HIGH"
            priority = "P2_CORRECTIVE"
        elif score >= 0.30:
            severity = "MEDIUM"
            priority = "P2_CORRECTIVE"
        else:
            severity = "LOW"
            priority = "P3_INFORMATIONAL"

        attribution = {
            "drop_velocity": round(drop_vel, 3),
            "throw_velocity": round(throw_vel, 3),
            "drag_velocity": round(drag_vel, 3)
        }

        return WarehouseRiskOutput(score, severity, priority, factors, attribution, self.version)


# ==============================================================================
# Model V1: Spatial Context & Equipment Proximity Model
# ==============================================================================
class WarehouseV1Context:
    """Spatial context model incorporating trolley presence, floor distance, and operator proximity."""

    def __init__(self):
        self.version = "V1_Context"

    def predict(self, features: List[float]) -> WarehouseRiskOutput:
        # Indices:
        # 0: drop_velocity, 1: impact_decel, 2: throw_vel, 3: sep_rate, 4: drag_floor_contact,
        # 5: drag_vel, 6: trolley_iou, 7: jerk, 8: zone_dist, 9: outside_staging, 10: op_dist
        drop_vel = features[0]
        impact_decel = features[1]
        throw_vel = features[2]
        sep_rate = features[3]
        floor_contact = features[4]
        drag_vel = features[5]
        trolley_iou = features[6]
        jerk = features[7]
        outside_staging = features[9]
        op_dist = features[10]

        factors = []
        raw_score = 0.05

        # Suppress dragging alarm if trolley is actively supporting package
        effective_drag = drag_vel * (1.0 - trolley_iou * 0.9) * floor_contact

        # Airborne throw risk requires worker separation
        effective_throw = throw_vel * (0.5 + 0.5 * op_dist) * (0.6 + 0.4 * sep_rate)

        # Drop risk weighted by floor proximity and impact
        effective_drop = drop_vel * (0.5 + 0.5 * impact_decel)

        if effective_throw > 0.40:
            raw_score += effective_throw * 0.90
            factors.append("BALLISTIC_THROW_TRAJECTORY")
        elif effective_drop > 0.30:
            raw_score += effective_drop * 0.85
            factors.append("DROP_IMPACT_RISK")

        if effective_drag > 0.25:
            raw_score += effective_drag * 0.65
            factors.append("UNSUPPORTED_FLOOR_DRAG")

        if jerk > 0.40:
            raw_score += jerk * 0.45
            factors.append("ROUGH_HANDLING_JERK")

        if outside_staging > 0.5:
            raw_score += 0.30
            factors.append("STAGING_BOUNDARY_BREACH")

        score = float(np.clip(raw_score, 0.05, 0.98))

        if score >= 0.82:
            severity = "CRITICAL"
            priority = "P1_IMMEDIATE"
        elif score >= 0.62:
            severity = "HIGH"
            priority = "P1_IMMEDIATE" if "DROP" in str(factors) else "P2_CORRECTIVE"
        elif score >= 0.30:
            severity = "MEDIUM"
            priority = "P2_CORRECTIVE"
        else:
            severity = "LOW"
            priority = "P3_INFORMATIONAL"

        attribution = {
            "effective_throw": round(effective_throw, 3),
            "effective_drop": round(effective_drop, 3),
            "effective_drag": round(effective_drag, 3),
            "jerk": round(jerk, 3)
        }

        return WarehouseRiskOutput(score, severity, priority, factors, attribution, self.version)


# ==============================================================================
# Model V2: Temporal Persistence & Impact Filter
# ==============================================================================
class WarehouseV2Temporal:
    """Stateful temporal filter that tracks physical event progressions across time windows."""

    def __init__(self, persistence_frames: int = 4):
        self.version = "V2_Temporal"
        self.persistence_frames = persistence_frames
        self.drop_counter = 0
        self.drag_counter = 0
        self.throw_counter = 0

    def reset(self):
        self.drop_counter = 0
        self.drag_counter = 0
        self.throw_counter = 0

    def predict(self, features: List[float]) -> WarehouseRiskOutput:
        drop_vel = features[0]
        impact_decel = features[1]
        throw_vel = features[2]
        floor_contact = features[4]
        drag_vel = features[5]
        trolley_iou = features[6]
        jerk = features[7]
        outside_staging = features[9]
        stat_post_impact = features[14]

        # Temporal counters
        if drop_vel > 0.25 or impact_decel > 0.35:
            self.drop_counter += 1
        else:
            self.drop_counter = max(0, self.drop_counter - 1)

        if drag_vel > 0.25 and floor_contact > 0.6 and trolley_iou < 0.2:
            self.drag_counter += 1
        else:
            self.drag_counter = max(0, self.drag_counter - 1)

        if throw_vel > 0.40:
            self.throw_counter += 1
        else:
            self.throw_counter = max(0, self.throw_counter - 1)

        factors = []
        score = 0.05

        # Throw verification
        if self.throw_counter >= 2:
            score = max(score, 0.92)
            factors.append("VERIFIED_BALLISTIC_THROW")

        # Drop verification: sustained descent or impact followed by stationary carton
        if self.drop_counter >= 2 or (impact_decel > 0.35 and stat_post_impact > 0.5):
            score = max(score, 0.88 if stat_post_impact > 0.5 else 0.76)
            factors.append("VERIFIED_UNCONTROLLED_DROP")
            if stat_post_impact > 0.5:
                factors.append("POST_IMPACT_STATIONARY_CONFIRMATION")

        # Drag verification: sustained continuous dragging
        if self.drag_counter >= self.persistence_frames:
            score = max(score, 0.74)
            factors.append("SUSTAINED_FLOOR_DRAGGING")

        if jerk > 0.50 and self.throw_counter == 0:
            score = max(score, 0.68)
            factors.append("REPEATED_HIGH_JERK_HANDLING")

        if outside_staging > 0.5:
            score = max(score, 0.52)
            factors.append("STAGING_BOUNDARY_BREACH")

        if score >= 0.80:
            severity = "CRITICAL"
            priority = "P1_IMMEDIATE"
        elif score >= 0.60:
            severity = "HIGH"
            priority = "P1_IMMEDIATE" if "DROP" in str(factors) else "P2_CORRECTIVE"
        elif score >= 0.30:
            severity = "MEDIUM"
            priority = "P2_CORRECTIVE"
        else:
            severity = "LOW"
            priority = "P3_INFORMATIONAL"

        attribution = {
            "drop_persistence": round(self.drop_counter / 5.0, 2),
            "drag_persistence": round(self.drag_counter / float(self.persistence_frames), 2),
            "throw_persistence": round(self.throw_counter / 3.0, 2)
        }

        return WarehouseRiskOutput(score, severity, priority, factors, attribution, self.version)


# ==============================================================================
# Model V3: Learned Dual-Head Neural MLP with Gradient Attribution
# ==============================================================================
class WarehouseRiskMLP(nn.Module):
    """Dual-Head PyTorch Neural Network for Warehouse Risk & Severity Prediction."""

    def __init__(self, in_features: int = 16):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.risk_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.severity_head = nn.Sequential(
            nn.Linear(32, 4) # 4 classes: LOW, MEDIUM, HIGH, CRITICAL
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        risk = self.risk_head(feat)
        severity_logits = self.severity_head(feat)
        return risk, severity_logits


class WarehouseV3Learned:
    """Production wrapper for learned V3 Warehouse Risk MLP."""

    def __init__(self, model_path: Optional[str] = "models/v3/warehouse_risk_mlp.pt"):
        self.version = "V3_Learned"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WarehouseRiskMLP(in_features=16).to(self.device)
        self.model.eval()
        self.is_trained = False

        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.is_trained = True
                print(f"Loaded trained Warehouse V3 model from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load V3 model from {model_path}: {e}")

    def predict(self, features: List[float]) -> WarehouseRiskOutput:
        x_tensor = torch.tensor([features], dtype=torch.float32, device=self.device, requires_grad=True)

        self.model.eval()
        risk_score_t, sev_logits_t = self.model(x_tensor)

        risk_score = float(risk_score_t.item())
        sev_idx = int(torch.argmax(sev_logits_t, dim=-1).item())
        severity = SEVERITY_CLASSES[sev_idx]

        # Compute gradient attribution
        risk_score_t.backward()
        grads = x_tensor.grad.detach().cpu().numpy()[0]
        feature_names = WarehouseKinematicFeatureExtractor.FEATURE_NAMES

        # Feature attribution = input * abs(gradient)
        attribution = {}
        for name, feat_val, g in zip(feature_names, features, grads):
            attribution[name] = round(float(abs(feat_val * g)), 4)

        # Sort top factors
        sorted_factors = sorted(attribution.items(), key=lambda x: x[1], reverse=True)
        top_factors = [k.upper() for k, v in sorted_factors[:3] if v > 0.01]
        if not top_factors:
            top_factors = ["NORMAL_COMPLIANT_HANDLING"] if severity == "LOW" else ["ELEVATED_MOTION_ENERGY"]

        # Derive Priority
        if severity == "CRITICAL" or risk_score >= 0.80:
            priority = "P1_IMMEDIATE"
        elif severity == "HIGH" or risk_score >= 0.55:
            priority = "P1_IMMEDIATE" if "DROP" in str(top_factors) or "THROW" in str(top_factors) else "P2_CORRECTIVE"
        elif severity == "MEDIUM" or risk_score >= 0.30:
            priority = "P2_CORRECTIVE"
        else:
            priority = "P3_INFORMATIONAL"

        return WarehouseRiskOutput(risk_score, severity, priority, top_factors, attribution, self.version)
