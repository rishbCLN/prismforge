"""HazardMesh V3 — Learned Risk Model (PyTorch).
Replaces hand-tuned heuristics with a neural network trained on structured spatio-temporal features.
Predicts both continuous hazard risk score and discretized severity classes.
"""
import os
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from src.features.feature_extractor import HazardFeatures
from src.risk_model.base import BaseRiskModel, RiskOutput


class RiskMLP(nn.Module):
    """Compact, interpretable Multi-Layer Perceptron for Hazard Risk Reasoning."""

    def __init__(self, input_dim: int = 19):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Head 1: 4-class severity logits (NONE=0, LOW=1, MEDIUM=2, HIGH=3)
        self.severity_head = nn.Linear(16, 4)
        # Head 2: continuous risk score in [0, 1]
        self.risk_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        severity_logits = self.severity_head(feat)
        risk_score = self.risk_head(feat)
        return severity_logits, risk_score


class V3LearnedRiskModel(BaseRiskModel):
    """V3: Learned PyTorch Risk Model."""

    SEVERITY_MAP = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}

    def __init__(self, model_path: Optional[str] = "models/v3/hazard_risk_mlp.pt"):
        super().__init__(version_name="v3_learned")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RiskMLP(input_dim=len(HazardFeatures.FEATURE_NAMES)).to(self.device)
        self.model_path = model_path
        self.feature_means = np.zeros(len(HazardFeatures.FEATURE_NAMES), dtype=np.float32)
        self.feature_stds = np.ones(len(HazardFeatures.FEATURE_NAMES), dtype=np.float32)

        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)
        else:
            # Initialize with sensible weights if checkpoint not yet trained
            self.model.eval()

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        if "feature_means" in checkpoint:
            self.feature_means = np.array(checkpoint["feature_means"], dtype=np.float32)
            self.feature_stds = np.array(checkpoint["feature_stds"], dtype=np.float32)
        self.model.eval()
        print(f"Loaded V3 PyTorch model checkpoint from {path}")

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "state_dict": self.model.state_dict(),
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist(),
            "feature_names": HazardFeatures.FEATURE_NAMES,
            "metadata": metadata or {}
        }
        torch.save(data, path)
        print(f"Saved V3 PyTorch checkpoint to {path}")

    def predict(self, features: HazardFeatures) -> RiskOutput:
        self.model.eval()
        vec = features.to_vector()

        # Normalize features
        norm_vec = (vec - self.feature_means) / (self.feature_stds + 1e-6)
        x_tensor = torch.tensor(norm_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_tensor.requires_grad = True

        sev_logits, risk_pred = self.model(x_tensor)
        risk_score = float(risk_pred.item())
        pred_class_idx = int(torch.argmax(sev_logits, dim=1).item())
        severity = self.SEVERITY_MAP.get(pred_class_idx, "NONE")
        priority = RiskOutput.severity_to_priority(severity)

        # Feature Attribution via input gradients for explainability
        risk_pred.backward()
        grads = x_tensor.grad.detach().cpu().numpy()[0]
        attributions = np.abs(grads * norm_vec)
        top_indices = np.argsort(attributions)[::-1][:4]

        primary_factors = []
        breakdown = {}
        for idx in top_indices:
            fname = HazardFeatures.FEATURE_NAMES[idx]
            val = float(vec[idx])
            breakdown[fname] = round(val, 3)
            # Human readable explanation
            if fname == "machine_proximity_severity" and val > 0.3:
                primary_factors.append(f"+ Heavy machinery proximity ({val:.2f})")
            elif fname == "helmet_missing" and val > 0.5:
                primary_factors.append("+ Missing safety helmet")
            elif fname == "vest_missing" and val > 0.5:
                primary_factors.append("+ Missing high-visibility vest")
            elif fname == "violation_duration" and val > 1.0:
                primary_factors.append(f"+ Persistent exposure ({val:.1f}s)")
            elif fname == "worker_count_machine_zone" and val > 1.0:
                primary_factors.append(f"+ Multiple workers in danger perimeter ({int(val)})")
            elif fname == "scene_hazard_density" and val > 0.3:
                primary_factors.append(f"+ High site hazard density ({val:.2f})")

        if not primary_factors:
            if severity == "NONE":
                primary_factors.append("Safe conditions: Full PPE and clear perimeter")
            else:
                primary_factors.append("Compound multi-feature risk interaction")

        return RiskOutput(
            risk_score=risk_score,
            severity=severity,
            priority=priority,
            primary_factors=primary_factors,
            factor_breakdown=breakdown
        )
