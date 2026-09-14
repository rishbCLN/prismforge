"""PyTorch Dataset & Dual-Head Risk Architecture for Image Hazard Features."""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class ImageHazardDataset(Dataset):
    """Dataset for training hazard risk networks on spatial image features."""

    SEVERITY_MAP = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def __init__(self, samples: List[Dict[str, Any]], means: Optional[np.ndarray] = None, stds: Optional[np.ndarray] = None):
        self.X_raw = []
        self.y_sev = []
        self.y_risk = []

        for s in samples:
            vec = s.get("feature_vector", [])
            self.X_raw.append(vec)
            self.y_sev.append(self.SEVERITY_MAP.get(s.get("severity", "NONE"), 0))
            self.y_risk.append(float(s.get("risk_score", 0.0)))

        self.X_raw = np.array(self.X_raw, dtype=np.float32)
        if len(self.X_raw) == 0:
            self.X = torch.empty((0, 12), dtype=torch.float32)
            self.means = np.zeros(12, dtype=np.float32)
            self.stds = np.ones(12, dtype=np.float32)
            return

        # Compute or use normalization
        if means is None:
            self.means = np.mean(self.X_raw, axis=0)
            self.stds = np.std(self.X_raw, axis=0) + 1e-6
        else:
            self.means = means
            self.stds = stds

        norm_X = (self.X_raw - self.means) / self.stds
        self.X = torch.tensor(norm_X, dtype=torch.float32)
        self.y_sev = torch.tensor(np.array(self.y_sev), dtype=torch.long)
        self.y_risk = torch.tensor(np.array(self.y_risk), dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_sev[idx], self.y_risk[idx]


class ImageHazardMLP(nn.Module):
    """Dual-head neural network predicting continuous risk score and discrete severity."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )

        # Head 1: Multi-class discrete severity classification (NONE, LOW, MEDIUM, HIGH)
        self.severity_head = nn.Linear(32, num_classes)

        # Head 2: Continuous calibrated risk score (0.0 to 1.0)
        self.risk_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        sev_logits = self.severity_head(features)
        risk_score = self.risk_head(features)
        return sev_logits, risk_score
