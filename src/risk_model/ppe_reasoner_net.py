"""PPEReasonerNet — Custom Neural Network for Processing YOLO Detections.
Takes raw bounding box coordinates, IoUs, and confidence scores from YOLO
and predicts true vest/hardhat wearing compliance, occlusion resistance, and risk level.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple


class PPEReasonerNet(nn.Module):
    """Deep neural network that processes YOLO detections into safety compliance decisions."""

    def __init__(self, input_dim: int = 14, hidden_dim: int = 64):
        super().__init__()

        # Feature processing backbone
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1)
        )

        # Head 1: Vest True Compliance Score (0.0 = Not Worn, 1.0 = Properly Worn)
        self.vest_compliance_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Head 2: Hardhat True Compliance Score (0.0 = Not Worn, 1.0 = Properly Worn)
        self.hardhat_compliance_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Head 3: Multi-class Violation Type:
        # 0: COMPLIANT (Both worn)
        # 1: MISSING_VEST_ONLY
        # 2: MISSING_HARDHAT_ONLY
        # 3: CRITICAL_NO_PPE (Both missing)
        self.violation_classifier = nn.Linear(32, 4)

        # Head 4: Continuous Site Risk Score (0.0 to 1.0)
        self.risk_score_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Input x: [batch_size, 14] spatial and confidence features from YOLO detections."""
        latent = self.encoder(x)
        return {
            "vest_compliance": self.vest_compliance_head(latent),
            "hardhat_compliance": self.hardhat_compliance_head(latent),
            "violation_logits": self.violation_classifier(latent),
            "risk_score": self.risk_score_head(latent)
        }


def extract_ppe_neural_features(
    worker_box: List[float],
    vest_box: List[float] = None,
    hardhat_box: List[float] = None,
    worker_conf: float = 1.0,
    vest_conf: float = 0.0,
    hardhat_conf: float = 0.0
) -> List[float]:
    """Converts raw YOLO boxes into the 14-dim input vector for PPEReasonerNet."""
    wx1, wy1, wx2, wy2 = worker_box
    ww = max(1e-4, wx2 - wx1)
    wh = max(1e-4, wy2 - wy1)

    # Worker torso region (middle 50% height)
    tx1, ty1, tx2, ty2 = wx1, wy1 + 0.25 * wh, wx2, wy1 + 0.75 * wh
    # Worker head region (upper 35% height)
    hx1, hy1, hx2, hy2 = wx1, wy1, wx2, wy1 + 0.35 * wh

    # Vest geometry relative to worker
    if vest_box:
        vx1, vy1, vx2, vy2 = vest_box
        v_iou_worker = _calc_iou(worker_box, vest_box)
        v_iou_torso = _calc_iou([tx1, ty1, tx2, ty2], vest_box)
        v_area_ratio = ((vx2 - vx1) * (vy2 - vy1)) / (ww * wh)
        has_vest_det = 1.0
    else:
        v_iou_worker = 0.0
        v_iou_torso = 0.0
        v_area_ratio = 0.0
        has_vest_det = 0.0

    # Hardhat geometry relative to worker
    if hardhat_box:
        hx_box, hy_box, hx2_box, hy2_box = hardhat_box
        h_iou_worker = _calc_iou(worker_box, hardhat_box)
        h_iou_head = _calc_iou([hx1, hy1, hx2, hy2], hardhat_box)
        h_area_ratio = ((hx2_box - hx_box) * (hy2_box - hy_box)) / (ww * wh)
        has_hardhat_det = 1.0
    else:
        h_iou_worker = 0.0
        h_iou_head = 0.0
        h_area_ratio = 0.0
        has_hardhat_det = 0.0

    return [
        worker_conf,
        has_vest_det,
        vest_conf,
        v_iou_worker,
        v_iou_torso,
        v_area_ratio,
        has_hardhat_det,
        hardhat_conf,
        h_iou_worker,
        h_iou_head,
        h_area_ratio,
        ww / wh,           # Worker aspect ratio
        min(1.0, ww * wh), # Worker screen coverage
        float(has_vest_det + has_hardhat_det) / 2.0
    ]


def _calc_iou(b1: List[float], b2: List[float]) -> float:
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0
