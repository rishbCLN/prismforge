"""PPEReasonerNet V2 — Advanced Residual Neural Schema for Spatial PPE Reasoning.
Processes raw YOLO bounding boxes, anatomical IoUs, aspect ratios, and crop-geometry features.
Specifically engineered to accurately detect safety vests in both full-body views and
chest-up portrait close-up crops without false negatives.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple


class PPEReasonerNet(nn.Module):
    """Deep multi-task neural network with crop-geometry awareness and residual blocks."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )

        # Residual Block (handles spatial scale and crop geometry invariance)
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.res_act = nn.LeakyReLU(0.1)

        # Bottleneck projection to latent safety representation
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1)
        )

        # Head 1: Vest Compliance Score (0.0 = Not Worn, 1.0 = Properly Worn)
        # Deep 2-layer MLP head with non-linear capacity for chest-up necklines
        self.vest_compliance_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Head 2: Hardhat Compliance Score (0.0 = Not Worn, 1.0 = Properly Worn)
        self.hardhat_compliance_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Head 3: Multi-class Violation Type:
        # 0: COMPLIANT (Both worn)
        # 1: MISSING_VEST_ONLY
        # 2: MISSING_HARDHAT_ONLY
        # 3: CRITICAL_NO_PPE (Both missing)
        self.violation_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 4)
        )

        # Head 4: Continuous Site Risk Score (0.0 to 1.0)
        self.risk_score_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Input x: [batch_size, 16] spatial, crop-geometry, and confidence features."""
        h0 = self.input_proj(x)
        h1 = self.res_act(h0 + self.res_block(h0))
        latent = self.bottleneck(h1)

        raw_hh = self.hardhat_compliance_head(latent)
        # Anatomical Spatial Gating:
        # A hardhat CANNOT be compliant if not worn on the cranial dome/head!
        # Feature 10 is h_iou_head; Feature 7 is has_hardhat_det (worn on head).
        # If a worker holds the hardhat in hand, h_iou_head is 0.0 -> compliance is strictly 0.0!
        h_iou_head = x[:, 10:11]
        has_hh = x[:, 7:8]
        spatial_gate = torch.clamp(h_iou_head / 0.06, 0.0, 1.0) * torch.clamp(has_hh, 0.0, 1.0)
        gated_hh = raw_hh * spatial_gate

        return {
            "vest_compliance": self.vest_compliance_head(latent),
            "hardhat_compliance": gated_hh,
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
    """Converts raw YOLO boxes into 16-dim input vector with crop-geometry awareness."""
    wx1, wy1, wx2, wy2 = worker_box
    ww = max(1e-4, wx2 - wx1)
    wh = max(1e-4, wy2 - wy1)
    aspect_ratio = ww / wh

    # Detect if image/box is a chest-up / portrait crop vs full standing body
    is_chest_up = 1.0 if aspect_ratio >= 0.48 else 0.0

    if is_chest_up > 0.5:
        # Chest-up framing: Head takes upper 48%, Upper chest spans 35% to bottom
        hx1, hy1, hx2, hy2 = wx1, wy1, wx2, wy1 + 0.48 * wh
        tx1, ty1, tx2, ty2 = wx1, wy1 + 0.35 * wh, wx2, wy2
        cx1, cy1, cx2, cy2 = wx1, wy1 + 0.35 * wh, wx2, wy2
    else:
        # Full-body standing framing: Head is top 28%, Torso is middle 50%
        hx1, hy1, hx2, hy2 = wx1, wy1, wx2, wy1 + 0.28 * wh
        tx1, ty1, tx2, ty2 = wx1, wy1 + 0.22 * wh, wx2, wy1 + 0.70 * wh
        cx1, cy1, cx2, cy2 = wx1, wy1 + 0.20 * wh, wx2, wy1 + 0.48 * wh

    # Vest geometry relative to worker
    if vest_box:
        vx1, vy1, vx2, vy2 = vest_box
        v_iou_worker = _calc_iou(worker_box, vest_box)
        v_iou_torso = _calc_iou([tx1, ty1, tx2, ty2], vest_box)
        v_iou_upper_chest = _calc_iou([cx1, cy1, cx2, cy2], vest_box)
        v_area_ratio = ((vx2 - vx1) * (vy2 - vy1)) / (ww * wh)
        has_vest_det = 1.0
    else:
        v_iou_worker = 0.0
        v_iou_torso = 0.0
        v_iou_upper_chest = 0.0
        v_area_ratio = 0.0
        has_vest_det = 0.0

    # Hardhat geometry relative to worker
    if hardhat_box:
        hx_box, hy_box, hx2_box, hy2_box = hardhat_box
        h_iou_worker = _calc_iou(worker_box, hardhat_box)
        # Cranial dome targeting: Hardhats sit on the upper cranial vault
        cranial_hy2 = hy1 + 0.65 * (hy2 - hy1)
        h_iou_head = max(
            _calc_iou([hx1, hy1, hx2, hy2], hardhat_box),
            _calc_iou([hx1, hy1, hx2, cranial_hy2], hardhat_box)
        )
        h_area_ratio = ((hx2_box - hx_box) * (hy2_box - hy_box)) / (ww * wh)

        # STRICT ANATOMICAL RULE: A hardhat is compliant IF AND ONLY IF worn on the head!
        # If the hardhat is held in hand, waist, or lap (h_iou_head < 0.06), it is NOT worn!
        is_worn_on_head = 1.0 if h_iou_head >= 0.06 else 0.0
        has_hardhat_det = is_worn_on_head
        calibrated_hh_conf = hardhat_conf if is_worn_on_head > 0.5 else 0.0
    else:
        h_iou_worker = 0.0
        h_iou_head = 0.0
        h_area_ratio = 0.0
        has_hardhat_det = 0.0
        calibrated_hh_conf = 0.0

    return [
        worker_conf,
        has_vest_det,
        vest_conf,
        v_iou_worker,
        v_iou_torso,
        v_iou_upper_chest,
        v_area_ratio,
        has_hardhat_det,
        calibrated_hh_conf,
        h_iou_worker,
        h_iou_head,
        h_area_ratio,
        aspect_ratio,
        is_chest_up,
        min(1.0, ww * wh),
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
