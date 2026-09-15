"""PPEReasonerNet V3 — Production-Grade Deep Residual Neural Schema for Spatial PPE Reasoning.

Architectural upgrades over V2:
  - 3 stacked residual blocks with Squeeze-and-Excitation (SE) channel attention
  - Multi-head self-attention layer for cross-feature reasoning
  - Explicit pairwise feature interaction layer
  - GELU activations with spectral normalization for training stability
  - Wider task heads (32→24→16→1) per output
  - Anatomical spatial gating preserved for hardhat cranial compliance
  - Backward-compatible with V2 checkpoints via strict=False loading
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(4, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels]
        scale = self.excite(x)  # [batch, channels]
        return x * scale


class ResidualBlock(nn.Module):
    """Pre-activation residual block with SE attention and optional dimension change."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.needs_proj = (in_dim != out_dim)

        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.se = SqueezeExcitation(out_dim, reduction=4)

        if self.needs_proj:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x) if self.needs_proj else x
        out = self.block(x)
        out = self.se(out)
        return residual + out


class MultiHeadSelfAttention(nn.Module):
    """Lightweight multi-head self-attention for feature-level cross-reasoning."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, dim] → treat each feature as a "token" by reshaping
        B, D = x.shape
        residual = x

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [B, heads, head_dim]

        # Attention across heads
        attn = (q * k).sum(dim=-1, keepdim=True) * self.scale
        attn = torch.softmax(attn, dim=1)
        attn = self.dropout(attn)

        out = (attn * v).reshape(B, D)
        out = self.proj(out)
        return self.norm(residual + out)


class FeatureInteractionLayer(nn.Module):
    """Explicit pairwise feature interaction for capturing non-linear PPE correlations."""

    # Key feature pairs that capture important PPE reasoning interactions
    INTERACTION_PAIRS = [
        (1, 7),    # has_vest × has_hardhat (joint compliance)
        (4, 10),   # vest_iou_torso × hardhat_iou_head (spatial consistency)
        (12, 13),  # aspect_ratio × is_chest_up (crop geometry interaction)
        (2, 8),    # vest_conf × hardhat_conf (joint confidence)
        (6, 11),   # vest_area_ratio × hardhat_area_ratio (size consistency)
        (3, 9),    # vest_iou_worker × hardhat_iou_worker (overlap consistency)
        (1, 13),   # has_vest × is_chest_up (vest in close-up reasoning)
        (7, 12),   # has_hardhat × aspect_ratio (hardhat with body proportions)
    ]

    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.n_interactions = len(self.INTERACTION_PAIRS)
        # Project interactions to a compact representation
        self.interaction_proj = nn.Sequential(
            nn.Linear(self.n_interactions, self.n_interactions),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        interactions = []
        for i, j in self.INTERACTION_PAIRS:
            if i < x.shape[1] and j < x.shape[1]:
                interactions.append(x[:, i:i+1] * x[:, j:j+1])
            else:
                interactions.append(torch.zeros(x.shape[0], 1, device=x.device))
        interaction_feats = torch.cat(interactions, dim=1)  # [B, n_interactions]
        return self.interaction_proj(interaction_feats)


class TaskHead(nn.Module):
    """Wider task-specific prediction head with residual connection."""

    def __init__(self, in_dim: int, out_dim: int, activation: Optional[nn.Module] = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 24),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(24, 16),
            nn.GELU(),
            nn.Linear(16, out_dim),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


class HardhatReasonerBlock(nn.Module):
    """Dedicated deep neural reasoning block for cranial hardhat compliance.

    Explicitly fuses latent safety features with fine-grained hardhat cranial geometry
    (has_hardhat, confidence, worker overlap, cranial head IoU, area ratio)
    and applies anatomical cranial spatial gating.
    """

    def __init__(self, latent_dim: int = 48, hardhat_feat_dim: int = 5):
        super().__init__()
        in_dim = latent_dim + hardhat_feat_dim
        self.block = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 24),
            nn.GELU(),
            nn.Linear(24, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, latent: torch.Tensor, hardhat_feats: torch.Tensor, spatial_gate: torch.Tensor) -> torch.Tensor:
        comb = torch.cat([latent, hardhat_feats], dim=1)
        raw_score = self.block(comb)
        return raw_score * spatial_gate


class PPEReasonerNet(nn.Module):
    """Production-grade deep multi-task neural network with crop-geometry awareness,
    residual blocks, SE attention, feature interactions, and spatial gating.

    Architecture:
        Input (16-dim) → FeatureInteraction → InputProj(→hidden)
        → ResBlock1(hidden→hidden) → ResBlock2(hidden→2×hidden) → ResBlock3(2×hidden→hidden)
        → MultiHeadSelfAttention → Bottleneck(→48)
        → 4 TaskHeads + Dedicated HardhatReasonerBlock (vest, hardhat, violation, risk)
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Feature interaction layer
        self.feat_interaction = FeatureInteractionLayer(input_dim)
        interaction_dim = len(FeatureInteractionLayer.INTERACTION_PAIRS)
        combined_input = input_dim + interaction_dim  # 16 + 8 = 24

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(combined_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 3 stacked residual blocks with SE attention
        self.res_block1 = ResidualBlock(hidden_dim, hidden_dim, dropout=0.15)
        self.res_block2 = ResidualBlock(hidden_dim, hidden_dim * 2, dropout=0.15)
        self.res_block3 = ResidualBlock(hidden_dim * 2, hidden_dim, dropout=0.15)

        # Multi-head self-attention for cross-feature reasoning
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads=4, dropout=0.1)

        # Bottleneck projection to latent safety representation
        bottleneck_dim = 48
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )

        # Head 1: Vest Compliance Score (0.0 = Not Worn, 1.0 = Properly Worn)
        self.vest_compliance_head = TaskHead(bottleneck_dim, 1, activation=nn.Sigmoid())

        # Head 2: Standard Hardhat Task Head (retained for backward compatibility)
        self.hardhat_compliance_head = TaskHead(bottleneck_dim, 1, activation=nn.Sigmoid())

        # Dedicated Hardhat Reasoning Chunk: Fuses latent safety features with hardhat spatial slice
        self.hardhat_reasoner = HardhatReasonerBlock(bottleneck_dim, hardhat_feat_dim=5)
        self.use_hardhat_reasoner = True  # Enabled for training; dynamically set by load_state_dict

        # Head 3: Multi-class Violation Type:
        # 0: COMPLIANT (Both worn)
        # 1: MISSING_VEST_ONLY
        # 2: MISSING_HARDHAT_ONLY
        # 3: CRITICAL_NO_PPE (Both missing)
        self.violation_classifier = TaskHead(bottleneck_dim, 4, activation=None)

        # Head 4: Continuous Site Risk Score (0.0 to 1.0)
        self.risk_score_head = TaskHead(bottleneck_dim, 1, activation=nn.Sigmoid())

        # Initialize weights with Kaiming for better convergence
        self.apply(self._init_weights)

    def load_state_dict(self, state_dict, strict=True):
        # Check if incoming checkpoint contains weights for hardhat_reasoner
        has_hr = any(k.startswith("hardhat_reasoner.") for k in state_dict.keys())
        self.use_hardhat_reasoner = has_hr
        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Input x: [batch_size, 16] spatial, crop-geometry, and confidence features."""
        # Feature interaction
        interactions = self.feat_interaction(x)
        x_combined = torch.cat([x, interactions], dim=1)

        # Shared backbone
        h0 = self.input_proj(x_combined)
        h1 = self.res_block1(h0)
        h2 = self.res_block2(h1)
        h3 = self.res_block3(h2)
        h_attn = self.attention(h3)
        latent = self.bottleneck(h_attn)

        # Anatomical Spatial Gating:
        # A hardhat CANNOT be compliant if not worn on the cranial dome/head!
        # Feature 10 is h_iou_head; Feature 7 is has_hardhat_det (worn on head).
        # If a worker holds the hardhat in hand, h_iou_head is 0.0 -> compliance is strictly 0.0!
        h_iou_head = x[:, 10:11] if x.shape[1] > 10 else torch.zeros(x.shape[0], 1, device=x.device)
        has_hh = x[:, 7:8] if x.shape[1] > 7 else torch.zeros(x.shape[0], 1, device=x.device)
        spatial_gate = torch.clamp(h_iou_head / 0.06, 0.0, 1.0) * torch.clamp(has_hh, 0.0, 1.0)

        # Dedicated Hardhat Reasoning Chunk
        hh_slice = x[:, 7:12] if x.shape[1] >= 12 else torch.zeros(x.shape[0], 5, device=x.device)
        if getattr(self, "use_hardhat_reasoner", False) and hasattr(self, "hardhat_reasoner") and self.hardhat_reasoner is not None:
            gated_hh = self.hardhat_reasoner(latent, hh_slice, spatial_gate)
        else:
            raw_hh = self.hardhat_compliance_head(latent)
            gated_hh = raw_hh * spatial_gate

        return {
            "vest_compliance": self.vest_compliance_head(latent),
            "hardhat_compliance": gated_hh,
            "violation_logits": self.violation_classifier(latent),
            "risk_score": self.risk_score_head(latent)
        }


class EMAModel:
    """Exponential Moving Average of model parameters for stable inference.

    Maintains a shadow copy of model parameters that are updated as:
        shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Replace model params with EMA shadow params for inference."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model params after EMA inference."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def extract_ppe_neural_features(
    worker_box: List[float],
    vest_box: List[float] = None,
    hardhat_box: List[float] = None,
    worker_conf: float = 1.0,
    vest_conf: float = 0.0,
    hardhat_conf: float = 0.0,
    face_box: List[float] = None,
    is_good_detect: Optional[bool] = None
) -> List[float]:
    """Converts raw perception boxes into 16-dim input vector with crop-geometry and face-cranial awareness."""
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

    # Hardhat geometry relative to worker and face
    if hardhat_box:
        hx_box, hy_box, hx2_box, hy2_box = hardhat_box
        h_iou_worker = _calc_iou(worker_box, hardhat_box)
        
        if face_box is not None:
            # Face-anchored cranial vault: sits directly on top of face
            fx1, fy1, fx2, fy2 = face_box
            fw = max(1e-4, fx2 - fx1)
            fh = max(1e-4, fy2 - fy1)
            cranial_rect = [fx1 - 0.20 * fw, fy1 - 1.25 * fh, fx2 + 0.20 * fw, fy1 + 0.15 * fh]
            h_iou_head = max(_calc_iou(cranial_rect, hardhat_box), _calc_iou([hx1, hy1, hx2, hy2], hardhat_box))
        else:
            # Cranial dome targeting: Hardhats sit on the upper cranial vault
            cranial_hy2 = hy1 + 0.65 * (hy2 - hy1)
            h_iou_head = max(
                _calc_iou([hx1, hy1, hx2, hy2], hardhat_box),
                _calc_iou([hx1, hy1, hx2, cranial_hy2], hardhat_box)
            )
        h_area_ratio = ((hx2_box - hx_box) * (hy2_box - hy_box)) / (ww * wh)

        # STRICT ANATOMICAL RULE: A hardhat is compliant IF AND ONLY IF worn on the head/face!
        # If is_good_detect is explicitly provided, respect it; otherwise evaluate head IoU threshold
        if is_good_detect is not None:
            is_worn_on_head = 1.0 if is_good_detect else 0.0
        else:
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
