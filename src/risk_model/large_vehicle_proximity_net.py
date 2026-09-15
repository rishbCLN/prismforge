"""LargeVehicleProximityNet — Deep Residual Neural Network for Worker-Vehicle Proximity Risk.

Architecture:
  - 20-dim input (spatial feature vector per worker-vehicle pair)
  - 3 stacked Residual Blocks with Squeeze-and-Excitation attention: 20→64→128→64
  - Multi-head self-attention (4 heads) for spatial relationship reasoning
  - Pairwise feature interaction layer for cross-feature reasoning
  - Three task heads:
      • proximity_score  — continuous [0,1] regression (risk intensity)
      • proximity_class  — 4-class categorical (SAFE/SUPERVISED/DANGER/TOO_FAR)
      • vehicle_class    — binary (tractor=0 / truck=1) — auxiliary supervision

All V3 PPEReasonerNet patterns applied: GELU activations, LayerNorm, dropout,
EMAModel wrapper, temperature-scaled inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

FEATURE_DIM = 20
PROXIMITY_CLASSES = 4  # SAFE, SUPERVISED, DANGER, TOO_FAR


# ---------------------------------------------------------------------------
# Shared building blocks (same design as PPEReasonerNet V3)
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Channel attention via squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(4, channels // reduction)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.excite(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with SE attention."""

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
        return residual + self.se(self.block(x))


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature reasoning.
    Operates on a sequence of 1 token (batch of scalar features expanded to [B, 1, D]).
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] — expand to [B, 1, D] for attention
        B, D = x.shape
        h = x.unsqueeze(1)  # [B, 1, D]
        norm_h = self.norm(h)
        qkv = self.qkv(norm_h).reshape(B, 1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, 1, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        out = self.proj(out).squeeze(1)  # [B, D]
        return x + out


class PairwiseInteraction(nn.Module):
    """Explicit pairwise feature interaction: concatenate [x, x*x, x**2] projections."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Element-wise squared interaction
        x2 = x * x
        combined = torch.cat([x, x2], dim=-1)
        return self.norm(F.gelu(self.linear(combined)))


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class LargeVehicleProximityNet(nn.Module):
    """Worker–vehicle proximity risk scoring network.

    Input:  20-dim spatial feature vector per (worker, vehicle) pair
    Output: dict with keys:
      'proximity_score'  — [B] float, continuous risk in [0, 1]
      'proximity_class'  — [B, 4] logits for 4-class proximity
      'vehicle_class'    — [B] logit for binary vehicle type (auxiliary)
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.15,
        num_proximity_classes: int = PROXIMITY_CLASSES,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual blocks: 64→128→64
        self.res1 = ResidualBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim * 2, hidden_dim * 2, dropout=dropout)
        self.res3 = ResidualBlock(hidden_dim * 2, hidden_dim, dropout=dropout)

        # Spatial self-attention
        self.attention = SpatialSelfAttention(hidden_dim, num_heads=4, dropout=0.1)

        # Pairwise interaction
        self.interaction = PairwiseInteraction(hidden_dim, hidden_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task heads
        head_mid = hidden_dim // 2

        # Proximity score head (regression)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, head_mid),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_mid, head_mid // 2),
            nn.GELU(),
            nn.Linear(head_mid // 2, 1),
            nn.Sigmoid(),
        )

        # Proximity class head (4-class)
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, head_mid),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_mid, head_mid // 2),
            nn.GELU(),
            nn.Linear(head_mid // 2, num_proximity_classes),
        )

        # Vehicle class auxiliary head (binary)
        self.vehicle_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, 20] feature tensor

        Returns:
            dict with 'proximity_score', 'proximity_class', 'vehicle_class'
        """
        # Encode
        h = self.input_proj(x)

        # Residual tower
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)

        # Attention branch
        ha = self.attention(h)

        # Interaction branch
        hi = self.interaction(h)

        # Fuse attention + interaction
        fused = self.fusion(torch.cat([ha, hi], dim=-1))

        return {
            "proximity_score": self.score_head(fused).squeeze(-1),   # [B]
            "proximity_class": self.class_head(fused),                 # [B, 4]
            "vehicle_class": self.vehicle_head(fused).squeeze(-1),    # [B]
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# EMA wrapper (identical pattern to PPEReasonerNet)
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model weights for inference stability."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]):
        self.shadow = state["shadow"]
        self.decay = state.get("decay", self.decay)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_proximity_model(
    model: LargeVehicleProximityNet,
    output_path: str,
    ema: Optional[EMAModel] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save model weights and optional EMA to disk."""
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "architecture": "LargeVehicleProximityNet",
        "version": "V1",
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_proximity_classes": PROXIMITY_CLASSES,
        "metadata": metadata or {},
    }
    torch.save(payload, output_path)

    if ema is not None:
        ema_path = output_path.replace(".pt", "_ema.pt")
        ema_payload = dict(payload)
        ema_payload["ema_state"] = ema.state_dict()
        torch.save(ema_payload, ema_path)


def load_proximity_model(
    model_path: str,
    device: Optional[str] = None,
    strict: bool = False,
) -> LargeVehicleProximityNet:
    """Load LargeVehicleProximityNet from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(model_path, map_location=device, weights_only=False)
    model = LargeVehicleProximityNet(
        input_dim=payload.get("input_dim", FEATURE_DIM),
        hidden_dim=payload.get("hidden_dim", 64),
        num_proximity_classes=payload.get("num_proximity_classes", PROXIMITY_CLASSES),
    )
    model.load_state_dict(payload["model_state"], strict=strict)
    model.to(device)
    model.eval()
    return model
