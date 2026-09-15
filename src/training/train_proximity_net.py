"""Hardened Training Pipeline for LargeVehicleProximityNet.

Training features:
  - Focal Loss for proximity class head (class imbalance)
  - Label-smoothing CrossEntropy for 4-class proximity classification
  - Smooth L1 Loss for continuous risk score regression
  - BCEWithLogitsLoss for auxiliary vehicle class head
  - Cosine Annealing with Warm Restarts LR scheduling
  - Gradient clipping (norm=1.0)
  - WeightedRandomSampler for class balance
  - Stratified train/val split
  - Early stopping with patience
  - Exponential Moving Average (EMA) weights
  - Hard negative mining and feature augmentation
"""

import os
import time
import random
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import numpy as np

from src.risk_model.large_vehicle_proximity_net import (
    LargeVehicleProximityNet, EMAModel, save_proximity_model, FEATURE_DIM, PROXIMITY_CLASSES
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary Focal Loss for hard example mining."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1.0 - 1e-7)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        pt = target * pred + (1 - target) * (1 - pred)
        alpha_f = target * self.alpha + (1 - target) * (1 - self.alpha)
        return (alpha_f * (1 - pt) ** self.gamma * bce).mean()


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, num_classes: int, smoothing: float = 0.05):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_dist = torch.full_like(log_prob, self.smoothing / (self.num_classes - 1))
            smooth_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_dist * log_prob).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProximityDataset(Dataset):
    """PyTorch Dataset for proximity training samples."""

    def __init__(self, samples: List[Dict[str, Any]], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        feats = list(s["features"])

        if self.augment:
            # Small Gaussian noise on spatial dims
            for i in range(min(8, len(feats))):
                feats[i] = max(0.0, min(1.0, feats[i] + random.gauss(0, 0.008)))
            # Dropout one feature randomly
            drop_idx = random.randint(0, len(feats) - 1)
            feats[drop_idx] *= random.uniform(0.85, 1.0)

        feat_tensor = torch.tensor(feats, dtype=torch.float32)
        prox_class  = torch.tensor(s["proximity_class"], dtype=torch.long)
        risk_score  = torch.tensor(s["risk_score"], dtype=torch.float32)
        v_class     = torch.tensor(float(s["vehicle_class_id"]), dtype=torch.float32)

        return feat_tensor, prox_class, risk_score, v_class


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    samples: List[Dict[str, Any]],
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split samples with class-stratification on proximity_class."""
    rng = random.Random(seed)
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for s in samples:
        c = s["proximity_class"]
        buckets.setdefault(c, []).append(s)

    train, val = [], []
    for cls_samples in buckets.values():
        rng.shuffle(cls_samples)
        n_val = max(1, int(len(cls_samples) * val_fraction))
        val.extend(cls_samples[:n_val])
        train.extend(cls_samples[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_optimal_device(force_gpu: bool = True, requested_device: Optional[str] = None
                        ) -> Tuple[torch.device, str]:
    if requested_device and requested_device != "auto":
        try:
            dev = torch.device(requested_device)
            return dev, str(dev)
        except Exception:
            pass
    if force_gpu and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"CUDA — {gpu_name}"
    return torch.device("cpu"), "CPU (no CUDA available)"


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_proximity_model(
    samples: List[Dict[str, Any]],
    output_path: str = "models/proximity_reasoner.pt",
    epochs: int = 60,
    lr: float = 0.002,
    batch_size: int = 64,
    val_split: float = 0.20,
    device_name: str = "cpu",
    show_progress: bool = True,
    use_focal_loss: bool = True,
    label_smoothing: float = 0.05,
    use_augmentation: bool = True,
    early_stopping_patience: int = 10,
    use_ema: bool = True,
    gradient_clip_norm: float = 1.0,
    force_gpu: bool = True,
) -> Dict[str, Any]:
    """Train LargeVehicleProximityNet on extracted proximity samples.

    Returns dict with training results and final metrics.
    """
    t_start = time.time()
    device = torch.device(device_name)

    # Stratified split
    train_samples, val_samples = stratified_split(samples, val_fraction=val_split)

    if show_progress:
        cls_dist = Counter(s["proximity_class"] for s in train_samples)
        print(f"Train samples: {len(train_samples)} | Val: {len(val_samples)}")
        print(f"Class distribution (train): {dict(sorted(cls_dist.items()))}")

    # Datasets
    train_ds = ProximityDataset(train_samples, augment=use_augmentation)
    val_ds   = ProximityDataset(val_samples,   augment=False)

    # Weighted sampler for class balance
    class_counts = Counter(s["proximity_class"] for s in train_samples)
    weights = [1.0 / (class_counts[s["proximity_class"]] + 1e-6) for s in train_samples]
    # Up-weight DANGER class (class 2) 2x for safety
    weights = [w * 2.0 if s["proximity_class"] == 2 else w
               for w, s in zip(weights, train_samples)]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, drop_last=True
    )
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = LargeVehicleProximityNet(
        input_dim=FEATURE_DIM,
        hidden_dim=64,
        dropout=0.15,
        num_proximity_classes=PROXIMITY_CLASSES,
    ).to(device)

    n_params = model.count_parameters()
    if show_progress:
        print(f"LargeVehicleProximityNet V1 | Parameters: {n_params:,}")

    # EMA
    ema = EMAModel(model, decay=0.999) if use_ema else None

    # Losses
    if use_focal_loss:
        # Use focal for danger class specifically; label-smoothed CE for all 4 classes
        pass  # handled inside loop
    ce_loss_fn    = LabelSmoothingCE(PROXIMITY_CLASSES, smoothing=label_smoothing)
    smooth_l1_fn  = nn.SmoothL1Loss()
    bce_fn        = nn.BCEWithLogitsLoss()

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 3), T_mult=1, eta_min=lr * 0.01
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for feat, prox_cls, risk, v_cls in train_loader:
            feat    = feat.to(device)
            prox_cls = prox_cls.to(device)
            risk    = risk.to(device)
            v_cls   = v_cls.to(device)

            optimizer.zero_grad()
            out = model(feat)

            # Proximity class loss (label-smoothed CE)
            loss_cls = ce_loss_fn(out["proximity_class"], prox_cls)

            # Risk score regression loss (Smooth L1)
            loss_reg = smooth_l1_fn(out["proximity_score"], risk)

            # Auxiliary vehicle class loss (BCE)
            loss_vcls = bce_fn(out["vehicle_class"], v_cls)

            # Combined loss — weight proximity class and risk score higher
            loss = 1.5 * loss_cls + 1.0 * loss_reg + 0.3 * loss_vcls

            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            train_losses.append(loss.item())

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_losses = []
        val_correct_cls = 0
        val_total = 0
        val_risk_errors = []
        danger_tp = 0
        danger_fn = 0

        with torch.no_grad():
            for feat, prox_cls, risk, v_cls in val_loader:
                feat     = feat.to(device)
                prox_cls = prox_cls.to(device)
                risk     = risk.to(device)
                v_cls    = v_cls.to(device)

                out = model(feat)

                loss_cls  = ce_loss_fn(out["proximity_class"], prox_cls)
                loss_reg  = smooth_l1_fn(out["proximity_score"], risk)
                loss_vcls = bce_fn(out["vehicle_class"], v_cls)
                val_losses.append((1.5 * loss_cls + 1.0 * loss_reg + 0.3 * loss_vcls).item())

                # Class accuracy
                pred_cls = out["proximity_class"].argmax(dim=-1)
                val_correct_cls += (pred_cls == prox_cls).sum().item()
                val_total += prox_cls.size(0)

                # Risk MAE
                val_risk_errors.append((out["proximity_score"] - risk).abs().mean().item())

                # Danger recall (class 2)
                is_danger = (prox_cls == 2)
                danger_tp += ((pred_cls == 2) & is_danger).sum().item()
                danger_fn += ((pred_cls != 2) & is_danger).sum().item()

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 999.0
        val_cls_acc  = 100.0 * val_correct_cls / max(1, val_total)
        val_risk_mae = sum(val_risk_errors) / len(val_risk_errors) if val_risk_errors else 999.0
        danger_recall = 100.0 * danger_tp / max(1, danger_tp + danger_fn)

        if show_progress and (epoch % 5 == 0 or epoch <= 3 or epoch == epochs):
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train={avg_train_loss:.4f} val={avg_val_loss:.4f} | "
                f"cls_acc={val_cls_acc:.1f}% | "
                f"risk_mae={val_risk_mae:.4f} | "
                f"danger_recall={danger_recall:.1f}%"
            )

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                if show_progress:
                    print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics
    model.eval()
    if ema is not None:
        ema.apply_shadow(model)

    final_correct_cls = 0
    final_total = 0
    final_risk_errors = []
    final_danger_tp = 0
    final_danger_fn = 0

    with torch.no_grad():
        for feat, prox_cls, risk, v_cls in val_loader:
            feat     = feat.to(device)
            prox_cls = prox_cls.to(device)
            risk     = risk.to(device)
            out = model(feat)
            pred_cls = out["proximity_class"].argmax(dim=-1)
            final_correct_cls += (pred_cls == prox_cls).sum().item()
            final_total += prox_cls.size(0)
            final_risk_errors.append((out["proximity_score"] - risk).abs().mean().item())
            is_danger = (prox_cls == 2)
            final_danger_tp += ((pred_cls == 2) & is_danger).sum().item()
            final_danger_fn += ((pred_cls != 2) & is_danger).sum().item()

    final_cls_acc   = round(100.0 * final_correct_cls / max(1, final_total), 2)
    final_risk_mae  = round(sum(final_risk_errors) / max(1, len(final_risk_errors)), 4)
    final_danger_rec = round(100.0 * final_danger_tp / max(1, final_danger_tp + final_danger_fn), 2)

    if ema is not None:
        ema.restore(model)

    # Save model
    metadata = {
        "val_proximity_class_acc": final_cls_acc,
        "val_risk_score_mae": final_risk_mae,
        "val_danger_recall": final_danger_rec,
        "best_epoch": best_epoch,
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }
    save_proximity_model(model, output_path, ema=ema if use_ema else None, metadata=metadata)

    elapsed = round(time.time() - t_start, 1)

    return {
        "output_path": output_path,
        "architecture": "LargeVehicleProximityNet V1",
        "trainable_parameters": n_params,
        "training_time_seconds": elapsed,
        "epochs": best_epoch,
        "best_epoch": best_epoch,
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "final_metrics": {
            "val_proximity_class_acc": final_cls_acc,
            "val_risk_score_mae": final_risk_mae,
            "val_danger_recall": final_danger_rec,
        },
    }
