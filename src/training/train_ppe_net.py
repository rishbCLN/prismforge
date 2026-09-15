"""Hardened Training Pipeline for PPEReasonerNet V3.

Production-grade training with:
  - Focal Loss for binary classification (hard example mining)
  - Label-smoothing CrossEntropy for violation classification
  - Cosine annealing with warm restarts LR scheduling
  - Gradient clipping for training stability
  - Class-balanced weighted random sampling
  - Early stopping with patience
  - Best model checkpointing (lowest validation loss)
  - Stratified train/val split
  - Exponential Moving Average (EMA) of weights
  - Extensive data augmentation (noise, dropout, confidence perturbation, scale jitter)
  - Extended hard-negative mining
"""
import os
import json
import time
import subprocess
import random
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

from src.risk_model.ppe_reasoner_net import PPEReasonerNet, EMAModel, extract_ppe_neural_features, _calc_iou


# ---------------------------------------------------------------------------
# Custom Loss Functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary Focal Loss for hard example mining.
    Down-weights well-classified examples, focuses on misclassified ones.
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1.0 - 1e-7)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        pt = target * pred + (1 - target) * (1 - pred)
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing for better generalization."""

    def __init__(self, smoothing: float = 0.05, num_classes: int = 4):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=1)
        nll_loss = F.nll_loss(log_probs, target, reduction='none')
        smooth_loss = -log_probs.mean(dim=1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

class PPEFeatureAugmentor:
    """Augments 16-dim PPE feature vectors during training for robustness."""

    def __init__(
        self,
        noise_std: float = 0.02,
        feature_dropout_prob: float = 0.05,
        confidence_perturbation: float = 0.1,
        iou_scale_jitter: float = 0.15,
        enabled: bool = True,
    ):
        self.noise_std = noise_std
        self.feature_dropout_prob = feature_dropout_prob
        self.confidence_perturbation = confidence_perturbation
        self.iou_scale_jitter = iou_scale_jitter
        self.enabled = enabled

    def augment_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a batch of feature vectors."""
        if not self.enabled:
            return X

        X_aug = X.clone()

        # 1. Gaussian noise on all features
        noise = torch.randn_like(X_aug) * self.noise_std
        X_aug = X_aug + noise

        # 2. Random feature dropout (zero out individual features per sample)
        dropout_mask = torch.rand_like(X_aug) > self.feature_dropout_prob
        X_aug = X_aug * dropout_mask.float()

        # 3. Confidence perturbation (features 0=worker_conf, 2=vest_conf, 8=hh_conf)
        conf_indices = [0, 2, 8]
        for idx in conf_indices:
            if idx < X_aug.shape[1]:
                perturb = (torch.rand(X_aug.shape[0], 1, device=X_aug.device) * 2 - 1) * self.confidence_perturbation
                X_aug[:, idx:idx+1] = torch.clamp(X_aug[:, idx:idx+1] + perturb, 0.0, 1.0)

        # 4. IoU scale jitter (features 3,4,5,9,10 are IoU values)
        iou_indices = [3, 4, 5, 9, 10]
        for idx in iou_indices:
            if idx < X_aug.shape[1]:
                scale = 1.0 + (torch.rand(X_aug.shape[0], 1, device=X_aug.device) * 2 - 1) * self.iou_scale_jitter
                X_aug[:, idx:idx+1] = torch.clamp(X_aug[:, idx:idx+1] * scale, 0.0, 1.0)

        # 5. Aspect ratio jitter (feature 12)
        if X_aug.shape[1] > 12:
            ar_perturb = (torch.rand(X_aug.shape[0], 1, device=X_aug.device) * 2 - 1) * 0.05
            X_aug[:, 12:13] = torch.clamp(X_aug[:, 12:13] + ar_perturb, 0.0, 2.0)

        return X_aug


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PPEDataset(Dataset):
    """PyTorch Dataset for PPEReasonerNet multi-task learning."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.X = torch.tensor([s["features"] for s in samples], dtype=torch.float32)
        self.y_vest = torch.tensor([[s["vest_label"]] for s in samples], dtype=torch.float32)
        self.y_hardhat = torch.tensor([[s["hardhat_label"]] for s in samples], dtype=torch.float32)
        self.y_violation = torch.tensor([s["violation_class"] for s in samples], dtype=torch.long)
        self.y_risk = torch.tensor([[s["risk_score"]] for s in samples], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_vest[idx], self.y_hardhat[idx], self.y_violation[idx], self.y_risk[idx]


# ---------------------------------------------------------------------------
# Device Selection
# ---------------------------------------------------------------------------

def get_optimal_device(force_gpu: bool = False, requested_device: Optional[str] = None) -> Tuple[torch.device, str]:
    """Detects GPU hardware and selects optimal PyTorch compute device."""
    gpu_name = "N/A"
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        if smi_out:
            gpu_name = smi_out.split("\n")[0]
    except Exception:
        pass

    if requested_device and requested_device.lower() == "cpu":
        return torch.device("cpu"), f"CPU (explicitly requested) [NVIDIA GPU '{gpu_name}' idle]"

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        dev_name = torch.cuda.get_device_name(0)
        return torch.device("cuda:0"), f"CUDA:0 ({dev_name})"

    # If GPU was requested or forced but CUDA is not available in PyTorch build
    if force_gpu or (requested_device and requested_device.lower() in ("cuda", "gpu")):
        if gpu_name != "N/A":
            print(f"\n[GPU Hardware Detected] {gpu_name}")
            print("[PyTorch Environment Note] Local Python is 3.14.2; official PyTorch CUDA binaries are currently built for Python <=3.12/3.13.")
            print("[Auto-Optimization] Leveraging multi-core CPU parallelism (OpenMP/MKL) for high throughput.\n")
        else:
            print("[Warning] No NVIDIA GPU found via nvidia-smi. Using CPU backend.\n")

    return torch.device("cpu"), f"CPU (Multi-threaded) [GPU hardware: {gpu_name}]"


# ---------------------------------------------------------------------------
# Hard Negative Mining & Data Augmentation (Sample-Level)
# ---------------------------------------------------------------------------

def _generate_hard_negatives(
    samples: List[Dict[str, Any]],
    worker_boxes_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate extended hard negative samples for robustness.

    Creates synthetic samples for difficult edge cases:
    1. Hardhat held at waist/lap (not on head)
    2. Hardhat at shoulder level (not on head)
    3. Hardhat on the ground near worker
    4. Partial vest occlusion (borderline visible)
    5. Extremely small/large workers (scale variation)
    6. Low-confidence PPE detections
    """
    hard_negatives = []

    for sample in samples:
        feats = sample["features"]
        has_vest = feats[1]  # has_vest_det
        has_hh = feats[7]    # has_hardhat_det

        # --- Hardhat position negatives ---
        if has_hh > 0.5 and random.random() < 0.25:
            # Hardhat at waist/hand level
            neg_feats = feats.copy()
            neg_feats[7] = 0.0   # has_hardhat_det = 0
            neg_feats[8] = 0.0   # calibrated_hh_conf = 0
            neg_feats[10] = 0.0  # h_iou_head = 0 (not on head)
            neg_feats[9] = max(0.02, feats[9] * 0.3)   # reduced h_iou_worker
            neg_feats[11] = feats[11] * 0.5  # smaller area ratio at waist

            viol = 2 if has_vest > 0.5 else 3
            risk = 0.55 if has_vest > 0.5 else 0.90
            hard_negatives.append({
                "features": neg_feats,
                "vest_label": sample["vest_label"],
                "hardhat_label": 0.0,
                "violation_class": viol,
                "risk_score": risk
            })

        if has_hh > 0.5 and random.random() < 0.15:
            # Hardhat at shoulder level
            neg_feats = feats.copy()
            neg_feats[7] = 0.0
            neg_feats[8] = 0.0
            neg_feats[10] = 0.03  # Very low head IoU (near threshold but below)
            neg_feats[9] = max(0.05, feats[9] * 0.5)

            viol = 2 if has_vest > 0.5 else 3
            risk = 0.55 if has_vest > 0.5 else 0.90
            hard_negatives.append({
                "features": neg_feats,
                "vest_label": sample["vest_label"],
                "hardhat_label": 0.0,
                "violation_class": viol,
                "risk_score": risk
            })

        if has_hh > 0.5 and random.random() < 0.10:
            # Hardhat on the ground near worker
            neg_feats = feats.copy()
            neg_feats[7] = 0.0
            neg_feats[8] = 0.0
            neg_feats[10] = 0.0
            neg_feats[9] = 0.0   # No IoU with worker at all
            neg_feats[11] = feats[11] * 0.8  # Similar size but on ground

            viol = 2 if has_vest > 0.5 else 3
            risk = 0.55 if has_vest > 0.5 else 0.90
            hard_negatives.append({
                "features": neg_feats,
                "vest_label": sample["vest_label"],
                "hardhat_label": 0.0,
                "violation_class": viol,
                "risk_score": risk
            })

        # --- Vest boundary negatives ---
        if has_vest > 0.5 and random.random() < 0.12:
            # Partial vest occlusion: vest IoU reduced but still present
            neg_feats = feats.copy()
            neg_feats[4] = max(0.06, feats[4] * 0.4)  # Reduced torso IoU
            neg_feats[5] = max(0.04, feats[5] * 0.3)  # Reduced upper chest IoU
            neg_feats[6] = feats[6] * 0.5  # Smaller visible area
            # Still compliant (vest IS worn, just partially occluded)
            hard_negatives.append({
                "features": neg_feats,
                "vest_label": 1.0,
                "hardhat_label": sample["hardhat_label"],
                "violation_class": sample["violation_class"],
                "risk_score": sample["risk_score"]
            })

        # --- Scale variation samples ---
        if random.random() < 0.08:
            # Very small worker (far from camera)
            neg_feats = feats.copy()
            neg_feats[14] = max(0.001, feats[14] * 0.1)  # Very small area
            neg_feats[12] = feats[12] * (0.8 + random.random() * 0.4)  # Slightly different aspect
            hard_negatives.append({
                "features": neg_feats,
                "vest_label": sample["vest_label"],
                "hardhat_label": sample["hardhat_label"],
                "violation_class": sample["violation_class"],
                "risk_score": sample["risk_score"]
            })

        # --- Low confidence PPE ---
        if (has_vest > 0.5 or has_hh > 0.5) and random.random() < 0.10:
            neg_feats = feats.copy()
            # Very low detection confidence
            neg_feats[0] = max(0.25, feats[0] * 0.5)  # Low worker conf
            if has_vest > 0.5:
                neg_feats[2] = max(0.3, feats[2] * 0.4)  # Low vest conf
            if has_hh > 0.5:
                neg_feats[8] = max(0.3, feats[8] * 0.4)  # Low hardhat conf
            # Still same labels (low confidence doesn't change ground truth)
            hard_negatives.append({
                "features": neg_feats,
                "vest_label": sample["vest_label"],
                "hardhat_label": sample["hardhat_label"],
                "violation_class": sample["violation_class"],
                "risk_score": sample["risk_score"]
            })

    return hard_negatives


# ---------------------------------------------------------------------------
# Sample Extraction from Ingested Data
# ---------------------------------------------------------------------------

def extract_ppe_training_samples(manifest: Dict[str, Any], show_progress: bool = True) -> List[Dict[str, Any]]:
    """Extracts 16-dim spatial features and multi-task supervision labels from ingested images."""
    images = manifest.get("images", [])
    samples = []

    iterator = tqdm(images, desc="Extracting PPE Spatial Features", unit="img") if show_progress else images

    for img in iterator:
        objs = img.get("objects", [])
        workers = [o for o in objs if o.get("class_name") == "person"]
        helmets = [o for o in objs if o.get("class_name") == "helmet"]
        no_helmets = [o for o in objs if o.get("class_name") == "no_helmet"]
        vests = [o for o in objs if o.get("class_name") == "vest"]
        no_vests = [o for o in objs if o.get("class_name") == "no_vest"]

        for w in workers:
            wb = [w["x1"], w["y1"], w["x2"], w["y2"]]
            wh = max(1e-4, w["y2"] - w["y1"])
            ww = max(1e-4, w["x2"] - w["x1"])
            aspect_ratio = ww / wh
            is_chest_up = aspect_ratio >= 0.48

            if is_chest_up:
                head_box = [w["x1"], w["y1"], w["x2"], w["y1"] + 0.48 * wh]
                torso_box = [w["x1"], w["y1"] + 0.32 * wh, w["x2"], w["y2"]]
            else:
                head_box = [w["x1"], w["y1"], w["x2"], w["y1"] + 0.28 * wh]
                torso_box = [w["x1"], w["y1"] + 0.20 * wh, w["x2"], w["y1"] + 0.72 * wh]

            # Best matching vest
            best_v = None
            best_v_iou = 0.0
            for v in vests:
                vb = [v["x1"], v["y1"], v["x2"], v["y2"]]
                iou = _calc_iou(torso_box, vb)
                if iou > best_v_iou:
                    best_v_iou = iou
                    best_v = vb

            has_explicit_no_vest = any(_calc_iou(torso_box, [nv["x1"], nv["y1"], nv["x2"], nv["y2"]]) > 0.05 for nv in no_vests)
            has_vest = (best_v is not None and best_v_iou > 0.05) and not has_explicit_no_vest

            # Best matching hardhat / helmet
            cranial_box = [w["x1"], w["y1"], w["x2"], w["y1"] + (0.35 * wh if is_chest_up else 0.20 * wh)]
            best_h = None
            best_h_iou = 0.0
            for h in helmets:
                hb = [h["x1"], h["y1"], h["x2"], h["y2"]]
                iou = max(_calc_iou(head_box, hb), _calc_iou(cranial_box, hb))
                if iou > best_h_iou:
                    best_h_iou = iou
                    best_h = hb

            has_explicit_no_helmet = any(max(_calc_iou(head_box, [nh["x1"], nh["y1"], nh["x2"], nh["y2"]]), _calc_iou(cranial_box, [nh["x1"], nh["y1"], nh["x2"], nh["y2"]])) > 0.05 for nh in no_helmets)
            has_hardhat = (best_h is not None and best_h_iou > 0.05) and not has_explicit_no_helmet

            feats = extract_ppe_neural_features(
                worker_box=wb,
                vest_box=best_v if has_vest else None,
                hardhat_box=best_h if has_hardhat else None,
                worker_conf=w.get("confidence", 1.0),
                vest_conf=0.90 if has_vest else 0.0,
                hardhat_conf=0.90 if has_hardhat else 0.0
            )

            v_label = 1.0 if has_vest else 0.0
            h_label = 1.0 if has_hardhat else 0.0
            if v_label == 1.0 and h_label == 1.0:
                viol = 0  # COMPLIANT
                risk = 0.05
            elif v_label == 0.0 and h_label == 1.0:
                viol = 1  # MISSING_VEST
                risk = 0.45
            elif v_label == 1.0 and h_label == 0.0:
                viol = 2  # MISSING_HARDHAT
                risk = 0.55
            else:
                viol = 3  # CRITICAL_NO_PPE
                risk = 0.90

            samples.append({
                "features": feats,
                "vest_label": v_label,
                "hardhat_label": h_label,
                "violation_class": viol,
                "risk_score": risk
            })

            # Augmentation: Explicit negative training for hardhat held in hand / waist
            if has_hardhat and len(samples) % 5 == 0:
                held_h = [w["x1"] + 0.15 * ww, w["y1"] + 0.58 * wh, w["x2"] - 0.15 * ww, w["y1"] + 0.78 * wh]
                feats_held = extract_ppe_neural_features(
                    worker_box=wb,
                    vest_box=best_v if has_vest else None,
                    hardhat_box=held_h,
                    worker_conf=w.get("confidence", 1.0),
                    vest_conf=0.90 if has_vest else 0.0,
                    hardhat_conf=0.90
                )
                samples.append({
                    "features": feats_held,
                    "vest_label": v_label,
                    "hardhat_label": 0.0,  # Strict rule: Holding hardhat in hand is non-compliant!
                    "violation_class": 2 if has_vest else 3,  # MISSING_HARDHAT
                    "risk_score": 0.55 if has_vest else 0.90
                })

    # Generate extended hard negatives
    print(f"\n--- Generating Hard Negative Samples for Robustness ---")
    hard_negs = _generate_hard_negatives(samples, [])
    print(f"Generated {len(hard_negs)} hard negative samples.")
    samples.extend(hard_negs)

    return samples


# ---------------------------------------------------------------------------
# Stratified Split
# ---------------------------------------------------------------------------

def _stratified_split(
    dataset: PPEDataset,
    val_fraction: float = 0.2
) -> Tuple[List[int], List[int]]:
    """Stratified train/val split preserving violation class distribution."""
    indices_by_class: Dict[int, List[int]] = {}
    for i in range(len(dataset)):
        cls = int(dataset.y_violation[i].item())
        indices_by_class.setdefault(cls, []).append(i)

    train_indices = []
    val_indices = []

    for cls, indices in indices_by_class.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def train_ppe_model(
    samples: List[Dict[str, Any]],
    output_path: str = "models/ppe_reasoner.pt",
    epochs: int = 30,
    lr: float = 0.003,
    batch_size: int = 64,
    val_split: float = 0.2,
    force_gpu: bool = False,
    device_name: Optional[str] = None,
    show_progress: bool = True,
    use_focal_loss: bool = True,
    label_smoothing: float = 0.05,
    use_augmentation: bool = True,
    early_stopping_patience: int = 10,
    use_ema: bool = True,
    gradient_clip_norm: float = 1.0,
) -> Dict[str, Any]:
    """Trains PPEReasonerNet V3 with production-grade training strategies."""
    if not samples:
        raise ValueError("Cannot train PPEReasonerNet: 0 samples provided.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device, device_desc = get_optimal_device(force_gpu=force_gpu, requested_device=device_name)
    if show_progress:
        print(f"\n[Compute Device] {device_desc}")

    # Print class distribution
    class_counts = Counter(s["violation_class"] for s in samples)
    class_names = {0: "COMPLIANT", 1: "MISSING_VEST", 2: "MISSING_HARDHAT", 3: "CRITICAL_NO_PPE"}
    if show_progress:
        print(f"\n[Class Distribution]")
        for cls_id in sorted(class_counts.keys()):
            pct = class_counts[cls_id] / len(samples) * 100
            print(f"  {class_names.get(cls_id, f'Class_{cls_id}')}: {class_counts[cls_id]:,} ({pct:.1f}%)")

    dataset = PPEDataset(samples)

    # Stratified split
    train_indices, val_indices = _stratified_split(dataset, val_fraction=val_split)
    train_size = len(train_indices)
    val_size = len(val_indices)

    if show_progress:
        print(f"\n[Data Split] Train: {train_size:,} | Val: {val_size:,} (stratified)")

    # Class-balanced WeightedRandomSampler for training
    train_labels = [int(dataset.y_violation[i].item()) for i in train_indices]
    train_class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in train_class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=min(batch_size, train_size), sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=min(batch_size, val_size), shuffle=False)

    # Model
    model = PPEReasonerNet(input_dim=16, hidden_dim=64).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if show_progress:
        print(f"\n[Model] PPEReasonerNet V3 | {param_count:,} trainable parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # LR Scheduler: Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(5, epochs // 4), T_mult=2, eta_min=lr * 0.01
    )

    # Loss functions
    if use_focal_loss:
        criterion_vest = FocalLoss(alpha=0.75, gamma=2.0)
        criterion_hardhat = FocalLoss(alpha=0.75, gamma=2.0)
    else:
        criterion_vest = nn.BCELoss()
        criterion_hardhat = nn.BCELoss()

    if label_smoothing > 0:
        criterion_ce = LabelSmoothingCrossEntropy(smoothing=label_smoothing, num_classes=4)
    else:
        criterion_ce = nn.CrossEntropyLoss()

    criterion_mse = nn.SmoothL1Loss()  # Huber loss: more robust to outliers than MSE

    # Augmentor
    augmentor = PPEFeatureAugmentor(enabled=use_augmentation)

    # EMA
    ema = EMAModel(model, decay=0.999) if use_ema else None

    # Loss weights: penalizing risk misestimation hardest
    W_VEST = 1.5
    W_HARDHAT = 1.5
    W_VIOLATION = 2.0
    W_RISK = 4.0

    # Early stopping state
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []
    start_time = time.time()

    if show_progress:
        loss_info = "Focal" if use_focal_loss else "BCE"
        print(f"\n[Training Config]")
        print(f"  Loss: {loss_info} + LabelSmooth({label_smoothing}) + SmoothL1")
        print(f"  Loss Weights: Vest={W_VEST}, Hardhat={W_HARDHAT}, Violation={W_VIOLATION}, Risk={W_RISK}")
        print(f"  LR Schedule: CosineAnnealingWarmRestarts (T0={max(5, epochs//4)}, eta_min={lr*0.01:.6f})")
        print(f"  Gradient Clip: {gradient_clip_norm}")
        print(f"  Augmentation: {use_augmentation}")
        print(f"  EMA: {use_ema}")
        print(f"  Early Stopping: patience={early_stopping_patience}")
        print()

    epoch_pbar = tqdm(range(epochs), desc="Training PPEReasonerNet V3", unit="epoch", disable=not show_progress)

    for ep in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_batches = 0
        correct_v, correct_h, correct_c, total_items = 0, 0, 0, 0

        for X_b, y_v, y_h, y_viol, y_r in train_loader:
            X_b = X_b.to(device)
            y_v = y_v.to(device)
            y_h = y_h.to(device)
            y_viol = y_viol.to(device)
            y_r = y_r.to(device)

            # Apply feature augmentation
            X_b_aug = augmentor.augment_batch(X_b)

            optimizer.zero_grad()
            out = model(X_b_aug)

            loss_v = criterion_vest(out["vest_compliance"], y_v)
            loss_h = criterion_hardhat(out["hardhat_compliance"], y_h)
            loss_c = criterion_ce(out["violation_logits"], y_viol)
            loss_r = criterion_mse(out["risk_score"], y_r)

            loss = W_VEST * loss_v + W_HARDHAT * loss_h + W_VIOLATION * loss_c + W_RISK * loss_r
            loss.backward()

            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

            # Update EMA
            if ema is not None:
                ema.update(model)

            train_loss += loss.item()
            train_batches += 1

            # Real-time train accuracy
            with torch.no_grad():
                v_pred = (out["vest_compliance"] >= 0.5).float()
                h_pred = (out["hardhat_compliance"] >= 0.5).float()
                c_pred = out["violation_logits"].argmax(dim=1)
                correct_v += (v_pred == y_v).sum().item()
                correct_h += (h_pred == y_h).sum().item()
                correct_c += (c_pred == y_viol).sum().item()
                total_items += len(y_v)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        avg_train_loss = train_loss / max(1, train_batches)
        vest_acc = (correct_v / max(1, total_items)) * 100.0
        hardhat_acc = (correct_h / max(1, total_items)) * 100.0
        violation_acc = (correct_c / max(1, total_items)) * 100.0

        # Validation phase (use EMA weights if available)
        val_metrics = {}
        if val_loader:
            if ema is not None:
                ema.apply_shadow(model)

            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_correct_v, val_correct_h, val_correct_c, val_items = 0, 0, 0, 0
            val_risk_mae = 0.0

            # Per-class confusion tracking
            val_confusion = torch.zeros(4, 4, dtype=torch.long)

            with torch.no_grad():
                for X_b, y_v, y_h, y_viol, y_r in val_loader:
                    X_b, y_v, y_h, y_viol, y_r = X_b.to(device), y_v.to(device), y_h.to(device), y_viol.to(device), y_r.to(device)
                    out = model(X_b)

                    lv = criterion_vest(out["vest_compliance"], y_v)
                    lh = criterion_hardhat(out["hardhat_compliance"], y_h)
                    lc = criterion_ce(out["violation_logits"], y_viol)
                    lr_val = criterion_mse(out["risk_score"], y_r)
                    val_loss += (W_VEST * lv + W_HARDHAT * lh + W_VIOLATION * lc + W_RISK * lr_val).item()
                    val_batches += 1

                    v_pred = (out["vest_compliance"] >= 0.5).float()
                    h_pred = (out["hardhat_compliance"] >= 0.5).float()
                    c_pred = out["violation_logits"].argmax(dim=1)

                    val_correct_v += (v_pred == y_v).sum().item()
                    val_correct_h += (h_pred == y_h).sum().item()
                    val_correct_c += (c_pred == y_viol).sum().item()
                    val_risk_mae += torch.abs(out["risk_score"] - y_r).sum().item()
                    val_items += len(y_v)

                    # Accumulate confusion matrix
                    for t, p in zip(y_viol.cpu(), c_pred.cpu()):
                        val_confusion[t.item(), p.item()] += 1

            avg_val_loss = val_loss / max(1, val_batches)
            val_metrics = {
                "val_loss": round(avg_val_loss, 4),
                "val_vest_acc": round((val_correct_v / max(1, val_items)) * 100.0, 2),
                "val_hardhat_acc": round((val_correct_h / max(1, val_items)) * 100.0, 2),
                "val_violation_acc": round((val_correct_c / max(1, val_items)) * 100.0, 2),
                "val_risk_mae": round(val_risk_mae / max(1, val_items), 4),
            }

            # Best model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = ep + 1
                patience_counter = 0

                # Save best checkpoint
                torch.save({
                    "state_dict": model.state_dict(),
                    "input_dim": 16,
                    "hidden_dim": 64,
                    "device": str(device),
                    "epochs": ep + 1,
                    "val_loss": avg_val_loss,
                    "version": "V3",
                    "architecture": "PPEReasonerNet_V3_DeepResidual"
                }, output_path)
            else:
                patience_counter += 1

            if ema is not None:
                ema.restore(model)

        history.append({
            "epoch": ep + 1,
            "train_loss": round(avg_train_loss, 4),
            "vest_acc": round(vest_acc, 2),
            "hardhat_acc": round(hardhat_acc, 2),
            "violation_acc": round(violation_acc, 2),
            "lr": round(current_lr, 6),
            **val_metrics
        })

        postfix = {
            "Loss": f"{avg_train_loss:.3f}",
            "VestAcc": f"{vest_acc:.1f}%",
            "HardhatAcc": f"{hardhat_acc:.1f}%",
            "ViolAcc": f"{violation_acc:.1f}%",
            "LR": f"{current_lr:.5f}"
        }
        if val_metrics:
            postfix["ValLoss"] = f"{val_metrics['val_loss']:.3f}"
            postfix["ValViolAcc"] = f"{val_metrics['val_violation_acc']:.1f}%"
        epoch_pbar.set_postfix(postfix)

        # Early stopping
        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            if show_progress:
                print(f"\n[Early Stopping] No improvement for {early_stopping_patience} epochs. Best epoch: {best_epoch}")
            break

    train_duration = time.time() - start_time

    # Save final EMA model if using EMA
    if ema is not None:
        ema.apply_shadow(model)
        ema_path = output_path.replace(".pt", "_ema.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": 16,
            "hidden_dim": 64,
            "device": str(device),
            "epochs": epochs,
            "version": "V3_EMA",
            "architecture": "PPEReasonerNet_V3_DeepResidual_EMA"
        }, ema_path)
        ema.restore(model)
        if show_progress:
            print(f"\n[EMA] Saved EMA model weights to {ema_path}")

    # Load best checkpoint for final metrics report
    if os.path.exists(output_path):
        best_ckpt = torch.load(output_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["state_dict"])

    # Final validation pass with best model for confusion matrix
    model.eval()
    final_confusion = torch.zeros(4, 4, dtype=torch.long)
    with torch.no_grad():
        for X_b, y_v, y_h, y_viol, y_r in val_loader:
            X_b = X_b.to(device)
            y_viol = y_viol.to(device)
            out = model(X_b)
            c_pred = out["violation_logits"].argmax(dim=1)
            for t, p in zip(y_viol.cpu(), c_pred.cpu()):
                final_confusion[t.item(), p.item()] += 1

    # Save metrics metadata
    meta_path = output_path.replace(".pt", "_metrics.json")
    final_metrics = history[-1] if history else {}
    summary = {
        "status": "trained",
        "version": "V3",
        "architecture": "PPEReasonerNet_V3_DeepResidual_SE_Attention",
        "epochs": len(history),
        "max_epochs": epochs,
        "best_epoch": best_epoch,
        "total_samples": len(samples),
        "train_samples": train_size,
        "val_samples": val_size,
        "trainable_parameters": param_count,
        "training_time_seconds": round(train_duration, 2),
        "device": str(device),
        "device_desc": device_desc,
        "output_path": output_path,
        "final_metrics": final_metrics,
        "class_distribution": {class_names.get(k, f"class_{k}"): v for k, v in class_counts.items()},
        "confusion_matrix": final_confusion.tolist(),
        "confusion_labels": ["COMPLIANT", "MISSING_VEST", "MISSING_HARDHAT", "CRITICAL_NO_PPE"],
        "training_config": {
            "focal_loss": use_focal_loss,
            "label_smoothing": label_smoothing,
            "augmentation": use_augmentation,
            "early_stopping_patience": early_stopping_patience,
            "ema": use_ema,
            "gradient_clip_norm": gradient_clip_norm,
            "lr": lr,
            "batch_size": batch_size,
            "loss_weights": {
                "vest": W_VEST,
                "hardhat": W_HARDHAT,
                "violation": W_VIOLATION,
                "risk": W_RISK
            }
        },
        "training_history": history
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print confusion matrix
    if show_progress:
        print(f"\n[Confusion Matrix] (Best Epoch {best_epoch})")
        print(f"{'':>20} | {'COMP':>8} {'M_VEST':>8} {'M_HHAT':>8} {'CRIT':>8}")
        print(f"{'─' * 60}")
        for i, name in enumerate(["COMPLIANT", "MISS_VEST", "MISS_HHAT", "CRITICAL"]):
            row = " ".join(f"{final_confusion[i, j].item():>8}" for j in range(4))
            print(f"{name:>20} | {row}")

    return summary
