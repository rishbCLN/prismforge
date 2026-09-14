"""Training Pipeline for PPEReasonerNet with Progress Bars, GPU Acceleration, and Real Dataset Ingestion."""
import os
import json
import time
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm

from src.risk_model.ppe_reasoner_net import PPEReasonerNet, extract_ppe_neural_features, _calc_iou


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


def extract_ppe_training_samples(manifest: Dict[str, Any], show_progress: bool = True) -> List[Dict[str, Any]]:
    """Extracts 14-dim spatial features and multi-task supervision labels from ingested images."""
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

    return samples


def train_ppe_model(
    samples: List[Dict[str, Any]],
    output_path: str = "models/ppe_reasoner.pt",
    epochs: int = 30,
    lr: float = 0.003,
    batch_size: int = 64,
    val_split: float = 0.2,
    force_gpu: bool = False,
    device_name: Optional[str] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    """Trains PPEReasonerNet with real-time tqdm progress bars and validation tracking."""
    if not samples:
        raise ValueError("Cannot train PPEReasonerNet: 0 samples provided.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device, device_desc = get_optimal_device(force_gpu=force_gpu, requested_device=device_name)
    if show_progress:
        print(f"\n[Compute Device] {device_desc}")

    dataset = PPEDataset(samples)
    val_size = int(len(dataset) * val_split) if len(dataset) > 5 else 0
    train_size = len(dataset) - val_size

    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=min(batch_size, train_size), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=min(batch_size, val_size), shuffle=False)
    else:
        train_loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        val_loader = None

    model = PPEReasonerNet(input_dim=16, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion_bce = nn.BCELoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    history = []
    start_time = time.time()

    epoch_pbar = tqdm(range(epochs), desc="Training PPEReasonerNet", unit="epoch", disable=not show_progress)

    for ep in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_batches = 0
        correct_v, correct_h, total_items = 0, 0, 0

        for X_b, y_v, y_h, y_viol, y_r in train_loader:
            X_b = X_b.to(device)
            y_v = y_v.to(device)
            y_h = y_h.to(device)
            y_viol = y_viol.to(device)
            y_r = y_r.to(device)

            optimizer.zero_grad()
            out = model(X_b)

            loss_v = criterion_bce(out["vest_compliance"], y_v)
            loss_h = criterion_bce(out["hardhat_compliance"], y_h)
            loss_c = criterion_ce(out["violation_logits"], y_viol)
            loss_r = criterion_mse(out["risk_score"], y_r)

            loss = loss_v + loss_h + loss_c + 2.0 * loss_r
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Real-time train accuracy
            with torch.no_grad():
                v_pred = (out["vest_compliance"] >= 0.5).float()
                h_pred = (out["hardhat_compliance"] >= 0.5).float()
                correct_v += (v_pred == y_v).sum().item()
                correct_h += (h_pred == y_h).sum().item()
                total_items += len(y_v)

        avg_train_loss = train_loss / max(1, train_batches)
        vest_acc = (correct_v / max(1, total_items)) * 100.0
        hardhat_acc = (correct_h / max(1, total_items)) * 100.0

        # Validation phase
        val_metrics = {}
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_correct_v, val_correct_h, val_correct_c, val_items = 0, 0, 0, 0
            val_risk_mae = 0.0

            with torch.no_grad():
                for X_b, y_v, y_h, y_viol, y_r in val_loader:
                    X_b, y_v, y_h, y_viol, y_r = X_b.to(device), y_v.to(device), y_h.to(device), y_viol.to(device), y_r.to(device)
                    out = model(X_b)

                    lv = criterion_bce(out["vest_compliance"], y_v)
                    lh = criterion_bce(out["hardhat_compliance"], y_h)
                    lc = criterion_ce(out["violation_logits"], y_viol)
                    lr_val = criterion_mse(out["risk_score"], y_r)
                    val_loss += (lv + lh + lc + 2.0 * lr_val).item()
                    val_batches += 1

                    v_pred = (out["vest_compliance"] >= 0.5).float()
                    h_pred = (out["hardhat_compliance"] >= 0.5).float()
                    c_pred = out["violation_logits"].argmax(dim=1)

                    val_correct_v += (v_pred == y_v).sum().item()
                    val_correct_h += (h_pred == y_h).sum().item()
                    val_correct_c += (c_pred == y_viol).sum().item()
                    val_risk_mae += torch.abs(out["risk_score"] - y_r).sum().item()
                    val_items += len(y_v)

            val_metrics = {
                "val_loss": round(val_loss / max(1, val_batches), 4),
                "val_vest_acc": round((val_correct_v / max(1, val_items)) * 100.0, 2),
                "val_hardhat_acc": round((val_correct_h / max(1, val_items)) * 100.0, 2),
                "val_violation_acc": round((val_correct_c / max(1, val_items)) * 100.0, 2),
                "val_risk_mae": round(val_risk_mae / max(1, val_items), 4),
            }

        history.append({
            "epoch": ep + 1,
            "train_loss": round(avg_train_loss, 4),
            "vest_acc": round(vest_acc, 2),
            "hardhat_acc": round(hardhat_acc, 2),
            **val_metrics
        })

        postfix = {
            "Loss": f"{avg_train_loss:.3f}",
            "VestAcc": f"{vest_acc:.1f}%",
            "HardhatAcc": f"{hardhat_acc:.1f}%"
        }
        if val_metrics:
            postfix["ValAcc"] = f"{val_metrics['val_violation_acc']:.1f}%"
        epoch_pbar.set_postfix(postfix)

    train_duration = time.time() - start_time

    # Save model weights
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": 16,
        "hidden_dim": 64,
        "device": str(device),
        "epochs": epochs
    }, output_path)

    # Save metrics metadata
    meta_path = output_path.replace(".pt", "_metrics.json")
    final_metrics = history[-1] if history else {}
    summary = {
        "status": "trained",
        "epochs": epochs,
        "total_samples": len(samples),
        "train_samples": train_size,
        "val_samples": val_size,
        "training_time_seconds": round(train_duration, 2),
        "device": str(device),
        "device_desc": device_desc,
        "output_path": output_path,
        "final_metrics": final_metrics
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

