"""Training Engine for Construction Image Hazard Risk Model."""
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import f1_score, accuracy_score

from src.training.image_hazard_dataset import ImageHazardDataset, ImageHazardMLP
from src.features.image_hazard_features import ImageWorkerHazardFeature
from src.prism.tracker import PrismTracker


def train_image_risk_model(
    features: List[ImageWorkerHazardFeature],
    output_dir: str = "models/image_models",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.003,
    val_split: float = 0.20
) -> Dict[str, Any]:
    """Trains dual-head hazard MLP on extracted image hazard features with validation."""
    if not features:
        raise ValueError("No features provided for training.")

    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Convert to dictionary representation
    samples = []
    for f in features:
        samples.append({
            "feature_vector": f.to_feature_vector(),
            "severity": f.ground_truth_severity,
            "risk_score": f.ground_truth_risk_score
        })

    # Shuffle and split
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split_idx = int(len(samples) * (1.0 - val_split))
    if split_idx == len(samples):
        split_idx = max(1, len(samples) - 1)

    train_samples = [samples[i] for i in indices[:split_idx]]
    val_samples = [samples[i] for i in indices[split_idx:]]

    train_ds = ImageHazardDataset(train_samples)
    val_ds = ImageHazardDataset(val_samples, means=train_ds.means, stds=train_ds.stds)

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)

    # Compute class weights for imbalanced severities
    counts = np.bincount(train_ds.y_sev.numpy(), minlength=4) + 1
    weights = 1.0 / counts
    weights = weights / np.sum(weights) * 4.0
    class_weights = torch.tensor(weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageHazardMLP(input_dim=12, hidden_dim=64, num_classes=4).to(device)
    class_weights = class_weights.to(device)

    criterion_sev = nn.CrossEntropyLoss(weight=class_weights)
    criterion_risk = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_metrics = {}
    best_weights_path = os.path.join(output_dir, "hazard_image_mlp.pt")

    print(f"\n--- Starting Image Hazard Risk Model Training ({epochs} Epochs on {device}) ---")
    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X_b, y_sev_b, y_risk_b in train_loader:
            X_b = X_b.to(device)
            y_sev_b = y_sev_b.to(device)
            y_risk_b = y_risk_b.to(device)

            optimizer.zero_grad()
            logits, pred_risk = model(X_b)
            loss_s = criterion_sev(logits, y_sev_b)
            loss_r = criterion_risk(pred_risk, y_risk_b)
            loss = loss_s + 2.0 * loss_r

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        all_sev_true = []
        all_sev_pred = []
        all_risk_true = []
        all_risk_pred = []

        with torch.no_grad():
            for X_b, y_sev_b, y_risk_b in val_loader:
                X_b = X_b.to(device)
                y_sev_b = y_sev_b.to(device)
                y_risk_b = y_risk_b.to(device)

                logits, pred_risk = model(X_b)
                loss_s = criterion_sev(logits, y_sev_b)
                loss_r = criterion_risk(pred_risk, y_risk_b)
                v_loss = loss_s + 2.0 * loss_r
                val_loss += v_loss.item() * len(X_b)

                all_sev_true.extend(y_sev_b.cpu().numpy())
                all_sev_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_risk_true.extend(y_risk_b.cpu().numpy().flatten())
                all_risk_pred.extend(pred_risk.cpu().numpy().flatten())

        val_loss /= len(val_ds) if len(val_ds) > 0 else 1.0
        val_acc = accuracy_score(all_sev_true, all_sev_pred) if all_sev_true else 0.0
        val_f1 = f1_score(all_sev_true, all_sev_pred, average="macro", zero_division=0) if all_sev_true else 0.0
        risk_mae = float(np.mean(np.abs(np.array(all_risk_true) - np.array(all_risk_pred)))) if all_risk_true else 0.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "macro_f1": round(val_f1, 4),
                "risk_mae": round(risk_mae, 4),
                "epoch": ep
            }
            # Save checkpoint with normalization parameters
            checkpoint = {
                "state_dict": model.state_dict(),
                "means": train_ds.means,
                "stds": train_ds.stds,
                "input_dim": 12,
                "hidden_dim": 64,
                "num_classes": 4,
                "metrics": best_metrics
            }
            torch.save(checkpoint, best_weights_path)

        if ep % 10 == 0 or ep == epochs:
            print(f"Epoch [{ep:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Risk MAE: {risk_mae:.4f}")

    print(f"\n[Training Complete] Best Checkpoint saved to: {best_weights_path}")
    print(f"Best Validation Macro F1: {best_metrics['macro_f1']} | Accuracy: {best_metrics['val_acc']} | MAE: {best_metrics['risk_mae']}")

    # Record run in PRISM tracker
    tracker = PrismTracker()
    tracker.record_run(
        model_version="image_hazard_mlp_v1",
        dataset_version="user_datasets",
        feature_version="f12_spatial_ppe_hazard",
        hyperparameters={"arch": "ImageHazardMLP_64_32", "lr": lr, "batch_size": batch_size, "epochs": epochs},
        metrics=best_metrics,
        diagnosis_note=f"Trained on user image dataset with {len(features)} worker instances. Best Val F1: {best_metrics['macro_f1']}."
    )

    return {
        "model_path": best_weights_path,
        "metrics": best_metrics,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds)
    }
