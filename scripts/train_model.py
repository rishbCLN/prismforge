"""Model Training Pipeline Script.
Trains HazardMesh risk models (specifically V3 learned PyTorch MLP)
on structured spatio-temporal features with held-out validation.
Usage:
    python scripts/train_model.py [--version v3] [--features-dir data/features] [--epochs 60] [--lr 0.005]
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.features.feature_extractor import HazardFeatures
from src.risk_model.v3_learned import V3LearnedRiskModel, RiskMLP


class HazardDataset(Dataset):
    def __init__(self, samples: list, feature_means: np.ndarray, feature_stds: np.ndarray):
        self.severity_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        self.X = []
        self.y_sev = []
        self.y_risk = []

        for s in samples:
            raw_vec = np.array(s["feature_vector"], dtype=np.float32)
            norm_vec = (raw_vec - feature_means) / (feature_stds + 1e-6)
            gt = s["ground_truth"]
            sev_idx = self.severity_map.get(gt["severity"], 0)
            risk_val = float(gt["risk_score"])

            self.X.append(norm_vec)
            self.y_sev.append(sev_idx)
            self.y_risk.append(risk_val)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y_sev = torch.tensor(np.array(self.y_sev), dtype=torch.long)
        self.y_risk = torch.tensor(np.array(self.y_risk), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_sev[idx], self.y_risk[idx]


def train_v3_model(features_dir: str, out_model_dir: str, epochs: int = 60, lr: float = 0.005, batch_size: int = 32):
    train_file = os.path.join(features_dir, "train_features.json")
    val_file = os.path.join(features_dir, "val_features.json")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training features {train_file} not found. Run build_features.py first.")

    with open(train_file, "r") as f:
        train_data = json.load(f)
    with open(val_file, "r") as f:
        val_data = json.load(f)

    train_samples = train_data["samples"]
    val_samples = val_data["samples"]
    print(f"Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    # Compute normalization statistics across training split only
    train_vectors = np.array([s["feature_vector"] for s in train_samples], dtype=np.float32)
    feature_means = np.mean(train_vectors, axis=0)
    feature_stds = np.std(train_vectors, axis=0)
    feature_stds[feature_stds == 0] = 1.0

    train_ds = HazardDataset(train_samples, feature_means, feature_stds)
    val_ds = HazardDataset(val_samples, feature_means, feature_stds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskMLP(input_dim=len(HazardFeatures.FEATURE_NAMES)).to(device)

    # Class-weighted loss to penalize under-predicting dangerous hazards
    class_weights = torch.tensor([1.0, 1.5, 2.0, 3.0], dtype=torch.float32, device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for bx, by_sev, by_risk in train_loader:
            bx = bx.to(device)
            by_sev = by_sev.to(device)
            by_risk = by_risk.to(device)

            optimizer.zero_grad()
            sev_logits, risk_pred = model(bx)

            loss_sev = ce_loss_fn(sev_logits, by_sev)
            loss_risk = mse_loss_fn(risk_pred, by_risk)
            loss = loss_sev + 2.0 * loss_risk

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(bx)

        train_loss = total_train_loss / len(train_ds)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for bx, by_sev, by_risk in val_loader:
                bx = bx.to(device)
                by_sev = by_sev.to(device)
                by_risk = by_risk.to(device)

                sev_logits, risk_pred = model(bx)
                loss_sev = ce_loss_fn(sev_logits, by_sev)
                loss_risk = mse_loss_fn(risk_pred, by_risk)
                loss = loss_sev + 2.0 * loss_risk

                total_val_loss += loss.item() * len(bx)
                preds = torch.argmax(sev_logits, dim=1)
                correct += (preds == by_sev).sum().item()

        val_loss = total_val_loss / len(val_ds)
        val_acc = correct / len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "state_dict": model.state_dict(),
                "feature_means": feature_means.tolist(),
                "feature_stds": feature_stds.tolist(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss
            }

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

    # Save checkpoint
    os.makedirs(out_model_dir, exist_ok=True)
    save_path = os.path.join(out_model_dir, "hazard_risk_mlp.pt")
    torch.save(best_state, save_path)
    print(f"Successfully trained V3 model. Saved best checkpoint to {save_path} (Val Acc: {best_state['val_acc']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Train HazardMesh risk model")
    parser.add_argument("--version", default="v3", choices=["v3"], help="Model version to train")
    parser.add_argument("--features-dir", default="data/features", help="Features directory")
    parser.add_argument("--out-dir", default="models/v3", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    args = parser.parse_args()

    train_v3_model(
        features_dir=args.features_dir,
        out_model_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
