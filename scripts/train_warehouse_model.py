"""Trains the V3 Dual-Head Warehouse Risk MLP with PyTorch.
Optimizes both continuous risk score regression (MSE Loss) and
discrete severity classification (Cross-Entropy Loss) with class weighting.
Saves model weights to models/v3/warehouse_risk_mlp.pt.
"""
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.risk_model.warehouse_models import WarehouseRiskMLP, SEVERITY_CLASSES


class WarehouseFeatureDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.samples = json.load(f)

        self.features = []
        self.risk_scores = []
        self.severity_indices = []

        sev_map = {name: i for i, name in enumerate(SEVERITY_CLASSES)}

        for s in self.samples:
            self.features.append(s["features"])
            self.risk_scores.append(s["target_risk_score"])
            sev_name = s.get("target_severity", "LOW")
            self.severity_indices.append(sev_map.get(sev_name, 0))

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.risk_scores = torch.tensor(self.risk_scores, dtype=torch.float32).unsqueeze(1)
        self.severity_indices = torch.tensor(self.severity_indices, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.features[idx], self.risk_scores[idx], self.severity_indices[idx]


def train_warehouse_v3(
    train_json: str = "data/features/warehouse_train_features.json",
    val_json: str = "data/features/warehouse_val_features.json",
    output_model_path: str = "models/v3/warehouse_risk_mlp.pt",
    epochs: int = 50,
    lr: float = 0.003
):
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    train_ds = WarehouseFeatureDataset(train_json)
    val_ds = WarehouseFeatureDataset(val_json)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WarehouseRiskMLP(in_features=16).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    print(f"Training Warehouse V3 Learned MLP on {len(train_ds)} samples, validating on {len(val_ds)} samples...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for feats, scores, sevs in train_loader:
            feats, scores, sevs = feats.to(device), scores.to(device), sevs.to(device)
            optimizer.zero_grad()

            pred_risk, pred_sev = model(feats)
            loss_reg = mse_criterion(pred_risk, scores)
            loss_cls = ce_criterion(pred_sev, sevs)
            loss = loss_reg + 1.2 * loss_cls

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(feats)

        avg_train_loss = total_train_loss / len(train_ds)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_sev = 0
        with torch.no_grad():
            for feats, scores, sevs in val_loader:
                feats, scores, sevs = feats.to(device), scores.to(device), sevs.to(device)
                pred_risk, pred_sev = model(feats)
                loss_reg = mse_criterion(pred_risk, scores)
                loss_cls = ce_criterion(pred_sev, sevs)
                loss = loss_reg + 1.2 * loss_cls
                total_val_loss += loss.item() * len(feats)

                preds = torch.argmax(pred_sev, dim=-1)
                correct_sev += (preds == sevs).sum().item()

        avg_val_loss = total_val_loss / len(val_ds)
        val_acc = (correct_sev / len(val_ds)) * 100.0

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Sev Acc: {val_acc:.1f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_model_path)

    print(f"\nTrained model successfully saved to: {output_model_path}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_warehouse_v3()
