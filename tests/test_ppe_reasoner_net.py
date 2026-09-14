"""Unit tests for PPEReasonerNet."""
import torch
from src.risk_model.ppe_reasoner_net import PPEReasonerNet, extract_ppe_neural_features
from src.training.train_ppe_net import train_ppe_model


def test_ppe_reasoner_net_forward():
    worker_box = [100, 50, 200, 300]
    vest_box = [110, 120, 190, 240]
    hardhat_box = [130, 50, 170, 90]

    features = extract_ppe_neural_features(
        worker_box=worker_box,
        vest_box=vest_box,
        hardhat_box=hardhat_box,
        worker_conf=0.95,
        vest_conf=0.88,
        hardhat_conf=0.92
    )

    assert len(features) == 14

    model = PPEReasonerNet(input_dim=14, hidden_dim=64)
    model.eval()

    X = torch.tensor([features], dtype=torch.float32)
    out = model(X)

    assert "vest_compliance" in out
    assert "hardhat_compliance" in out
    assert "violation_logits" in out
    assert "risk_score" in out

    assert out["vest_compliance"].shape == (1, 1)
    assert out["hardhat_compliance"].shape == (1, 1)
    assert out["violation_logits"].shape == (1, 4)
    assert out["risk_score"].shape == (1, 1)


def test_ppe_reasoner_training_loop(tmp_path):
    mock_samples = [
        {
            "features": [0.9, 1.0, 0.85, 0.4, 0.8, 0.3, 1.0, 0.9, 0.2, 0.8, 0.1, 0.5, 0.1, 1.0],
            "vest_label": 1.0,
            "hardhat_label": 1.0,
            "violation_class": 0,
            "risk_score": 0.05
        },
        {
            "features": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0],
            "vest_label": 0.0,
            "hardhat_label": 0.0,
            "violation_class": 3,
            "risk_score": 0.95
        }
    ]

    out_file = str(tmp_path / "test_ppe_model.pt")
    res = train_ppe_model(mock_samples, output_path=out_file, epochs=2)
    assert res["status"] == "trained"
