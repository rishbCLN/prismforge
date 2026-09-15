"""Comprehensive Robustness Tests for PPEReasonerNet V3.

Tests:
  - Forward pass shape verification
  - All 4 violation class detection
  - Hardhat held-in-hand spatial gate (non-compliant)
  - Hardhat at various non-head positions
  - Noise robustness (output stability under small perturbation)
  - Scale invariance (same PPE state at different worker scales)
  - Edge cases (zero-area boxes, full-frame worker, tiny worker)
  - Batch consistency (batch of 1 vs batch of 32 gives same output)
  - Feature interaction layer output shape
  - EMA model functionality
  - Training loop with V3 architecture
  - PPE evaluator integration
"""
import torch
import numpy as np
from src.risk_model.ppe_reasoner_net import PPEReasonerNet, EMAModel, extract_ppe_neural_features
from src.training.train_ppe_net import train_ppe_model, PPEFeatureAugmentor
from src.evaluation.ppe_evaluator import PPEEvaluator


# ===================================================================
# 1. Architecture & Forward Pass Tests
# ===================================================================

def test_ppe_reasoner_net_v3_forward():
    """Validates V3 architecture produces correct output shapes."""
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

    assert len(features) == 16

    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
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


def test_parameter_count_v3():
    """V3 should have more parameters than V2's ~5K params."""
    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # V3 with 3 residual blocks + attention + feature interactions should have significantly more params
    assert param_count > 10000, f"V3 should have >10K params, got {param_count}"


# ===================================================================
# 2. All 4 Violation Class Tests
# ===================================================================

def test_compliant_worker_features():
    """Worker with both vest and hardhat should produce COMPLIANT features."""
    feats = extract_ppe_neural_features(
        worker_box=[0.1, 0.1, 0.3, 0.8],
        vest_box=[0.1, 0.3, 0.3, 0.7],
        hardhat_box=[0.12, 0.1, 0.28, 0.22],
        worker_conf=0.95,
        vest_conf=0.92,
        hardhat_conf=0.90
    )
    assert feats[1] == 1.0, "has_vest_det should be 1.0"
    assert feats[7] == 1.0, "has_hardhat_det should be 1.0"


def test_missing_vest_features():
    """Worker with hardhat but no vest should produce MISSING_VEST features."""
    feats = extract_ppe_neural_features(
        worker_box=[0.1, 0.1, 0.3, 0.8],
        vest_box=None,
        hardhat_box=[0.12, 0.1, 0.28, 0.22],
        worker_conf=0.95,
        vest_conf=0.0,
        hardhat_conf=0.90
    )
    assert feats[1] == 0.0, "has_vest_det should be 0.0"
    assert feats[7] == 1.0, "has_hardhat_det should be 1.0"


def test_missing_hardhat_features():
    """Worker with vest but no hardhat should produce MISSING_HARDHAT features."""
    feats = extract_ppe_neural_features(
        worker_box=[0.1, 0.1, 0.3, 0.8],
        vest_box=[0.1, 0.3, 0.3, 0.7],
        hardhat_box=None,
        worker_conf=0.95,
        vest_conf=0.92,
        hardhat_conf=0.0
    )
    assert feats[1] == 1.0, "has_vest_det should be 1.0"
    assert feats[7] == 0.0, "has_hardhat_det should be 0.0"


def test_critical_no_ppe_features():
    """Worker with neither vest nor hardhat should produce CRITICAL_NO_PPE features."""
    feats = extract_ppe_neural_features(
        worker_box=[0.1, 0.1, 0.3, 0.8],
        vest_box=None,
        hardhat_box=None,
        worker_conf=0.95,
        vest_conf=0.0,
        hardhat_conf=0.0
    )
    assert feats[1] == 0.0, "has_vest_det should be 0.0"
    assert feats[7] == 0.0, "has_hardhat_det should be 0.0"


# ===================================================================
# 3. Spatial Gate / Hardhat Position Tests
# ===================================================================

def test_ppe_reasoner_held_in_hand_is_non_compliant():
    """Validates that a hardhat held in hand (at waist level) is non-compliant (0.0)."""
    worker_box = [100, 50, 200, 300]
    vest_box = [110, 120, 190, 240]
    held_hardhat_box = [110, 200, 170, 250]  # Held at waist/hand, h_iou_head is 0.0

    features = extract_ppe_neural_features(
        worker_box=worker_box,
        vest_box=vest_box,
        hardhat_box=held_hardhat_box,
        worker_conf=0.95,
        vest_conf=0.90,
        hardhat_conf=0.90
    )

    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
    model.eval()

    X = torch.tensor([features], dtype=torch.float32)
    out = model(X)

    # Compliance must be strictly 0.0 due to spatial head gating
    assert float(out["hardhat_compliance"][0, 0].item()) == 0.0
    # Vest should remain compliant
    assert float(out["vest_compliance"][0, 0].item()) > 0.0


def test_hardhat_at_shoulder_non_compliant():
    """Hardhat at shoulder level (not on head) should be non-compliant."""
    worker_box = [100, 50, 200, 400]
    # Shoulder level: about 40% down the worker box
    shoulder_hardhat = [110, 190, 170, 230]

    features = extract_ppe_neural_features(
        worker_box=worker_box,
        vest_box=None,
        hardhat_box=shoulder_hardhat,
        worker_conf=0.95,
        vest_conf=0.0,
        hardhat_conf=0.90
    )

    # has_hardhat_det should be 0.0 (not on head)
    assert features[7] == 0.0, "Hardhat at shoulder should have has_hardhat_det=0.0"


def test_hardhat_on_ground_non_compliant():
    """Hardhat on ground near worker should be non-compliant."""
    worker_box = [100, 50, 200, 400]
    # Ground level: at the bottom of the worker or below
    ground_hardhat = [120, 380, 180, 420]

    features = extract_ppe_neural_features(
        worker_box=worker_box,
        vest_box=None,
        hardhat_box=ground_hardhat,
        worker_conf=0.95,
        vest_conf=0.0,
        hardhat_conf=0.90
    )

    # has_hardhat_det should be 0.0 (not on head)
    assert features[7] == 0.0, "Hardhat on ground should have has_hardhat_det=0.0"


def test_hardhat_properly_on_head_is_compliant():
    """Hardhat properly positioned on top of head should be compliant."""
    worker_box = [100, 50, 200, 400]
    # Properly on head: at the top of the worker box
    head_hardhat = [110, 50, 190, 100]

    features = extract_ppe_neural_features(
        worker_box=worker_box,
        vest_box=None,
        hardhat_box=head_hardhat,
        worker_conf=0.95,
        vest_conf=0.0,
        hardhat_conf=0.90
    )

    # has_hardhat_det should be 1.0 (on head)
    assert features[7] == 1.0, "Hardhat properly on head should have has_hardhat_det=1.0"
    assert features[10] > 0.06, "h_iou_head should be above threshold"


# ===================================================================
# 4. Noise Robustness Tests
# ===================================================================

def test_noise_robustness_output_stability():
    """Model output should not flip classification under ±0.01 input perturbation."""
    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
    model.eval()

    # Create a clear-cut compliant worker
    feats = extract_ppe_neural_features(
        worker_box=[0.1, 0.1, 0.3, 0.8],
        vest_box=[0.1, 0.3, 0.3, 0.7],
        hardhat_box=[0.12, 0.1, 0.28, 0.22],
        worker_conf=0.95,
        vest_conf=0.92,
        hardhat_conf=0.90
    )

    X = torch.tensor([feats], dtype=torch.float32)
    with torch.no_grad():
        out_clean = model(X)
        clean_viol = out_clean["violation_logits"].argmax(dim=1).item()

    # Run 20 noisy passes with small perturbation
    flips = 0
    for _ in range(20):
        noise = torch.randn_like(X) * 0.01
        X_noisy = X + noise
        with torch.no_grad():
            out_noisy = model(X_noisy)
            noisy_viol = out_noisy["violation_logits"].argmax(dim=1).item()
        if noisy_viol != clean_viol:
            flips += 1

    # Allow at most 2 flips out of 20 (10% instability tolerance)
    assert flips <= 2, f"Model flipped {flips}/20 times under ±0.01 noise — too unstable"


# ===================================================================
# 5. Scale Invariance Tests
# ===================================================================

def test_scale_invariance_same_ppe_state():
    """Same PPE state at different worker scales should give consistent detection."""
    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
    model.eval()

    # Small worker (far from camera)
    feats_small = extract_ppe_neural_features(
        worker_box=[0.4, 0.4, 0.5, 0.6],
        vest_box=[0.4, 0.45, 0.5, 0.55],
        hardhat_box=[0.42, 0.4, 0.48, 0.44],
        worker_conf=0.80,
        vest_conf=0.85,
        hardhat_conf=0.85
    )

    # Large worker (close to camera)
    feats_large = extract_ppe_neural_features(
        worker_box=[0.05, 0.05, 0.95, 0.95],
        vest_box=[0.05, 0.35, 0.95, 0.75],
        hardhat_box=[0.1, 0.05, 0.9, 0.25],
        worker_conf=0.95,
        vest_conf=0.92,
        hardhat_conf=0.92
    )

    X = torch.tensor([feats_small, feats_large], dtype=torch.float32)
    with torch.no_grad():
        out = model(X)

    # Both should detect vest and hardhat as present
    vest_small = out["vest_compliance"][0, 0].item()
    vest_large = out["vest_compliance"][1, 0].item()
    # Both should be on the same side of 0.5
    assert (vest_small >= 0.5) == (vest_large >= 0.5) or abs(vest_small - vest_large) < 0.3, \
        f"Vest detection inconsistent across scales: small={vest_small:.3f}, large={vest_large:.3f}"


# ===================================================================
# 6. Edge Case Tests
# ===================================================================

def test_edge_case_tiny_worker():
    """Extremely small worker box should not crash."""
    feats = extract_ppe_neural_features(
        worker_box=[0.5, 0.5, 0.501, 0.501],
        vest_box=None,
        hardhat_box=None,
        worker_conf=0.3,
        vest_conf=0.0,
        hardhat_conf=0.0
    )
    assert len(feats) == 16
    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
    model.eval()
    X = torch.tensor([feats], dtype=torch.float32)
    out = model(X)
    # Should not produce NaN
    for key, val in out.items():
        assert not torch.isnan(val).any(), f"NaN detected in {key} for tiny worker"


def test_edge_case_full_frame_worker():
    """Worker filling entire frame should not crash."""
    feats = extract_ppe_neural_features(
        worker_box=[0.0, 0.0, 1.0, 1.0],
        vest_box=[0.0, 0.3, 1.0, 0.8],
        hardhat_box=[0.1, 0.0, 0.9, 0.2],
        worker_conf=0.99,
        vest_conf=0.95,
        hardhat_conf=0.95
    )
    assert len(feats) == 16
    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
    model.eval()
    X = torch.tensor([feats], dtype=torch.float32)
    out = model(X)
    for key, val in out.items():
        assert not torch.isnan(val).any(), f"NaN detected in {key} for full-frame worker"


def test_edge_case_zero_confidence():
    """Zero confidence worker should not crash."""
    feats = extract_ppe_neural_features(
        worker_box=[0.1, 0.1, 0.3, 0.8],
        vest_box=None,
        hardhat_box=None,
        worker_conf=0.0,
        vest_conf=0.0,
        hardhat_conf=0.0
    )
    assert len(feats) == 16
    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
    model.eval()
    X = torch.tensor([feats], dtype=torch.float32)
    out = model(X)
    for key, val in out.items():
        assert not torch.isnan(val).any(), f"NaN detected in {key} for zero-conf worker"


# ===================================================================
# 7. Batch Consistency Tests
# ===================================================================

def test_batch_consistency():
    """Same sample in batch of 1 vs batch of 8 should give identical outputs."""
    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
    model.eval()

    feats = extract_ppe_neural_features(
        worker_box=[0.1, 0.1, 0.3, 0.8],
        vest_box=[0.1, 0.3, 0.3, 0.7],
        hardhat_box=[0.12, 0.1, 0.28, 0.22],
        worker_conf=0.95,
        vest_conf=0.92,
        hardhat_conf=0.90
    )

    X_single = torch.tensor([feats], dtype=torch.float32)
    X_batch = torch.tensor([feats] * 8, dtype=torch.float32)

    with torch.no_grad():
        out_single = model(X_single)
        out_batch = model(X_batch)

    for key in ["vest_compliance", "hardhat_compliance", "risk_score"]:
        val_single = out_single[key][0].item()
        val_batch = out_batch[key][0].item()
        assert abs(val_single - val_batch) < 1e-5, \
            f"Batch inconsistency in {key}: single={val_single}, batch={val_batch}"

    viol_single = out_single["violation_logits"][0].tolist()
    viol_batch = out_batch["violation_logits"][0].tolist()
    for i, (vs, vb) in enumerate(zip(viol_single, viol_batch)):
        assert abs(vs - vb) < 1e-5, \
            f"Batch inconsistency in violation_logits[{i}]: single={vs}, batch={vb}"


# ===================================================================
# 8. EMA Model Tests
# ===================================================================

def test_ema_model_apply_and_restore():
    """EMA shadow weights should be applied and restored correctly."""
    model = PPEReasonerNet(input_dim=16, hidden_dim=64)
    ema = EMAModel(model, decay=0.999)

    # Simulate a few update steps
    for _ in range(5):
        for param in model.parameters():
            param.data.add_(torch.randn_like(param.data) * 0.01)
        ema.update(model)

    # Save original params
    original_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    # Apply shadow
    ema.apply_shadow(model)

    # Params should differ from original
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(param.data, original_params[name]), \
                f"EMA shadow should differ from original for {name}"

    # Restore
    ema.restore(model)

    # Params should be back to original
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert torch.allclose(param.data, original_params[name]), \
                f"EMA restore should match original for {name}"


# ===================================================================
# 9. Training Pipeline Tests
# ===================================================================

def test_ppe_v3_training_loop(tmp_path):
    """Test that V3 training loop runs end-to-end with all hardening features."""
    mock_samples = [
        {
            "features": [0.9, 1.0, 0.85, 0.4, 0.8, 0.7, 0.3, 1.0, 0.9, 0.2, 0.8, 0.1, 0.5, 1.0, 0.1, 1.0],
            "vest_label": 1.0,
            "hardhat_label": 1.0,
            "violation_class": 0,
            "risk_score": 0.05
        },
        {
            "features": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.1, 0.0],
            "vest_label": 0.0,
            "hardhat_label": 0.0,
            "violation_class": 3,
            "risk_score": 0.95
        },
        {
            "features": [0.9, 1.0, 0.85, 0.4, 0.8, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.1, 0.5],
            "vest_label": 1.0,
            "hardhat_label": 0.0,
            "violation_class": 2,
            "risk_score": 0.55
        },
        {
            "features": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.2, 0.8, 0.1, 0.3, 0.0, 0.1, 0.5],
            "vest_label": 0.0,
            "hardhat_label": 1.0,
            "violation_class": 1,
            "risk_score": 0.45
        },
    ] * 10  # 40 samples total

    out_file = str(tmp_path / "test_ppe_v3.pt")
    res = train_ppe_model(
        mock_samples,
        output_path=out_file,
        epochs=3,
        batch_size=16,
        use_focal_loss=True,
        label_smoothing=0.05,
        use_augmentation=True,
        early_stopping_patience=0,  # Disable for short test
        use_ema=True,
        gradient_clip_norm=1.0,
        show_progress=False,
    )

    assert res["status"] == "trained"
    assert res["version"] == "V3"
    assert "confusion_matrix" in res
    assert "class_distribution" in res
    assert "training_config" in res
    assert res["training_config"]["focal_loss"] is True
    assert res["training_config"]["ema"] is True


# ===================================================================
# 10. Feature Augmentor Tests
# ===================================================================

def test_augmentor_shape_preserved():
    """Feature augmentor should preserve tensor shape."""
    aug = PPEFeatureAugmentor(enabled=True)
    X = torch.randn(32, 16)
    X_aug = aug.augment_batch(X)
    assert X_aug.shape == X.shape


def test_augmentor_disabled():
    """Disabled augmentor should return identical tensor."""
    aug = PPEFeatureAugmentor(enabled=False)
    X = torch.randn(32, 16)
    X_aug = aug.augment_batch(X)
    assert torch.allclose(X, X_aug)


# ===================================================================
# 11. PPE Evaluator Tests
# ===================================================================

def test_ppe_evaluator_basic():
    """PPE evaluator should produce comprehensive metrics."""
    y_true_viol = [0, 1, 2, 3, 0, 0, 1, 3]
    y_pred_viol = [0, 1, 2, 3, 0, 1, 1, 3]
    y_true_vest = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    y_pred_vest = [0.9, 0.1, 0.8, 0.15, 0.95, 0.4, 0.1, 0.05]
    y_true_hh = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    y_pred_hh = [0.85, 0.9, 0.1, 0.05, 0.92, 0.88, 0.95, 0.1]
    y_true_risk = [0.05, 0.45, 0.55, 0.90, 0.05, 0.05, 0.45, 0.90]
    y_pred_risk = [0.06, 0.42, 0.52, 0.88, 0.04, 0.35, 0.44, 0.91]

    results = PPEEvaluator.evaluate(
        y_true_violation=y_true_viol,
        y_pred_violation=y_pred_viol,
        y_true_vest=y_true_vest,
        y_pred_vest=y_pred_vest,
        y_true_hardhat=y_true_hh,
        y_pred_hardhat=y_pred_hh,
        y_true_risk=y_true_risk,
        y_pred_risk=y_pred_risk,
    )

    assert "violation_classification" in results
    assert "vest_compliance" in results
    assert "hardhat_compliance" in results
    assert "safety_cost" in results
    assert "risk_score" in results
    assert "boundary_analysis" in results
    assert "summary" in results

    # Check confusion matrix shape
    cm = results["violation_classification"]["confusion_matrix"]
    assert len(cm) == 4
    assert len(cm[0]) == 4

    # Format report should not crash
    report = PPEEvaluator.format_report(results)
    assert len(report) > 100


def test_ppe_evaluator_perfect_predictions():
    """Perfect predictions should yield zero cost and 100% accuracy."""
    n = 20
    y_true = list(range(4)) * 5
    results = PPEEvaluator.evaluate(
        y_true_violation=y_true,
        y_pred_violation=y_true.copy(),
        y_true_vest=[1.0 if v in [0, 2] else 0.0 for v in y_true],
        y_pred_vest=[0.95 if v in [0, 2] else 0.05 for v in y_true],
        y_true_hardhat=[1.0 if v in [0, 1] else 0.0 for v in y_true],
        y_pred_hardhat=[0.95 if v in [0, 1] else 0.05 for v in y_true],
        y_true_risk=[0.05, 0.45, 0.55, 0.90] * 5,
        y_pred_risk=[0.05, 0.45, 0.55, 0.90] * 5,
    )

    assert results["violation_classification"]["overall_accuracy"] == 1.0
    assert results["safety_cost"]["avg_asymmetric_cost_per_sample"] == 0.0
    assert results["safety_cost"]["critical_violation_miss_rate"] == 0.0
