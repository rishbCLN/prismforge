"""Robustness tests for LargeVehicleProximityNet and proximity pipeline.

Run with:
    python -m pytest tests/test_proximity_net.py -v
"""

import math
import random
import pytest
import torch

from src.risk_model.large_vehicle_proximity_net import (
    LargeVehicleProximityNet, EMAModel, FEATURE_DIM, PROXIMITY_CLASSES,
    SqueezeExcitation, ResidualBlock, SpatialSelfAttention,
)
from src.features.proximity_features import (
    extract_proximity_feature, derive_proximity_label,
    extract_proximity_samples, samples_to_tensors,
    DANGER_THRESH, SUPERVISED_THRESH, TOO_FAR_THRESH,
    PROXIMITY_CLASS_NAMES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    m = LargeVehicleProximityNet()
    m.eval()
    return m


@pytest.fixture
def batch_feat():
    return torch.rand(8, FEATURE_DIM)


def _rand_bbox():
    cx, cy = random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)
    w = random.uniform(0.05, 0.3)
    h = random.uniform(0.05, 0.3)
    return [cx, cy, w, h]


def _vehicle_det(cls="truck"):
    return {"bbox_norm": _rand_bbox(), "class": cls}


def _simple_manifest(n_images=10):
    """Create a minimal ingestor manifest for testing."""
    entries = []
    for _ in range(n_images):
        entries.append({
            "vehicles": [{"class": "tractor", "bbox_norm": _rand_bbox()},
                         {"class": "truck",   "bbox_norm": _rand_bbox()}],
            "persons":  [{"bbox_norm": _rand_bbox()}, {"bbox_norm": _rand_bbox()}],
            "dataset_source": "test",
        })
    return {"entries": entries}


# ---------------------------------------------------------------------------
# 1. Architecture & shape tests
# ---------------------------------------------------------------------------

class TestModelArchitecture:

    def test_output_keys(self, model, batch_feat):
        out = model(batch_feat)
        assert "proximity_score" in out
        assert "proximity_class" in out
        assert "vehicle_class" in out

    def test_proximity_score_shape(self, model, batch_feat):
        out = model(batch_feat)
        assert out["proximity_score"].shape == (8,), f"Got {out['proximity_score'].shape}"

    def test_proximity_class_shape(self, model, batch_feat):
        out = model(batch_feat)
        assert out["proximity_class"].shape == (8, PROXIMITY_CLASSES)

    def test_vehicle_class_shape(self, model, batch_feat):
        out = model(batch_feat)
        assert out["vehicle_class"].shape == (8,)

    def test_proximity_score_range(self, model, batch_feat):
        out = model(batch_feat)
        scores = out["proximity_score"]
        assert (scores >= 0.0).all(), "Scores below 0"
        assert (scores <= 1.0).all(), "Scores above 1"

    def test_parameter_count(self, model):
        n = model.count_parameters()
        assert n > 10_000, f"Model suspiciously small: {n} params"
        assert n < 5_000_000, f"Model unexpectedly large: {n} params"

    def test_single_sample(self, model):
        feat = torch.rand(1, FEATURE_DIM)
        out = model(feat)
        assert out["proximity_score"].shape == (1,)
        assert out["proximity_class"].shape == (1, PROXIMITY_CLASSES)

    def test_large_batch(self, model):
        feat = torch.rand(256, FEATURE_DIM)
        out = model(feat)
        assert out["proximity_score"].shape == (256,)

    def test_no_nan_in_output(self, model):
        feat = torch.rand(16, FEATURE_DIM)
        out = model(feat)
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in {k}"
            assert not torch.isinf(v).any(), f"Inf in {k}"

    def test_grad_flows(self):
        model = LargeVehicleProximityNet()
        model.train()
        feat = torch.rand(4, FEATURE_DIM, requires_grad=False)
        out = model(feat)
        loss = (out["proximity_score"].mean()
                + out["proximity_class"].mean()
                + out["vehicle_class"].mean())
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# 2. Sub-module tests
# ---------------------------------------------------------------------------

class TestSubModules:

    def test_squeeze_excitation_shape(self):
        se = SqueezeExcitation(64)
        x = torch.rand(4, 64)
        out = se(x)
        assert out.shape == (4, 64)

    def test_squeeze_excitation_range(self):
        """SE output should stay in a reasonable magnitude range."""
        se = SqueezeExcitation(32)
        x = torch.rand(8, 32)
        out = se(x)
        assert not torch.isnan(out).any()

    def test_residual_block_same_dim(self):
        blk = ResidualBlock(64, 64)
        x = torch.rand(4, 64)
        assert blk(x).shape == (4, 64)

    def test_residual_block_dim_change(self):
        blk = ResidualBlock(20, 64)
        x = torch.rand(4, 20)
        assert blk(x).shape == (4, 64)

    def test_self_attention_shape(self):
        attn = SpatialSelfAttention(64, num_heads=4)
        x = torch.rand(8, 64)
        assert attn(x).shape == (8, 64)

    def test_self_attention_no_nan(self):
        attn = SpatialSelfAttention(64, num_heads=4)
        x = torch.rand(16, 64)
        assert not torch.isnan(attn(x)).any()


# ---------------------------------------------------------------------------
# 3. Feature extraction tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction:

    def test_feature_length(self):
        feat = extract_proximity_feature([0.5, 0.5, 0.1, 0.2], [0.3, 0.3, 0.4, 0.3], "truck")
        assert len(feat) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(feat)}"

    def test_feature_range(self):
        for _ in range(20):
            feat = extract_proximity_feature(_rand_bbox(), _rand_bbox(), "tractor")
            for i, v in enumerate(feat):
                assert -0.1 <= v <= 1.5, f"Feature dim {i} out of range: {v}"

    def test_tractor_class_id(self):
        feat = extract_proximity_feature([0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3], "tractor")
        assert feat[8] == 0.0, "tractor should map to class_id=0"

    def test_truck_class_id(self):
        feat = extract_proximity_feature([0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3], "truck")
        assert feat[8] == 1.0, "truck should map to class_id=1"

    def test_danger_zone_flag(self):
        # Worker very close to vehicle center
        v_bbox = [0.5, 0.5, 0.3, 0.3]
        w_bbox = [0.52, 0.52, 0.05, 0.1]  # nearly same center
        feat = extract_proximity_feature(w_bbox, v_bbox, "truck")
        assert feat[17] == 1.0, "Should be flagged as in danger zone"

    def test_too_far_flag(self):
        # Worker far from vehicle
        v_bbox = [0.1, 0.1, 0.15, 0.15]
        w_bbox = [0.9, 0.9, 0.05, 0.1]
        feat = extract_proximity_feature(w_bbox, v_bbox, "truck")
        assert feat[19] == 1.0, "Should be flagged as too far"


# ---------------------------------------------------------------------------
# 4. Label derivation tests
# ---------------------------------------------------------------------------

class TestLabelDerivation:

    def test_danger_class_when_overlapping(self):
        v = [0.5, 0.5, 0.3, 0.3]
        w = [0.51, 0.51, 0.04, 0.08]  # inside vehicle bbox
        cls, risk = derive_proximity_label(w, v)
        assert cls == 2, f"Expected DANGER (2), got {cls}"
        assert risk >= 0.75

    def test_too_far_class(self):
        v = [0.1, 0.1, 0.1, 0.1]
        w = [0.95, 0.95, 0.05, 0.1]
        cls, risk = derive_proximity_label(w, v)
        assert cls == 3, f"Expected TOO_FAR (3), got {cls}"

    def test_supervised_class(self):
        # Place worker at exactly supervised zone distance
        v = [0.5, 0.5, 0.1, 0.1]
        d = (DANGER_THRESH + SUPERVISED_THRESH) / 2.0
        w = [0.5 + d, 0.5, 0.04, 0.08]
        cls, risk = derive_proximity_label(w, v)
        assert cls == 1, f"Expected SUPERVISED (1), got {cls} at dist={d:.3f}"

    def test_risk_score_range(self):
        for _ in range(50):
            cls, risk = derive_proximity_label(_rand_bbox(), _rand_bbox())
            assert 0.0 <= risk <= 1.0, f"Risk out of range: {risk}"

    def test_class_is_valid(self):
        for _ in range(50):
            cls, _ = derive_proximity_label(_rand_bbox(), _rand_bbox())
            assert cls in (0, 1, 2, 3), f"Invalid class: {cls}"


# ---------------------------------------------------------------------------
# 5. Sample extraction tests
# ---------------------------------------------------------------------------

class TestSampleExtraction:

    def test_extracts_samples(self):
        manifest = _simple_manifest(5)
        samples = extract_proximity_samples(manifest, augment=False)
        assert len(samples) > 0

    def test_sample_has_required_keys(self):
        manifest = _simple_manifest(3)
        samples = extract_proximity_samples(manifest, augment=False)
        for s in samples[:5]:
            assert "features" in s
            assert "proximity_class" in s
            assert "risk_score" in s
            assert "vehicle_class_id" in s

    def test_sample_feature_length(self):
        manifest = _simple_manifest(3)
        samples = extract_proximity_samples(manifest, augment=False)
        for s in samples[:5]:
            assert len(s["features"]) == FEATURE_DIM

    def test_augmentation_increases_samples(self):
        manifest = _simple_manifest(5)
        base = extract_proximity_samples(manifest, augment=False)
        aug  = extract_proximity_samples(manifest, augment=True)
        assert len(aug) > len(base), "Augmentation should produce more samples"

    def test_no_vehicle_entries_skipped(self):
        manifest = {"entries": [{"vehicles": [], "persons": [{"bbox_norm": _rand_bbox()}], "dataset_source": "test"}]}
        samples = extract_proximity_samples(manifest, augment=False)
        assert len(samples) == 0, "Entries with no vehicles should produce no samples"

    def test_tensor_conversion(self):
        manifest = _simple_manifest(5)
        samples = extract_proximity_samples(manifest, augment=False)
        feats, prox, risk, v_cls = samples_to_tensors(samples)
        assert feats.shape[1] == FEATURE_DIM
        assert feats.shape[0] == prox.shape[0]
        assert prox.dtype == torch.long
        assert risk.dtype == torch.float32


# ---------------------------------------------------------------------------
# 6. EMA tests
# ---------------------------------------------------------------------------

class TestEMAModel:

    def test_ema_updates(self):
        model = LargeVehicleProximityNet()
        ema = EMAModel(model, decay=0.9)
        # Do a fake gradient update
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        feat = torch.rand(4, FEATURE_DIM)
        out = model(feat)
        out["proximity_score"].mean().backward()
        opt.step()
        ema.update(model)
        # EMA shadow should differ from initial
        assert len(ema.shadow) > 0

    def test_ema_apply_restore(self):
        model = LargeVehicleProximityNet()
        ema = EMAModel(model, decay=0.99)
        # Record original first weight
        first_key = list(dict(model.named_parameters()).keys())[0]
        orig_val = dict(model.named_parameters())[first_key].data.clone()
        ema.apply_shadow(model)
        ema.restore(model)
        restored_val = dict(model.named_parameters())[first_key].data
        assert torch.allclose(orig_val, restored_val, atol=1e-6)

    def test_ema_state_dict_roundtrip(self):
        model = LargeVehicleProximityNet()
        ema = EMAModel(model, decay=0.999)
        state = ema.state_dict()
        ema2 = EMAModel(model, decay=0.9)
        ema2.load_state_dict(state)
        assert ema2.decay == 0.999


# ---------------------------------------------------------------------------
# 7. Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_worker_on_vehicle_center(self, model):
        # Worker exactly at vehicle center — should be DANGER
        v = [0.5, 0.5, 0.3, 0.3]
        w = [0.5, 0.5, 0.04, 0.08]  # same center
        feat = extract_proximity_feature(w, v, "tractor")
        cls, risk = derive_proximity_label(w, v)
        assert cls == 2
        assert risk >= 0.75

    def test_worker_at_image_boundary(self, model):
        w = [0.01, 0.01, 0.02, 0.04]
        v = [0.5, 0.5, 0.3, 0.3]
        feat = extract_proximity_feature(w, v, "truck")
        assert len(feat) == FEATURE_DIM
        feat_t = torch.tensor([feat], dtype=torch.float32)
        out = model(feat_t)
        assert not torch.isnan(out["proximity_score"]).any()

    def test_zero_size_worker(self, model):
        w = [0.5, 0.5, 0.0, 0.0]
        v = [0.3, 0.3, 0.2, 0.2]
        feat = extract_proximity_feature(w, v, "truck")
        feat_t = torch.tensor([feat], dtype=torch.float32)
        out = model(feat_t)
        assert not torch.isnan(out["proximity_score"]).any()

    def test_deterministic_eval(self, model):
        feat = torch.rand(4, FEATURE_DIM)
        model.eval()
        with torch.no_grad():
            out1 = model(feat)
            out2 = model(feat)
        assert torch.allclose(out1["proximity_score"], out2["proximity_score"])

    def test_unknown_vehicle_class(self):
        feat = extract_proximity_feature(_rand_bbox(), _rand_bbox(), "crane")
        assert feat[8] == 1.0  # unknown classes default to truck (1)

    def test_empty_manifest(self):
        manifest = {"entries": []}
        samples = extract_proximity_samples(manifest, augment=False)
        assert samples == []
