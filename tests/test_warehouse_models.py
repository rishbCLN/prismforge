"""Tests for Warehouse Risk Models V0, V1, V2, V3."""
import pytest
from src.risk_model.warehouse_models import (
    WarehouseV0Baseline,
    WarehouseV1Context,
    WarehouseV2Temporal,
    WarehouseV3Learned
)


@pytest.fixture
def dummy_features():
    # 16 features: default safe handling
    return [0.0] * 16


def test_v0_baseline_predictions(dummy_features):
    model = WarehouseV0Baseline()
    out = model.predict(dummy_features)
    assert out.severity == "LOW"
    assert out.intervention_priority == "P3_INFORMATIONAL"

    # Drop violation: index 0
    drop_feats = list(dummy_features)
    drop_feats[0] = 0.8
    out_drop = model.predict(drop_feats)
    assert out_drop.severity in ["HIGH", "CRITICAL"]
    assert "HIGH_DROP_VELOCITY" in out_drop.contributing_factors


def test_v1_context_predictions(dummy_features):
    model = WarehouseV1Context()
    out = model.predict(dummy_features)
    assert out.severity == "LOW"

    # Drag with trolley: index 5=drag, index 6=trolley_iou
    trolley_drag = list(dummy_features)
    trolley_drag[4] = 1.0 # floor contact
    trolley_drag[5] = 0.6 # drag vel
    trolley_drag[6] = 0.8 # trolley overlap
    out_trolley = model.predict(trolley_drag)
    # Trolley presence should suppress high risk
    assert out_trolley.severity in ["LOW", "MEDIUM"]


def test_v2_temporal_persistence(dummy_features):
    model = WarehouseV2Temporal(persistence_frames=3)

    drag_feats = list(dummy_features)
    drag_feats[4] = 1.0 # floor contact
    drag_feats[5] = 0.5 # drag vel
    drag_feats[6] = 0.0 # no trolley

    # Frame 1: transient, not yet verified
    out1 = model.predict(drag_feats)
    assert "SUSTAINED_FLOOR_DRAGGING" not in out1.contributing_factors

    # Frames 2 and 3
    model.predict(drag_feats)
    out3 = model.predict(drag_feats)
    assert "SUSTAINED_FLOOR_DRAGGING" in out3.contributing_factors
    assert out3.severity in ["HIGH", "CRITICAL"]


def test_v3_learned_mlp_schema(dummy_features):
    model = WarehouseV3Learned("models/v3/warehouse_risk_mlp.pt")
    out = model.predict(dummy_features)

    assert 0.0 <= out.risk_score <= 1.0
    assert out.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert out.intervention_priority in ["P1_IMMEDIATE", "P2_CORRECTIVE", "P3_INFORMATIONAL"]
    assert len(out.attribution) == 16
