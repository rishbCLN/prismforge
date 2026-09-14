"""Tests for Warehouse Kinematic Feature Extractor."""
import pytest
from src.features.warehouse_features import WarehouseKinematicFeatureExtractor


def test_warehouse_feature_extractor_dimensions():
    extractor = WarehouseKinematicFeatureExtractor()
    carton = {"x": 200, "y": 240, "w": 38, "h": 30}
    operator = {"x": 220, "y": 240}
    trolley = {"x": 200, "y": 250}

    feats = extractor.extract_features(carton, operator, trolley, carton_id=101)
    assert len(feats) == 16
    assert len(WarehouseKinematicFeatureExtractor.FEATURE_NAMES) == 16


def test_warehouse_drop_velocity_calculation():
    extractor = WarehouseKinematicFeatureExtractor()
    # Step 1: at y=200
    extractor.extract_features({"x": 200, "y": 200}, {"x": 220, "y": 200}, carton_id=102)
    # Step 2: sudden descent to y=250
    feats = extractor.extract_features({"x": 200, "y": 250}, {"x": 220, "y": 200}, carton_id=102)

    drop_vel = feats[0]
    assert drop_vel > 0.5, f"Expected elevated drop velocity, got {drop_vel}"


def test_warehouse_outside_staging_zone():
    extractor = WarehouseKinematicFeatureExtractor()
    # Inside zone: (40-260, 180-330)
    feats_inside = extractor.extract_features({"x": 100, "y": 220}, carton_id=103)
    assert feats_inside[9] == 0.0 # is_outside_staging
    assert feats_inside[8] == 0.0 # staging_zone_distance

    # Outside zone: x=500
    feats_outside = extractor.extract_features({"x": 500, "y": 220}, carton_id=104)
    assert feats_outside[9] == 1.0
    assert feats_outside[8] > 0.0


def test_warehouse_drag_floor_contact():
    extractor = WarehouseKinematicFeatureExtractor()
    # On floor: y=290
    feats_floor = extractor.extract_features({"x": 300, "y": 290}, carton_id=105)
    assert feats_floor[4] >= 0.9, f"Expected floor contact ~1.0, got {feats_floor[4]}"

    # In air: y=100
    feats_air = extractor.extract_features({"x": 300, "y": 100}, carton_id=106)
    assert feats_air[4] == 0.0
