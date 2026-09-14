"""Unit tests for Real Image Ingestion, Feature Extraction, and Training Pipeline."""
import os
import tempfile
import json
import pytest
import torch
import numpy as np
from PIL import Image

from src.ingestion.image_dataset_ingestor import ImageDatasetIngestor, BoundingBox
from src.features.image_hazard_features import ImageHazardFeatureExtractor
from src.training.image_hazard_dataset import ImageHazardDataset, ImageHazardMLP


def test_image_dataset_ingestor_yolo_parsing():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create YOLO dataset structure
        images_dir = os.path.join(tmpdir, "images")
        labels_dir = os.path.join(tmpdir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Create dummy image
        img_path = os.path.join(images_dir, "sample1.jpg")
        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        img.save(img_path)

        # Create YOLO label: class 0 (person), center 0.5 0.5, width 0.2, height 0.4
        label_path = os.path.join(labels_dir, "sample1.txt")
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.4\n")

        # Create classes.txt
        classes_path = os.path.join(tmpdir, "classes.txt")
        with open(classes_path, "w") as f:
            f.write("person\nhelmet\nvest\nmachinery\n")

        ingestor = ImageDatasetIngestor(datasets_root=os.path.dirname(tmpdir))
        fmt = ingestor.detect_format(tmpdir)
        assert fmt == "yolo"

        parsed = ingestor.ingest_dataset(tmpdir)
        assert len(parsed) == 1
        assert parsed[0].width == 640
        assert parsed[0].height == 480
        assert len(parsed[0].objects) == 1
        assert parsed[0].objects[0].class_name == "person"
        assert abs(parsed[0].objects[0].x1 - 0.4) < 1e-4
        assert abs(parsed[0].objects[0].x2 - 0.6) < 1e-4


def test_image_hazard_feature_extractor():
    extractor = ImageHazardFeatureExtractor()
    mock_image = {
        "image_id": "test_img",
        "image_path": "/fake/path.jpg",
        "objects": [
            # Person in danger zone near machine
            {"class_name": "person", "x1": 0.4, "y1": 0.3, "x2": 0.6, "y2": 0.8},
            # Missing helmet (no helmet object)
            # Vest present
            {"class_name": "vest", "x1": 0.42, "y1": 0.45, "x2": 0.58, "y2": 0.65},
            # Heavy excavator very close
            {"class_name": "machinery", "x1": 0.55, "y1": 0.2, "x2": 0.9, "y2": 0.9}
        ]
    }

    features = extractor.extract_features_from_image(mock_image)
    assert len(features) == 1
    f = features[0]

    assert f.helmet_missing == 1.0
    assert f.vest_missing == 0.0
    assert f.has_machinery_in_scene == 1.0
    assert f.in_critical_danger_zone == 1.0  # Very close to machine
    assert f.ground_truth_severity == "HIGH"
    assert len(f.to_feature_vector()) == 12


def test_image_hazard_dataset_and_forward():
    samples = [
        {
            "feature_vector": [1.0, 0.0, 1.0, 1.0, 0.1, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.1],
            "severity": "HIGH",
            "risk_score": 0.85
        },
        {
            "feature_vector": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.1, 0.05],
            "severity": "NONE",
            "risk_score": 0.05
        }
    ]

    ds = ImageHazardDataset(samples)
    assert len(ds) == 2

    model = ImageHazardMLP(input_dim=12, hidden_dim=32, num_classes=4)
    X, y_sev, y_risk = ds[0]
    logits, risk_pred = model(X.unsqueeze(0))

    assert logits.shape == (1, 4)
    assert risk_pred.shape == (1, 1)
    assert 0.0 <= risk_pred.item() <= 1.0
