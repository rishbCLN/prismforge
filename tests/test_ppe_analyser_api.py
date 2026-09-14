"""Unit tests for PPE Analyser API Endpoints."""
import os
import cv2
import numpy as np
from fastapi.testclient import TestClient
from web.server import app

client = TestClient(app)


def test_ppe_info_endpoint():
    res = client.get("/api/ppe/info")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert "files_needed" in data
    assert len(data["files_needed"]) >= 4
    assert data["weights_path"] == "models/ppe_reasoner.pt"


def test_ppe_samples_endpoint():
    res = client.get("/api/ppe/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data


def test_ppe_analyze_upload():
    # Generate mock 300x300 BGR test image with yellow helmet-like region and orange vest-like region
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Head region (yellow)
    img[20:70, 100:160] = [0, 255, 255]
    # Torso region (orange)
    img[80:180, 80:180] = [0, 140, 255]

    _, encoded = cv2.imencode(".jpg", img)
    img_bytes = encoded.tobytes()

    response = client.post(
        "/api/ppe/analyze",
        files={"file": ("test_worker.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "summary_metrics" in data
    assert "workers" in data
    assert "annotated_image_base64" in data
    assert data["annotated_image_base64"].startswith("data:image/jpeg;base64,")
    assert "files_needed_by_analyser" in data
