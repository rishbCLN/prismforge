"""Test suite for VideoPPEPipeline: frame dissection, batch inference, timeline generation, and API."""
import os
import cv2
import numpy as np
import pytest

from src.features.video_ppe_pipeline import VideoPPEPipeline
from src.features.ppe_inference import PPEInferenceEngine


@pytest.fixture
def sample_synthetic_video(tmp_path):
    """Generates a small 10-frame synthetic construction safety video for pipeline testing."""
    video_path = str(tmp_path / "test_synth_stream.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    width, height = 320, 240
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame_i in range(12):
        # Create dark background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 35

        # Moving worker: x moves across frames
        wx = 60 + frame_i * 8
        wy = 40

        # Worker body [wx, wy, wx + 80, wy + 160]
        frame[wy:wy + 160, wx:wx + 80] = [120, 120, 120]
        # Face skin [wx + 15:wx + 65, wy + 25:wy + 65]
        frame[wy + 25:wy + 65, wx + 15:wx + 65] = [110, 140, 200]
        # Hardhat on head [wx + 10:wx + 70, wy:wy + 30] (safety yellow)
        frame[wy:wy + 30, wx + 10:wx + 70] = [0, 230, 230]

        writer.write(frame)

    writer.release()
    yield video_path

    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass


def test_video_ppe_pipeline_dissection_and_timeline(sample_synthetic_video):
    """Validates that VideoPPEPipeline extracts frames, computes compliance per frame, and outputs violation timeline."""
    pipeline = VideoPPEPipeline()

    res = pipeline.process_video(
        sample_synthetic_video,
        target_fps=5.0,
        max_frames=15,
        jpeg_quality=60
    )

    assert res["status"] == "success"
    assert "video_metadata" in res
    assert "clip_summary" in res
    assert "violation_timeline" in res
    assert "frames" in res

    meta = res["video_metadata"]
    assert meta["extracted_frames_count"] > 0
    assert meta["duration_sec"] > 0.5
    assert meta["resolution"]["width"] == 320
    assert meta["resolution"]["height"] == 240

    frames = res["frames"]
    assert len(frames) == meta["extracted_frames_count"]

    timeline = res["violation_timeline"]
    assert len(timeline) == len(frames)

    # Check first frame structure
    f0 = frames[0]
    assert "frame_index" in f0
    assert "timestamp_sec" in f0
    assert "time_display" in f0
    assert "annotated_image_base64" in f0
    assert f0["annotated_image_base64"].startswith("data:image/jpeg;base64,")
    assert "workers" in f0
    assert "summary_metrics" in f0

    # Check timeline tick format
    t0 = timeline[0]
    assert "frame_index" in t0
    assert "status_code" in t0
    assert t0["status_code"] in (0, 1, 2)
    assert t0["tick_color"] in ("emerald", "amber", "red")


def test_video_ppe_pipeline_missing_file():
    """Validates that passing a non-existent video path raises FileNotFoundError."""
    pipeline = VideoPPEPipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.process_video("non_existent_stream_xyz.mp4")


def test_api_analyze_video_endpoint(sample_synthetic_video):
    """Validates POST /api/ppe/analyze_video accepts multipart video upload and returns timeline."""
    from fastapi.testclient import TestClient
    from web.server import app

    client = TestClient(app)
    with open(sample_synthetic_video, "rb") as f:
        resp = client.post(
            "/api/ppe/analyze_video",
            files={"file": ("test_clip.mp4", f, "video/mp4")},
            data={"target_fps": "5.0", "max_frames": "10"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "frames" in data
    assert len(data["frames"]) > 0
    assert "clip_summary" in data
    assert "video_metadata" in data
    assert data["video_metadata"]["filename"] == "test_clip.mp4"


def test_api_analyze_video_sample_endpoint(sample_synthetic_video):
    """Validates POST /api/ppe/analyze_video_sample accepts JSON request and returns timeline."""
    from fastapi.testclient import TestClient
    from web.server import app

    client = TestClient(app)
    resp = client.post(
        "/api/ppe/analyze_video_sample",
        json={"sample_path": sample_synthetic_video, "target_fps": 5.0, "max_frames": 10}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "frames" in data
    assert len(data["frames"]) > 0
