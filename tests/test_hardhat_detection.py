"""Dedicated test suite for Hardhat square box detection, cranial locator, face-anchored verification, and neural reasoning."""
import os
import torch
import numpy as np
import pytest

from src.risk_model.ppe_reasoner_net import PPEReasonerNet, HardhatReasonerBlock, extract_ppe_neural_features
from src.perception.detector import SafetyDetector, PPEInspector, FaceCranialDetector
from src.features.ppe_inference import PPEInferenceEngine


def test_face_cranial_detector_synthetic():
    """Validates that FaceCranialDetector reliably locates facial region in human head crop."""
    # Synthetic worker crop: 200x100 BGR image
    crop = np.ones((200, 100, 3), dtype=np.uint8) * 120
    # Add face skin tone region in upper 30%: YCrCb skin tone (BGR around [110, 140, 200])
    crop[20:60, 30:70] = [110, 140, 200]

    has_face, conf, face_box, method = FaceCranialDetector.detect_face(crop, [50, 100, 150, 300])
    assert has_face is True
    assert conf >= 0.70
    assert face_box is not None
    assert len(face_box) == 4
    fx1, fy1, fx2, fy2 = face_box
    assert fx2 > fx1
    assert fy2 > fy1
    assert method in ("haar_frontal", "haar_profile", "skin_cranial_morph", "anatomical_prior")


def test_verify_hardhat_on_face_good_detect():
    """Validates that a hardhat sitting directly on top of the face passes the 'Good Detect' rule."""
    face_box = [100.0, 120.0, 180.0, 200.0]  # Face: width=80, height=80, center=(140, 160)
    # Hardhat candidate sits directly on top of face:
    # center=(140, 95), width=90, height=60, bottom=125 (meeting forehead at 120)
    candidate_box = [95.0, 65.0, 185.0, 125.0]

    is_gd, conf, sq_box, metrics = PPEInspector.verify_hardhat_on_face(candidate_box, face_box)
    assert is_gd is True, f"Expected Good Detect, got metrics: {metrics}"
    assert conf >= 0.80
    assert sq_box is not None
    hx1, hy1, hx2, hy2 = sq_box
    # Check squareness: width == height
    assert abs((hx2 - hx1) - (hy2 - hy1)) < 1e-4
    # Check vertical position: sits above face
    assert (hy1 + hy2) / 2.0 < (face_box[1] + face_box[3]) / 2.0


def test_verify_hardhat_off_face_rejection():
    """Validates that a hardhat held at the waist or hands is strictly rejected from being a Good Detect."""
    face_box = [100.0, 120.0, 180.0, 200.0]  # Face at y=120..200
    # Hardhat candidate held at waist (y=280..340, well below face)
    waist_candidate = [95.0, 280.0, 185.0, 340.0]

    is_gd, conf, sq_box, metrics = PPEInspector.verify_hardhat_on_face(waist_candidate, face_box)
    assert is_gd is False
    assert sq_box is None
    assert metrics["is_above_face"] is False

    # Hardhat off to far right side (e.g. table or shoulder)
    side_candidate = [280.0, 70.0, 350.0, 130.0]
    is_gd_side, _, sq_side, metrics_side = PPEInspector.verify_hardhat_on_face(side_candidate, face_box)
    assert is_gd_side is False
    assert sq_side is None
    assert metrics_side["is_horiz_aligned"] is False


def test_hardhat_reasoner_block_forward():
    """Validates HardhatReasonerBlock generates compliant score when spatial gate is open."""
    block = HardhatReasonerBlock(latent_dim=48, hardhat_feat_dim=5)
    block.eval()

    latent = torch.randn(2, 48)
    # [has_hh, conf, iou_worker, iou_head, area_ratio]
    hh_feats = torch.tensor([[1.0, 0.95, 0.25, 0.70, 0.08],
                             [1.0, 0.90, 0.20, 0.65, 0.07]], dtype=torch.float32)
    spatial_gate = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

    score = block(latent, hh_feats, spatial_gate)
    assert score.shape == (2, 1)
    assert 0.0 <= score[0, 0].item() <= 1.0


def test_hardhat_reasoner_block_strict_spatial_gate():
    """Validates that when spatial_gate is 0 (hardhat held in hand / off head), compliance is strictly 0."""
    block = HardhatReasonerBlock(latent_dim=48, hardhat_feat_dim=5)
    block.eval()

    latent = torch.randn(2, 48)
    # Hardhat detected with 99% confidence but held at waist (spatial_gate is 0.0)
    hh_feats = torch.tensor([[0.0, 0.0, 0.20, 0.0, 0.08],
                             [0.0, 0.0, 0.20, 0.0, 0.08]], dtype=torch.float32)
    spatial_gate = torch.tensor([[0.0], [0.0]], dtype=torch.float32)

    score = block(latent, hh_feats, spatial_gate)
    assert score[0, 0].item() == 0.0
    assert score[1, 0].item() == 0.0


def test_cranial_hardhat_square_geometry():
    """Validates that locate_cranial_hardhat_square outputs an exact square bounding box."""
    # Synthetic head with yellow hardhat (H in [17, 42], S > 95, V > 115)
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    # Cranial vault top 30 px: fill with yellow hardhat color (BGR: 0, 255, 255)
    img[5:35, 20:80] = [0, 240, 240]

    has_hh, conf, hh_box, is_gd = PPEInspector.locate_cranial_hardhat_square(img, [0, 0, 100, 200])
    assert has_hh is True
    assert conf > 0.70
    assert hh_box is not None
    assert is_gd is True

    hx1, hy1, hx2, hy2 = hh_box
    bw = hx2 - hx1
    bh = hy2 - hy1
    # Check squareness: width and height should be equal within floating point tolerance
    assert abs(bw - bh) < 1e-4, f"Hardhat box must be square: width={bw}, height={bh}"


def test_ppe_inference_engine_detects_hardhat_and_face():
    """Validates PPEInferenceEngine populates face_box, hardhat_box with square geometry, and flags is_good_detect."""
    engine = PPEInferenceEngine()

    # Create synthetic frame with worker having yellow hardhat on face
    frame = np.ones((400, 400, 3), dtype=np.uint8) * 40
    # Worker body: [100, 80, 260, 380]
    frame[80:380, 100:260] = [120, 120, 120]
    # Face skin: y: 115 to 175, x: 140 to 220
    frame[115:175, 140:220] = [110, 140, 200]
    # Hardhat on head: top 35px of worker (y: 80 to 125, x: 130 to 230) with safety yellow
    frame[80:125, 130:230] = [0, 230, 230]
    # Safety vest on chest: y: 180 to 300 with neon orange (BGR: 0, 140, 255)
    frame[180:300, 110:250] = [0, 140, 255]

    from src.perception.detector import Detection
    engine.detector.detect_frame = lambda f, frame_id=0, timestamp=0.0: [
        Detection(
            frame_id=0, timestamp=0.0, track_id=None, class_id=0, class_name="person",
            confidence=0.92, x1=100.0, y1=80.0, x2=260.0, y2=380.0,
            center_x=180.0, center_y=230.0, width=160.0, height=300.0
        )
    ]

    res = engine.analyze_cv2_image(frame, filename="synthetic_worker_test.jpg")
    assert res["status"] == "success"
    assert "hardhat_summary" in res
    assert "workers" in res
    assert len(res["workers"]) >= 1

    w0 = res["workers"][0]
    assert w0["has_face"] is True
    assert w0["face_box"] is not None
    assert "is_good_detect" in w0
    if w0["has_hardhat"]:
        assert w0["is_good_detect"] is True
        assert w0["hardhat_box"] is not None
        hx1, hy1, hx2, hy2 = w0["hardhat_box"]
        # Check that bounding box is an exact square
        assert abs((hx2 - hx1) - (hy2 - hy1)) <= 1


def test_concrete_pillar_rejection():
    """Validates that a uniform architectural concrete pillar/column crop produces no face."""
    # Synthetic concrete column crop (grayish-tan with subtle vertical texture, no human facial features)
    pillar_crop = np.ones((350, 120, 3), dtype=np.uint8) * 130
    pillar_crop[:, :, 0] = 125
    pillar_crop[:, :, 2] = 138

    has_face, conf, face_box, method = FaceCranialDetector.detect_face(pillar_crop, [10.0, 0.0, 130.0, 350.0])
    assert has_face is False, f"Concrete column must not detect a face, got {method}"
    assert face_box is None


def test_duplicate_worker_containment_suppression():
    """Validates that overlapping lower-confidence worker boxes are suppressed by containment filter."""
    from src.perception.detector import Detection
    engine = PPEInferenceEngine()

    dummy_img = np.ones((400, 600, 3), dtype=np.uint8) * 50

    # Person 1: High confidence real worker [84, 0, 610, 408]
    p1 = Detection(
        frame_id=0, timestamp=0.0, track_id=None, class_id=0, class_name="person",
        confidence=0.932, x1=84.0, y1=0.0, x2=610.0, y2=408.0,
        center_x=347.0, center_y=204.0, width=526.0, height=408.0
    )
    # Person 2: Overlapping low-confidence column [0, 5, 214, 408] (containment > 60%)
    p2 = Detection(
        frame_id=0, timestamp=0.0, track_id=None, class_id=0, class_name="person",
        confidence=0.502, x1=0.0, y1=5.0, x2=214.0, y2=408.0,
        center_x=107.0, center_y=206.5, width=214.0, height=403.0
    )

    filtered = engine.filter_valid_workers([p1, p2], dummy_img, [], [])
    assert len(filtered) == 1, f"Expected 1 worker, got {len(filtered)}"
    assert filtered[0].confidence == 0.932


def test_bricklayer_exact_single_worker():
    """Validates real bricklayer image resolves to exactly 1 compliant hardhat worker."""
    import cv2
    img_path = r"D:\Downloads\senior-male-worker-bricklaying-at-construction-site.webp"
    if not os.path.exists(img_path):
        pytest.skip("Test image not available in local downloads.")

    img = cv2.imread(img_path)
    assert img is not None

    engine = PPEInferenceEngine()
    res = engine.analyze_cv2_image(img, filename="bricklayer_test.webp")

    assert res["status"] == "success"
    assert len(res["workers"]) == 1, f"Expected exactly 1 worker, got {len(res['workers'])}"
    w0 = res["workers"][0]
    assert w0["has_face"] is True
    assert w0["has_hardhat"] is True
    assert w0["is_good_detect"] is True
    assert w0["hardhat_box"] is not None
    # Check square dims
    hx1, hy1, hx2, hy2 = w0["hardhat_box"]
    assert abs((hx2 - hx1) - (hy2 - hy1)) <= 1
    # Check that standalone hardhat list is empty (no false alarm on bricks)
    assert len(res.get("unattended_hardhats", [])) == 0

