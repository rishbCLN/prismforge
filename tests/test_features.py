"""Unit tests for HazardMesh Feature Extraction."""
import unittest
import numpy as np
from src.features.feature_extractor import FeatureExtractor, HazardFeatures
from src.tracking.tracker import WorkerTrack


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(danger_threshold=0.22)

    def test_distance_normalization(self):
        pos1 = (100.0, 100.0)
        pos2 = (100.0, 100.0)
        dist = self.extractor.calculate_normalized_distance(pos1, pos2, frame_w=640, frame_h=360)
        self.assertAlmostEqual(dist, 0.0, places=4)

        pos3 = (0.0, 0.0)
        pos4 = (640.0, 360.0)
        dist_diag = self.extractor.calculate_normalized_distance(pos3, pos4, frame_w=640, frame_h=360)
        self.assertAlmostEqual(dist_diag, 1.0, delta=0.01)

    def test_box_overlap(self):
        boxA = (10.0, 10.0, 50.0, 50.0)
        boxB = (10.0, 10.0, 50.0, 50.0)
        ov = self.extractor.calculate_box_overlap(boxA, boxB)
        self.assertAlmostEqual(ov, 1.0, places=3)

        boxC = (100.0, 100.0, 150.0, 150.0)
        ov_zero = self.extractor.calculate_box_overlap(boxA, boxC)
        self.assertAlmostEqual(ov_zero, 0.0, places=3)

    def test_feature_vector_dimensions(self):
        track = WorkerTrack(
            track_id=1,
            first_seen=0.0,
            last_seen=2.0,
            frames_seen=50,
            current_position=(400.0, 200.0),
            bbox=(380.0, 160.0, 420.0, 240.0),
            associated_ppe={"helmet_missing": True, "vest_missing": False, "helmet_conf": 0.95, "vest_conf": 0.90},
            violation_start_time=0.5,
            consecutive_violating_frames=37,
            total_violating_frames=37
        )

        machine = {
            "machine_id": 1,
            "center": (420.0, 200.0),
            "bbox": (370.0, 150.0, 470.0, 250.0),
            "class_name": "excavator",
            "confidence": 0.95
        }

        feat = self.extractor.extract_features(
            worker=track,
            all_workers=[track],
            all_machines=[machine],
            frame_id=50,
            timestamp=2.0,
            frame_shape=(360, 640)
        )

        vec = feat.to_vector()
        self.assertEqual(len(vec), len(HazardFeatures.FEATURE_NAMES))
        self.assertEqual(feat.helmet_missing, 1.0)
        self.assertEqual(feat.vest_missing, 0.0)
        self.assertGreater(feat.machine_proximity_severity, 0.5)
        self.assertGreater(feat.violation_duration, 1.0)


if __name__ == "__main__":
    unittest.main()
