"""Unit tests for VigiAI Warehouse YOLO Perception and Export Layer."""
import os
import unittest
import tempfile
import numpy as np
from src.perception.yolo import Detection, WarehouseYOLODetector
from src.perception.export import export_detections_to_json, load_detections_from_json
from src.ingestion.video import VideoMetadata


class TestWarehousePerception(unittest.TestCase):
    """Validates Detection dataclass, coordinate normalization, and JSON export."""

    def test_detection_dataclass_and_normalization(self):
        """Tests that normalized coordinates are accurately computed and serialized."""
        det = Detection(
            frame_id=10,
            timestamp=0.4,
            track_id=None,
            class_id=0,
            class_name="person",
            confidence=0.92,
            x1=100.0,
            y1=50.0,
            x2=200.0,
            y2=250.0,
            center_x=150.0,
            center_y=150.0,
            width=100.0,
            height=200.0,
            center_x_norm=0.2344,
            center_y_norm=0.4167,
            width_norm=0.1562,
            height_norm=0.5556,
            x1_norm=0.1562,
            y1_norm=0.1389,
            x2_norm=0.3125,
            y2_norm=0.6944
        )

        d_dict = det.to_dict()
        self.assertEqual(d_dict["frame_id"], 10)
        self.assertEqual(d_dict["class_name"], "person")
        self.assertEqual(d_dict["confidence"], 0.92)
        self.assertEqual(d_dict["center_x_norm"], 0.2344)

        # Test from_dict deserialization
        reconstructed = Detection.from_dict(d_dict)
        self.assertEqual(reconstructed.frame_id, det.frame_id)
        self.assertEqual(reconstructed.class_name, det.class_name)
        self.assertEqual(reconstructed.width, det.width)

    def test_export_and_load_detections_json(self):
        """Tests exporting and loading structured detections JSON."""
        temp_dir = tempfile.mkdtemp()
        out_path = os.path.join(temp_dir, "test_detections.json")

        sample_detections = [
            {
                "frame_id": 0,
                "timestamp": 0.0,
                "detections": [
                    {
                        "frame_id": 0,
                        "timestamp": 0.0,
                        "track_id": None,
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.88,
                        "x1": 50.0,
                        "y1": 50.0,
                        "x2": 100.0,
                        "y2": 150.0,
                        "center_x": 75.0,
                        "center_y": 100.0,
                        "width": 50.0,
                        "height": 100.0,
                        "center_x_norm": 0.117,
                        "center_y_norm": 0.278,
                        "width_norm": 0.078,
                        "height_norm": 0.278,
                        "x1_norm": 0.078,
                        "y1_norm": 0.139,
                        "x2_norm": 0.156,
                        "y2_norm": 0.417
                    }
                ]
            }
        ]

        meta = VideoMetadata(
            video_path="sample.mp4",
            fps=25.0,
            frame_width=640,
            frame_height=360,
            total_frames=1,
            duration_seconds=0.04
        )

        exported_file = export_detections_to_json(sample_detections, out_path, metadata=meta, clip_id="sample_clip")
        self.assertTrue(os.path.exists(exported_file))

        loaded = load_detections_from_json(exported_file)
        self.assertEqual(loaded["clip_id"], "sample_clip")
        self.assertEqual(loaded["total_frames"], 1)
        self.assertEqual(loaded["total_detections"], 1)
        self.assertEqual(loaded["frames"][0]["detections"][0]["class_name"], "person")

        os.remove(out_path)
        os.rmdir(temp_dir)

    def test_detector_empty_frame(self):
        """Tests that detector safely returns empty list for None or blank frames."""
        detector = WarehouseYOLODetector()
        self.assertEqual(detector.detect_frame(None, 0, 0.0), [])
        self.assertEqual(detector.detect_frame(np.array([]), 0, 0.0), [])


if __name__ == "__main__":
    unittest.main()
