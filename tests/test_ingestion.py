"""Unit tests for HazardMesh Data Ingestion Pipeline."""
import os
import unittest
import numpy as np
import cv2
from src.ingestion.kaggle_ingestor import KaggleIngestor, DEFAULT_KAGGLE_TOKEN
from src.ingestion.local_ingestor import LocalIngestor


class TestIngestion(unittest.TestCase):
    def setUp(self):
        self.test_dir = "data/ingested/test_scratch"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_kaggle_token_config(self):
        ing = KaggleIngestor()
        conn = ing.check_connection()
        self.assertIn("status", conn)
        self.assertEqual(conn["status"], "connected")
        self.assertTrue(conn["token_prefix"].startswith("KGAT_"))

    def test_local_video_standardization(self):
        # Generate a small temporary test video (e.g. 50 frames @ 30fps 320x240)
        temp_video_path = os.path.join(self.test_dir, "test_raw_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (320, 240))
        for _ in range(50):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        ingestor = LocalIngestor(raw_output_dir=self.test_dir)
        clips = ingestor.standardize_and_chunk_video(temp_video_path, clip_prefix="test_chunk")

        self.assertGreater(len(clips), 0)
        # Verify output video properties
        cap = cv2.VideoCapture(clips[0])
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 640)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 360)
        self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 25.0, delta=0.5)
        cap.release()


if __name__ == "__main__":
    unittest.main()
