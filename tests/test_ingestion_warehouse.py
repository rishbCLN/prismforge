"""Unit tests for VigiAI Warehouse Video Ingestion Layer."""
import os
import unittest
import tempfile
import cv2
import numpy as np
from src.ingestion.video import VideoIngestor, VideoMetadata, process_video


class TestWarehouseVideoIngestion(unittest.TestCase):
    """Validates robust video ingestion, metadata extraction, and error handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.temp_dir, "test_clip.mp4")

        # Create a synthetic 10-frame 320x240 MP4 video @ 20 FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.test_video_path, fourcc, 20.0, (320, 240))
        for i in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Add simple gradient to ensure frame is valid
            frame[:, :] = (i * 20, 100, 150)
            out.write(frame)
        out.release()

    def tearDown(self):
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_valid_video_metadata(self):
        """Tests that video dimensions, FPS, and frame count are accurately read."""
        metadata, frame_gen = process_video(self.test_video_path)
        self.assertIsInstance(metadata, VideoMetadata)
        self.assertEqual(metadata.frame_width, 320)
        self.assertEqual(metadata.frame_height, 240)
        self.assertEqual(metadata.fps, 20.0)
        self.assertEqual(metadata.total_frames, 10)
        self.assertAlmostEqual(metadata.duration_seconds, 0.5, places=2)

    def test_frame_iteration(self):
        """Tests that frame generator yields correct indices and timestamps."""
        metadata, frame_gen = process_video(self.test_video_path)
        frames = list(frame_gen)
        self.assertEqual(len(frames), 10)

        # First frame checks
        first_id, first_ts, first_frame = frames[0]
        self.assertEqual(first_id, 0)
        self.assertEqual(first_ts, 0.0)
        self.assertEqual(first_frame.shape, (240, 320, 3))

        # Last frame checks
        last_id, last_ts, last_frame = frames[-1]
        self.assertEqual(last_id, 9)
        self.assertAlmostEqual(last_ts, 9 / 20.0, places=2)

    def test_stride_and_max_frames(self):
        """Tests subsampling with stride and max_frames limit."""
        metadata, frame_gen = process_video(self.test_video_path, config={"stride": 2, "max_frames": 3})
        frames = list(frame_gen)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0][0], 0)
        self.assertEqual(frames[1][0], 2)
        self.assertEqual(frames[2][0], 4)

    def test_missing_file_raises(self):
        """Tests that a non-existent path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            process_video(os.path.join(self.temp_dir, "non_existent.mp4"))

    def test_unsupported_extension_raises(self):
        """Tests that an unsupported file format raises ValueError."""
        dummy_txt = os.path.join(self.temp_dir, "test.txt")
        with open(dummy_txt, "w") as f:
            f.write("not a video")
        with self.assertRaises(ValueError):
            process_video(dummy_txt)
        os.remove(dummy_txt)

    def test_directory_path_raises(self):
        """Tests that passing a directory path raises ValueError."""
        with self.assertRaises(ValueError):
            process_video(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
