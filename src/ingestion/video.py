"""VigiAI Warehouse Video Ingestion Layer.
Provides robust video loading, frame iteration, and metadata extraction for MP4/AVI warehouse videos.
"""
from dataclasses import dataclass, asdict
from typing import Generator, Tuple, Optional, Dict, Any
import os
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Metadata container for ingested video files."""
    video_path: str
    fps: float
    frame_width: int
    frame_height: int
    total_frames: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VideoIngestor:
    """Robust video reader for warehouse prerecorded videos."""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, video_path: str, max_frames: Optional[int] = None, stride: int = 1):
        """Initializes and validates the video file.

        Args:
            video_path: Absolute or relative path to the video file.
            max_frames: Optional limit on the number of frames to process.
            stride: Process every n-th frame (default: 1, every frame).

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the file path is a directory or has an unsupported format.
            RuntimeError: If OpenCV cannot open or read frames from the video.
        """
        self.video_path = os.path.abspath(video_path)
        self.max_frames = max_frames
        self.stride = max(1, stride)

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        if os.path.isdir(self.video_path):
            raise ValueError(f"Path is a directory, not a video file: {self.video_path}")

        ext = os.path.splitext(self.video_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported video extension '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file (corrupt or unreadable codec): {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Fallback to standard 25.0 FPS if cap.get returns invalid/0
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = max(0, total_frames)

        # Quick validation: check if at least first frame is readable
        ret, first_frame = cap.read()
        cap.release()

        if not ret or first_frame is None:
            raise RuntimeError(f"Video contains zero readable frames: {self.video_path}")

        # If OpenCV reports 0 total frames, fallback to at least 1
        if self.total_frames == 0:
            self.total_frames = 1

        self.duration_seconds = round(self.total_frames / self.fps, 3)

        self.metadata = VideoMetadata(
            video_path=self.video_path,
            fps=self.fps,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            total_frames=self.total_frames,
            duration_seconds=self.duration_seconds
        )

    def iter_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Generator yielding (frame_id, timestamp_seconds, frame_bgr) for each frame.

        Yields:
            Tuple containing:
                - frame_id (int): Zero-indexed consecutive frame index.
                - timestamp (float): Elapsed time in seconds.
                - frame (np.ndarray): Decoded BGR image matrix.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video during iteration: {self.video_path}")

        raw_index = 0
        yielded_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if raw_index % self.stride == 0:
                    timestamp = round(raw_index / self.fps, 3)
                    yield raw_index, timestamp, frame
                    yielded_count += 1

                    if self.max_frames is not None and yielded_count >= self.max_frames:
                        break

                raw_index += 1
        finally:
            cap.release()


def process_video(
    video_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[VideoMetadata, Generator[Tuple[int, float, np.ndarray], None, None]]:
    """Standardized entrypoint to ingest and process a warehouse video file.

    Args:
        video_path: Path to the video file.
        config: Optional configuration dictionary containing 'max_frames' or 'stride'.

    Returns:
        Tuple of (VideoMetadata, frame_generator).
    """
    max_frames = config.get("max_frames") if config else None
    stride = config.get("stride", 1) if config else 1
    ingestor = VideoIngestor(video_path=video_path, max_frames=max_frames, stride=stride)
    return ingestor.metadata, ingestor.iter_frames()
