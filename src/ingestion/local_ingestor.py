"""Local File and Video Ingestor Module.
Ingests user-provided videos (MP4, AVI, MOV) and ZIP archives,
standardizes video dimensions (640x360), frame rates (25 FPS), and chunks long footage.
"""
import os
import zipfile
import shutil
import cv2
from typing import List, Dict, Any, Optional


class LocalIngestor:
    """Standardizes and chunks local video footage or extracted archives."""

    TARGET_WIDTH = 640
    TARGET_HEIGHT = 360
    TARGET_FPS = 25.0
    CHUNK_DURATION_S = 4.0   # 4.0s = 100 frames per chunk

    def __init__(self, raw_output_dir: str = "data/raw"):
        self.raw_output_dir = raw_output_dir
        os.makedirs(raw_output_dir, exist_ok=True)

    def ingest_archive(self, zip_path: str, extract_dir: str = "data/ingested/uploads") -> List[str]:
        """Extracts a zip file and standardizes all contained videos."""
        os.makedirs(extract_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        unzip_path = os.path.join(extract_dir, base_name)
        os.makedirs(unzip_path, exist_ok=True)

        print(f"Extracting archive {zip_path} to {unzip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(unzip_path)

        # Locate all video files in extracted directory
        found_videos = []
        for root, _, files in os.walk(unzip_path):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    found_videos.append(os.path.join(root, f))

        generated_clips = []
        for v in found_videos:
            clips = self.standardize_and_chunk_video(v)
            generated_clips.extend(clips)

        return generated_clips

    def standardize_and_chunk_video(self, input_video_path: str, clip_prefix: Optional[str] = None) -> List[str]:
        """Reads a video, standardizes to 640x360 @ 25fps, and chunks into 4-second clips."""
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video file not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        base_name = clip_prefix or os.path.splitext(os.path.basename(input_video_path))[0]
        # Clean name
        safe_base = "".join(c if c.isalnum() or c == "_" else "_" for c in base_name)

        frames_per_chunk = int(self.CHUNK_DURATION_S * self.TARGET_FPS)
        chunk_idx = 1
        current_chunk_frames = []
        generated_clip_paths = []

        frame_read_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to target dimensions
            if frame.shape[1] != self.TARGET_WIDTH or frame.shape[0] != self.TARGET_HEIGHT:
                frame = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))

            current_chunk_frames.append(frame)
            frame_read_count += 1

            if len(current_chunk_frames) >= frames_per_chunk:
                out_filename = f"{safe_base}_chunk_{chunk_idx:02d}.mp4"
                out_path = os.path.join(self.raw_output_dir, out_filename)
                self._write_video_file(current_chunk_frames, out_path, self.TARGET_FPS)
                generated_clip_paths.append(out_path)
                current_chunk_frames = []
                chunk_idx += 1

        cap.release()

        # Write remaining frames if at least 1.5s
        if len(current_chunk_frames) >= int(1.5 * self.TARGET_FPS):
            out_filename = f"{safe_base}_chunk_{chunk_idx:02d}.mp4"
            out_path = os.path.join(self.raw_output_dir, out_filename)
            self._write_video_file(current_chunk_frames, out_path, self.TARGET_FPS)
            generated_clip_paths.append(out_path)

        # If video was very short and nothing wrote, write whatever we had
        if not generated_clip_paths and current_chunk_frames:
            out_filename = f"{safe_base}_chunk_01.mp4"
            out_path = os.path.join(self.raw_output_dir, out_filename)
            self._write_video_file(current_chunk_frames, out_path, self.TARGET_FPS)
            generated_clip_paths.append(out_path)

        print(f"Standardized {input_video_path} into {len(generated_clip_paths)} clip(s).")
        return generated_clip_paths

    @staticmethod
    def _write_video_file(frames: List[cv2.typing.MatLike], output_path: str, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
