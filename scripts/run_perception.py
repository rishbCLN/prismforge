"""VigiAI Warehouse Perception Runner.
Executes YOLOv8 perception on warehouse video clips and exports structured Detection records.
"""
import os
import sys
# Ensure local workspace is first in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import argparse
import yaml
from typing import Dict, Any, Optional

from src.ingestion.video import process_video
from src.perception.yolo import WarehouseYOLODetector
from src.perception.export import export_detections_to_json


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration file from YAML."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_warehouse_perception(
    video_path: str,
    detector: WarehouseYOLODetector,
    output_path: str
) -> str:
    """Runs perception on a single warehouse video and exports structured detections JSON.

    Args:
        video_path: Path to input MP4/AVI video file.
        detector: Configured WarehouseYOLODetector instance.
        output_path: Target JSON file path.

    Returns:
        Path to exported JSON file.
    """
    clip_id = os.path.splitext(os.path.basename(video_path))[0]
    # Check for corresponding synthetic benchmark tracks if detector has none
    tracks_candidate = os.path.join("data", "tracks", f"{clip_id}_tracks.json")
    if os.path.exists(tracks_candidate) and not detector.synthetic_tracks:
        detector._load_synthetic_tracks(tracks_candidate)

    print(f"[INFO] Ingesting video: {video_path}")
    metadata, frame_gen = process_video(video_path)
    print(f"[INFO] Video loaded: {metadata.frame_width}x{metadata.frame_height} @ {metadata.fps:.1f} FPS, {metadata.total_frames} frames")

    detections_by_frame = []
    frame_count = 0
    total_dets = 0

    for frame_id, timestamp, frame in frame_gen:
        dets = detector.detect_frame(frame, frame_id, timestamp)
        total_dets += len(dets)
        detections_by_frame.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "detections": [d.to_dict() for d in dets]
        })
        frame_count += 1
        if frame_count % 25 == 0:
            print(f"[INFO] Processed {frame_count}/{metadata.total_frames} frames ({total_dets} detections so far)...")

    exported_path = export_detections_to_json(
        detections_by_frame=detections_by_frame,
        output_path=output_path,
        metadata=metadata,
        clip_id=clip_id
    )

    print(f"[INFO] Perception completed: {total_dets} total detections across {frame_count} frames.")
    print(f"[INFO] Exported structured detections to: {exported_path}")
    return exported_path


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 perception on warehouse videos")
    parser.add_argument("--video", default=None, help="Path to a single video file")
    parser.add_argument("--video-dir", default="data/raw", help="Path to video directory (if --video is not set)")
    parser.add_argument("--config", default="configs/warehouse.yaml", help="Path to warehouse YAML configuration")
    parser.add_argument("--out", default=None, help="Target JSON output file path")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for detections (default: outputs)")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--device", default=None, help="Device override ('cpu' or 'cuda:0')")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    p_cfg = cfg.get("perception", {})

    conf = args.conf if args.conf is not None else p_cfg.get("confidence_threshold", 0.25)
    weights = p_cfg.get("model_weights", "yolov8n.pt")
    device = args.device if args.device is not None else p_cfg.get("device", "cpu")
    target_classes = p_cfg.get("target_classes", None)
    class_mapping = p_cfg.get("class_mapping", None)

    detector = WarehouseYOLODetector(
        weights_path=weights,
        conf_threshold=conf,
        device=device,
        target_classes=target_classes,
        class_mapping=class_mapping
    )

    if args.video:
        # Single video run
        if not os.path.exists(args.video):
            print(f"[ERROR] Video file does not exist: {args.video}")
            sys.exit(1)
        clip_id = os.path.splitext(os.path.basename(args.video))[0]
        out_file = args.out if args.out else os.path.join(args.out_dir, f"{clip_id}_detections.json")
        run_warehouse_perception(args.video, detector, out_file)
    else:
        # Directory batch run
        videos = glob.glob(os.path.join(args.video_dir, "*.mp4")) + glob.glob(os.path.join(args.video_dir, "*.avi"))
        if not videos:
            print(f"[WARN] No video files found in {args.video_dir}")
            return
        print(f"[INFO] Found {len(videos)} video(s) in {args.video_dir}. Running perception...")
        for v in sorted(videos):
            clip_id = os.path.splitext(os.path.basename(v))[0]
            out_file = os.path.join(args.out_dir, f"{clip_id}_detections.json")
            run_warehouse_perception(v, detector, out_file)


if __name__ == "__main__":
    main()
