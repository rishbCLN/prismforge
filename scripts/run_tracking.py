"""VigiAI Warehouse Tracking Runner.
Executes multi-stage ByteTrack tracking on structured perception detections,
maintains sliding-window trajectory histories, exports tracks.json, and
renders a debug visualization video with bounding boxes, persistent IDs,
and historical motion trails.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import yaml
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from src.perception.yolo import Detection
from src.tracking.warehouse_tracker import WarehouseTracker, WarehouseTrack


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration file from YAML."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Class color palette for debug visualization (BGR format)
CLASS_COLORS = {
    "person": (245, 200, 20),     # Bright cyan-blue
    "carton": (20, 140, 245),     # Orange / amber
    "pallet": (200, 80, 200),     # Purple
    "trolley": (50, 205, 50),     # Lime green
    "default": (200, 200, 200)    # Light grey
}


def render_tracking_visualization(
    video_path: str,
    tracked_frames_data: List[Dict[str, Any]],
    output_viz_path: str,
    fps: float = 25.0
) -> str:
    """Renders debug visualization video with bounding boxes, persistent IDs, and trajectory trails.

    Args:
        video_path: Path to raw input video.
        tracked_frames_data: List of frame dicts with active tracks and historical trajectories.
        output_viz_path: Path to save rendered MP4 video.
        fps: Video playback framerate.

    Returns:
        Absolute path to generated visualization video.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_viz_path)), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video for visualization: {video_path}")
        return ""

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use PyAV libx264 for universal browser playback (Chrome/Edge/Safari/Firefox)
    use_av = False
    av_container = None
    av_stream = None
    cv_out = None

    try:
        import av
        av_container = av.open(output_viz_path, mode="w")
        av_stream = av_container.add_stream("libx264", rate=int(fps))
        av_stream.width = width
        av_stream.height = height
        av_stream.pix_fmt = "yuv420p"
        av_stream.options = {"crf": "22", "preset": "fast"}
        use_av = True
    except Exception as e:
        print(f"[WARN] PyAV H264 unavailable ({e}). Falling back to cv2.VideoWriter.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv_out = cv2.VideoWriter(output_viz_path, fourcc, fps, (width, height))

    # Index tracked data by frame_id
    frames_by_id = {f["frame_id"]: f for f in tracked_frames_data}

    # Running trajectory trails per track_id: list of (x, y) points
    track_trails: Dict[int, List[Tuple[int, int]]] = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        f_data = frames_by_id.get(frame_idx)
        if f_data:
            active_tracks = f_data.get("tracks", [])

            # Draw trajectory trails
            for t in active_tracks:
                tid = t["track_id"]
                cx, cy = int(t["position"][0]), int(t["position"][1])
                if tid not in track_trails:
                    track_trails[tid] = []
                track_trails[tid].append((cx, cy))
                if len(track_trails[tid]) > 20:
                    track_trails[tid].pop(0)

                # Draw trail polylines
                pts = track_trails[tid]
                color = CLASS_COLORS.get(t["class_name"], CLASS_COLORS["default"])
                for k in range(1, len(pts)):
                    thickness = max(1, int(k / 5))
                    cv2.line(frame, pts[k - 1], pts[k], color, thickness)

            # Draw bounding boxes and text badges
            for t in active_tracks:
                tid = t["track_id"]
                cls_name = t["class_name"]
                color = CLASS_COLORS.get(cls_name, CLASS_COLORS["default"])
                x1, y1, x2, y2 = [int(v) for v in t["bbox"]]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label banner
                speed_str = f" v={t['speed']:.2f}" if t.get("speed", 0.0) > 0.05 else ""
                label = f"#{tid} {cls_name} ({t['confidence']:.2f}){speed_str}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - lh - 6)), (x1 + lw + 8, max(lh + 6, y1)), color, -1)
                cv2.putText(frame, label, (x1 + 4, max(lh + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 15), 1)

            # Draw Telemetry HUD
            hud = frame.copy()
            cv2.rectangle(hud, (10, 10), (310, 65), (20, 20, 25), -1)
            cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, f"VigiAI Tracker: Frame {frame_idx:03d}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 220, 245), 1)
            cv2.putText(frame, f"Active Tracks: {len(active_tracks)} | Total Track IDs: {len(track_trails)}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        if use_av:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vframe = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in av_stream.encode(vframe):
                av_container.mux(packet)
        elif cv_out is not None:
            cv_out.write(frame)

        frame_idx += 1

    cap.release()
    if use_av:
        for packet in av_stream.encode():
            av_container.mux(packet)
        av_container.close()
    elif cv_out is not None:
        cv_out.release()

    print(f"[INFO] Rendered tracking debug visualization to: {output_viz_path} ({frame_idx} frames)")
    return os.path.abspath(output_viz_path)


def run_warehouse_tracking(
    detections_path: str,
    output_path: str,
    config_path: Optional[str] = "configs/warehouse.yaml",
    video_path: Optional[str] = None,
    output_viz_path: Optional[str] = None
) -> str:
    """Executes tracking over a detections JSON document and exports structured tracks.

    Args:
        detections_path: Path to input detections JSON file.
        output_path: Path to save exported tracks JSON.
        config_path: Path to warehouse configuration YAML.
        video_path: Optional path to raw video for debug visualization.
        output_viz_path: Optional path for debug visualization MP4.

    Returns:
        Absolute path to exported tracks JSON.
    """
    if not os.path.exists(detections_path):
        raise FileNotFoundError(f"Detections file not found: {detections_path}")

    with open(detections_path, "r", encoding="utf-8") as f:
        det_doc = json.load(f)

    clip_id = det_doc.get("clip_id", "unknown_clip")
    metadata = det_doc.get("metadata", {}) or {}
    frame_w = metadata.get("frame_width", 640)
    frame_h = metadata.get("frame_height", 360)
    fps = metadata.get("fps", 25.0)

    # Load tracker config
    cfg = load_config(config_path) if config_path else {}
    t_cfg = cfg.get("tracking", {})
    tracker = WarehouseTracker(
        high_conf_thresh=t_cfg.get("high_conf_thresh", 0.50),
        low_conf_thresh=t_cfg.get("low_conf_thresh", 0.15),
        iou_threshold=t_cfg.get("iou_threshold", 0.25),
        max_lost_frames=t_cfg.get("max_lost_frames", 15),
        history_window_size=t_cfg.get("history_window_size", 16)
    )

    tracked_frames = []
    unique_track_ids = set()

    for f_data in det_doc.get("frames", []):
        frame_id = f_data["frame_id"]
        timestamp = f_data["timestamp"]

        # Parse detections
        detections = []
        for d in f_data.get("detections", []):
            detections.append(Detection.from_dict(d))

        active_tracks = tracker.update(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            frame_w=frame_w,
            frame_h=frame_h
        )

        frame_tracks_serialized = []
        for t in active_tracks:
            unique_track_ids.add(t.track_id)
            frame_tracks_serialized.append(t.to_dict())

        tracked_frames.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "tracks": frame_tracks_serialized
        })

    # Export complete trajectory histories per track_id
    trajectories = {}
    for tid, t in tracker.active_tracks.items():
        trajectories[str(tid)] = t.history.to_dict()

    out_doc = {
        "clip_id": clip_id,
        "total_frames": len(tracked_frames),
        "total_unique_tracks": len(unique_track_ids),
        "metadata": metadata,
        "frames": tracked_frames,
        "trajectories": trajectories
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = output_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_path, output_path)

    print(f"[INFO] Tracking completed: {len(unique_track_ids)} unique persistent tracks identified across {len(tracked_frames)} frames.")
    print(f"[INFO] Exported structured tracks to: {output_path}")

    # Optional debug visualization
    if video_path and os.path.exists(video_path):
        viz_path = output_viz_path or os.path.join(os.path.dirname(output_path), f"{clip_id}_tracking_viz.mp4")
        render_tracking_visualization(video_path, tracked_frames, viz_path, fps=fps)

    return os.path.abspath(output_path)


def main():
    parser = argparse.ArgumentParser(description="Run ByteTrack tracking on warehouse detections")
    parser.add_argument("--detections", default="outputs/detections.json", help="Path to detections JSON file")
    parser.add_argument("--video", default=None, help="Path to raw video for debug visualization")
    parser.add_argument("--config", default="configs/warehouse.yaml", help="Path to warehouse YAML configuration")
    parser.add_argument("--out", default="outputs/tracks.json", help="Path to output tracks JSON file")
    parser.add_argument("--viz-out", default="outputs/tracking_viz.mp4", help="Path to output debug visualization video")
    args = parser.parse_args()

    run_warehouse_tracking(
        detections_path=args.detections,
        output_path=args.out,
        config_path=args.config,
        video_path=args.video,
        output_viz_path=args.viz_out
    )


if __name__ == "__main__":
    main()
