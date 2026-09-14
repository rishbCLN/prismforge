"""VigiAI Warehouse Feed Batch Processor.
Processes all raw warehouse CCTV videos from data/raw/:
- Executes YOLOv8 object perception (workers, cartons, pallets, equipment)
- Applies ByteTrack multi-stage tracking and velocity estimation
- Builds full lifetime continuous trajectory histories
- Renders AI visualization frames into outputs/<key>_frames/
- Exports structured track JSON into outputs/<key>_tracks.json
"""
import os
import sys
import glob
import json
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

# Ensure workspace root in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.perception.yolo import WarehouseYOLODetector, Detection
from src.tracking.warehouse_tracker import WarehouseTracker

RAW_FEEDS = [
    {
        "key": "throw_seating",
        "video": "Throwing seating cartons, using strap to hold.mp4",
        "title": "Throwing seating cartons, using strap to hold",
        "stride": 2, # 451 -> 226 frames @ 15fps
    },
    {
        "key": "drag_cupboard",
        "video": "Dock level, dragging cupboard.mp4",
        "title": "Dock level, dragging cupboard",
        "stride": 3, # 930 -> 310 frames @ 10fps
    },
    {
        "key": "kd_packets",
        "video": "KD packets dragged, heavy box kept on other packets.mp4",
        "title": "KD packets dragged, heavy box kept on other packets",
        "stride": 3, # 1012 -> 338 frames @ 10fps
    },
    {
        "key": "throw_mattress",
        "video": "Throwing Mattresses.mp4",
        "title": "Throwing Mattresses",
        "stride": 4, # 1249 -> 313 frames @ 7.5fps
    },
    {
        "key": "stepping_cartons",
        "video": "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4",
        "title": "Stepping on cartons, vertical product kept horizontally",
        "stride": 4, # 1471 -> 368 frames @ 7.5fps
    },
]

CLASS_COLORS = {
    "person": (245, 200, 20),      # Bright cyan-blue
    "carton": (20, 145, 255),      # Bright amber-orange
    "pallet": (200, 80, 200),      # Purple
    "trolley": (50, 205, 50),      # Lime green
    "default": (180, 180, 180)     # Light grey
}

def process_single_video(feed_info: Dict[str, Any], detector: WarehouseYOLODetector, outputs_dir: str):
    key = feed_info["key"]
    video_filename = feed_info["video"]
    stride = feed_info.get("stride", 1)
    video_path = os.path.join("data", "raw", video_filename)

    if not os.path.exists(video_path):
        print(f"[WARN] Video not found: {video_path}")
        return

    frames_out_dir = os.path.join(outputs_dir, f"{key}_frames")
    tracks_out_path = os.path.join(outputs_dir, f"{key}_tracks.json")
    os.makedirs(frames_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_fps = orig_fps / stride

    print(f"\n[INFO] === Processing: {video_filename} ===")
    print(f"[INFO] Raw frames: {total_raw_frames} @ {orig_fps:.1f} FPS | Sampling stride: {stride} -> Effective FPS: {effective_fps:.1f}")

    tracker = WarehouseTracker(
        high_conf_thresh=0.45,
        low_conf_thresh=0.15,
        iou_threshold=0.25,
        max_lost_frames=20,
        history_window_size=30
    )

    raw_frame_idx = 0
    saved_frame_idx = 0
    unique_track_ids = set()
    frames_data = []

    # Full lifetime trajectory store for ALL tracks
    all_trajectories: Dict[str, List[Dict[str, Any]]] = {}
    track_trails: Dict[int, List[tuple]] = {}

    start_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if raw_frame_idx % stride == 0:
            timestamp = saved_frame_idx / effective_fps

            # Run YOLO perception
            dets = detector.detect_frame(frame, saved_frame_idx, timestamp)

            # Update tracker
            active_tracks = tracker.update(
                frame_id=saved_frame_idx,
                timestamp=timestamp,
                detections=dets,
                frame_w=orig_w,
                frame_h=orig_h
            )

            serialized_tracks = []
            for t in active_tracks:
                unique_track_ids.add(t.track_id)
                t_dict = t.to_dict()
                serialized_tracks.append(t_dict)

                # Record full continuous trajectory
                str_id = str(t.track_id)
                if str_id not in all_trajectories:
                    all_trajectories[str_id] = []
                all_trajectories[str_id].append({
                    "frame_id": saved_frame_idx,
                    "timestamp": timestamp,
                    "center_x_norm": t.current_position_norm[0],
                    "center_y_norm": t.current_position_norm[1],
                    "speed": t_dict.get("speed", 0.0),
                    "confidence": t.confidence,
                    "class_name": t.class_name
                })

            frames_data.append({
                "frame_id": saved_frame_idx,
                "timestamp": timestamp,
                "tracks": serialized_tracks
            })

            # Render visualization frame with overlays
            viz_frame = frame.copy()

            # Trails
            for t in active_tracks:
                tid = t.track_id
                cx, cy = int(t.current_position[0]), int(t.current_position[1])
                if tid not in track_trails:
                    track_trails[tid] = []
                track_trails[tid].append((cx, cy))
                if len(track_trails[tid]) > 18:
                    track_trails[tid].pop(0)

                pts = track_trails[tid]
                color = CLASS_COLORS.get(t.class_name, CLASS_COLORS["default"])
                for k in range(1, len(pts)):
                    cv2.line(viz_frame, pts[k-1], pts[k], color, max(1, int(k / 5)))

            # Bounding boxes
            for t in active_tracks:
                tid = t.track_id
                cls_name = t.class_name
                color = CLASS_COLORS.get(cls_name, CLASS_COLORS["default"])
                x1, y1, x2, y2 = [int(v) for v in t.current_bbox]

                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
                speed_val = getattr(t, "speed", 0.0)
                speed_str = f" v={speed_val:.2f}" if speed_val > 0.05 else ""
                label = f"#{tid} {cls_name} ({t.confidence:.2f}){speed_str}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(viz_frame, (x1, max(0, y1 - lh - 6)), (x1 + lw + 8, max(lh + 6, y1)), color, -1)
                cv2.putText(viz_frame, label, (x1 + 4, max(lh + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 15), 1)

            # Top HUD
            hud = viz_frame.copy()
            cv2.rectangle(hud, (10, 10), (320, 65), (15, 18, 24), -1)
            cv2.addWeighted(hud, 0.75, viz_frame, 0.25, 0, viz_frame)
            cv2.putText(viz_frame, f"VigiAI Tracker: Frame {saved_frame_idx:03d}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 220, 245), 1)
            cv2.putText(viz_frame, f"Active: {len(active_tracks)} | Total Unique IDs: {len(unique_track_ids)}", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)

            # Save JPEG frame
            out_img_path = os.path.join(frames_out_dir, f"frame_{saved_frame_idx:03d}.jpg")
            cv2.imwrite(out_img_path, viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            saved_frame_idx += 1
            if saved_frame_idx % 50 == 0:
                print(f"[INFO] Processed {saved_frame_idx} output frames (raw frame {raw_frame_idx}/{total_raw_frames})...")

        raw_frame_idx += 1

    cap.release()
    elapsed = time.time() - start_t
    print(f"[INFO] Finished {key}: {saved_frame_idx} frames rendered in {elapsed:.1f}s ({saved_frame_idx/elapsed:.1f} fps)")

    # Build trajectories dictionary formatted for view_live.html
    trajectories_export = {}
    for str_id, obs_list in all_trajectories.items():
        trajectories_export[str_id] = {
            "track_id": int(str_id),
            "observations": obs_list
        }

    out_doc = {
        "clip_id": key,
        "total_frames": saved_frame_idx,
        "total_unique_tracks": len(unique_track_ids),
        "metadata": {
            "original_video": video_filename,
            "frame_width": orig_w,
            "frame_height": orig_h,
            "fps": effective_fps,
            "duration_seconds": saved_frame_idx / effective_fps
        },
        "frames": frames_data,
        "trajectories": trajectories_export
    }

    with open(tracks_out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)

    print(f"[INFO] Exported tracks & trajectories to: {tracks_out_path}")

def main():
    detector = WarehouseYOLODetector(
        weights_path="yolov8n.pt",
        conf_threshold=0.20,
        device="cpu",
        target_classes=["person", "box", "package", "pallet", "trolley", "suitcase", "backpack", "handbag"],
        class_mapping={
            "person": "person",
            "box": "carton",
            "package": "carton",
            "suitcase": "carton",
            "backpack": "carton",
            "handbag": "carton",
            "pallet": "pallet",
            "trolley": "trolley",
        }
    )

    outputs_dir = os.path.join(WORKSPACE_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    for feed in RAW_FEEDS:
        process_single_video(feed, detector, outputs_dir)

    print("\n[SUCCESS] All warehouse feeds successfully processed and exported!")

if __name__ == "__main__":
    main()
