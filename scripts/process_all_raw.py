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
    "person": (166, 184, 20),      # #14B8A6 Teal
    "carton": (246, 130, 59),      # #3B82F6 Primary Blue
    "pallet": (166, 184, 20),      # Teal
    "trolley": (34, 197, 94),      # #22C55E Green
    "default": (166, 184, 20)      # Teal
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

    min_dim = min(orig_w, orig_h) if (orig_w > 0 and orig_h > 0) else 720
    scale_up = max(1.0, 720.0 / float(min_dim)) if min_dim > 0 else 1.0
    viz_w = int(round(orig_w * scale_up))
    viz_h = int(round(orig_h * scale_up))

    print(f"\n[INFO] === Processing: {video_filename} ===")
    print(f"[INFO] Raw frames: {total_raw_frames} @ {orig_fps:.1f} FPS | Sampling stride: {stride} -> Effective FPS: {effective_fps:.1f} (Viz: {viz_w}x{viz_h})")

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

            for t in active_tracks:
                unique_track_ids.add(t.track_id)

            # Convert to serializable format for JSON export
            serialized_tracks = []
            for t in active_tracks:
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

            # Render visualization frame with overlays in HD
            if scale_up > 1.001:
                viz_frame = cv2.resize(frame, (viz_w, viz_h), interpolation=cv2.INTER_CUBIC)
            else:
                viz_frame = frame.copy()

            ref_dim = min(viz_w, viz_h)
            font_scale = max(0.68, min(ref_dim / 850.0, 1.25) * 0.85)
            text_thick = max(2, int(round(font_scale * 2.2)))
            font = cv2.FONT_HERSHEY_SIMPLEX
            pad_h = int(10 * scale_up)
            pad_v = int(6 * scale_up)
            (_, base_lh), _ = cv2.getTextSize("TEST", font, font_scale, text_thick)

            # Motion trails
            for t in active_tracks:
                tid = t.track_id
                cx = int(t.current_position[0] * scale_up)
                cy = int(t.current_position[1] * scale_up)
                if tid not in track_trails:
                    track_trails[tid] = []
                track_trails[tid].append((cx, cy))
                if len(track_trails[tid]) > 18:
                    track_trails[tid].pop(0)

                pts = track_trails[tid]
                color = CLASS_COLORS.get(t.class_name, CLASS_COLORS["default"])
                for k in range(1, len(pts)):
                    thick_k = max(2, int(round(k / 4.0 * scale_up)))
                    cv2.line(viz_frame, pts[k-1], pts[k], color, thick_k, lineType=cv2.LINE_AA)

            # Bounding boxes with high-contrast badge labels
            for t in active_tracks:
                tid = t.track_id
                cls_name = t.class_name
                color = CLASS_COLORS.get(cls_name, CLASS_COLORS["default"])
                bx1 = int(t.current_bbox[0] * scale_up)
                by1 = int(t.current_bbox[1] * scale_up)
                bx2 = int(t.current_bbox[2] * scale_up)
                by2 = int(t.current_bbox[3] * scale_up)

                box_thick = max(2, int(round(2.2 * scale_up)))
                cv2.rectangle(viz_frame, (bx1, by1), (bx2, by2), color, box_thick, lineType=cv2.LINE_AA)

                # Corner brackets
                c_len = min(int(min(bx2 - bx1, by2 - by1) * 0.22), int(16 * scale_up))
                if c_len > 4:
                    cv2.line(viz_frame, (bx1, by1), (bx1 + c_len, by1), color, box_thick + 1, lineType=cv2.LINE_AA)
                    cv2.line(viz_frame, (bx1, by1), (bx1, by1 + c_len), color, box_thick + 1, lineType=cv2.LINE_AA)
                    cv2.line(viz_frame, (bx2, by1), (bx2 - c_len, by1), color, box_thick + 1, lineType=cv2.LINE_AA)
                    cv2.line(viz_frame, (bx2, by1), (bx2, by1 + c_len), color, box_thick + 1, lineType=cv2.LINE_AA)
                    cv2.line(viz_frame, (bx1, by2), (bx1 + c_len, by2), color, box_thick + 1, lineType=cv2.LINE_AA)
                    cv2.line(viz_frame, (bx1, by2), (bx1, by2 - c_len), color, box_thick + 1, lineType=cv2.LINE_AA)
                    cv2.line(viz_frame, (bx2, by2), (bx2 - c_len, by2), color, box_thick + 1, lineType=cv2.LINE_AA)
                    cv2.line(viz_frame, (bx2, by2), (bx2, by2 - c_len), color, box_thick + 1, lineType=cv2.LINE_AA)

                speed_val = getattr(t, "speed", 0.0)
                speed_str = f" | {speed_val:.1f}m/s" if speed_val > 0.05 else ""
                label = f"#{tid} {cls_name.upper()} ({int(t.confidence*100)}%){speed_str}"

                lbl_scale = font_scale * 0.88
                (lw, lh), _ = cv2.getTextSize(label, font, lbl_scale, text_thick)
                pad_h = int(10 * scale_up)
                pad_v = int(6 * scale_up)

                if by1 - lh - pad_v * 2 - 4 > 0:
                    ly1 = by1 - lh - pad_v * 2 - 4
                    ly2 = by1
                    txt_y = by1 - pad_v - 2
                else:
                    ly1 = by1
                    ly2 = by1 + lh + pad_v * 2 + 4
                    txt_y = by1 + lh + pad_v + 2

                lx1 = max(0, bx1)
                lx2 = min(viz_w - 1, bx1 + lw + pad_h * 2)
                txt_x = lx1 + pad_h

                cv2.rectangle(viz_frame, (lx1, ly1), (lx2, ly2), (32, 18, 11), -1) # #0B1220
                cv2.rectangle(viz_frame, (lx1, ly1), (lx2, ly2), color, max(1, text_thick - 1), lineType=cv2.LINE_AA)
                cv2.putText(viz_frame, label, (txt_x + 1, txt_y + 1), font, lbl_scale, (0, 0, 0), text_thick + 1, lineType=cv2.LINE_AA)
                cv2.putText(viz_frame, label, (txt_x, txt_y), font, lbl_scale, (255, 255, 255), text_thick, lineType=cv2.LINE_AA)

            # Top HUD Card
            hud_pad = int(12 * scale_up)
            hud_w = min(viz_w - hud_pad * 2, int(max(400, viz_w * 0.88)))
            hud_lh = int(base_lh * 1.55)
            hud_h = hud_lh * 2 + int(pad_v * 2.5)

            hx1 = hud_pad
            hy1 = hud_pad
            hx2 = hx1 + hud_w
            hy2 = hy1 + hud_h

            cv2.rectangle(viz_frame, (hx1, hy1), (hx2, hy2), (32, 18, 11), -1) # #0B1220
            cv2.rectangle(viz_frame, (hx1, hy1), (hx2, hy2), (77, 54, 38), 1, lineType=cv2.LINE_AA) # #26364D
            cv2.rectangle(viz_frame, (hx1, hy1), (hx2, hy1 + max(3, int(3 * scale_up))), (246, 130, 59), -1) # Blue bar

            l1_y = hy1 + int(hud_lh * 0.95) + pad_v
            l1_text = f"VigiAI Tracker: Frame {saved_frame_idx:03d}/{total_raw_frames//stride}"
            cv2.putText(viz_frame, l1_text, (hx1 + pad_h + 1, l1_y + 1), font, font_scale * 0.92, (0, 0, 0), text_thick + 1, lineType=cv2.LINE_AA)
            cv2.putText(viz_frame, l1_text, (hx1 + pad_h, l1_y), font, font_scale * 0.92, (246, 130, 59), text_thick, lineType=cv2.LINE_AA)

            l2_y = l1_y + hud_lh
            l2_text = f"Active: {len(active_tracks)} | Total Unique IDs: {len(unique_track_ids)}"
            cv2.putText(viz_frame, l2_text, (hx1 + pad_h + 1, l2_y + 1), font, font_scale * 0.82, (0, 0, 0), text_thick, lineType=cv2.LINE_AA)
            cv2.putText(viz_frame, l2_text, (hx1 + pad_h, l2_y), font, font_scale * 0.82, (184, 163, 148), max(1, text_thick - 1), lineType=cv2.LINE_AA)

            # Save JPEG frame
            out_img_path = os.path.join(frames_out_dir, f"frame_{saved_frame_idx:03d}.jpg")
            cv2.imwrite(out_img_path, viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

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
