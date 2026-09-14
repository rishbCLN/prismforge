import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import argparse
from src.perception.detector import Detection
from src.tracking.tracker import HazardTracker


def run_tracking_on_clip(det_file: str, out_dir: str):
    with open(det_file, "r") as f:
        data = json.load(f)

    clip_id = data["clip_id"]
    fps = data.get("fps", 25.0)
    tracker = HazardTracker()

    tracked_frames = []

    for frame_data in data["frames"]:
        frame_id = frame_data["frame_id"]
        timestamp = frame_data["timestamp"]

        detections = []
        for d in frame_data["detections"]:
            det = Detection.from_dict({
                "frame_id": d["frame_id"],
                "timestamp": d["timestamp"],
                "track_id": d.get("track_id"),
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "x1": d["x1"],
                "y1": d["y1"],
                "x2": d["x2"],
                "y2": d["y2"],
                "center_x": d["center_x"],
                "center_y": d["center_y"],
                "width": d["width"],
                "height": d["height"]
            })
            # Attach PPE attributes if present
            det.has_helmet = d.get("has_helmet", True)
            det.helmet_conf = d.get("helmet_conf", 0.9)
            det.has_vest = d.get("has_vest", True)
            det.vest_conf = d.get("vest_conf", 0.9)
            detections.append(det)

        worker_tracks, active_machines = tracker.update(detections, timestamp)

        # Update PPE states on active tracks from detections
        for w_track in worker_tracks:
            # Find matching detection
            for det in detections:
                if det.track_id == w_track.track_id:
                    has_h = getattr(det, "has_helmet", True)
                    h_conf = getattr(det, "helmet_conf", 0.9)
                    has_v = getattr(det, "has_vest", True)
                    v_conf = getattr(det, "vest_conf", 0.9)
                    w_track.update_ppe(
                        helmet_missing=not has_h,
                        vest_missing=not has_v,
                        helmet_conf=h_conf,
                        vest_conf=v_conf,
                        timestamp=timestamp
                    )
                    break

        tracked_frames.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "workers": [w.to_dict() for w in worker_tracks],
            "machinery": active_machines
        })

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{clip_id}_tracks.json")
    with open(out_file, "w") as f:
        json.dump({
            "clip_id": clip_id,
            "fps": fps,
            "total_frames": len(tracked_frames),
            "frames": tracked_frames
        }, f, indent=2)

    print(f"Exported tracking tracks: {out_file} ({len(tracked_frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Run multi-object tracking on detected scenes")
    parser.add_argument("--det-dir", default="data/detections", help="Input detections directory")
    parser.add_argument("--out-dir", default="data/tracks", help="Output tracks directory")
    args = parser.parse_args()

    det_files = glob.glob(os.path.join(args.det_dir, "*_detections.json"))
    if not det_files:
        print(f"No detection files found in {args.det_dir}")
        return

    print(f"Found {len(det_files)} detection files. Running tracking...")
    for df in sorted(det_files):
        run_tracking_on_clip(df, args.out_dir)


if __name__ == "__main__":
    main()
