import os
import sys
# Ensure local workspace is first in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import argparse
import cv2
from src.perception.detector import SafetyDetector, PPEInspector


def run_perception_on_clip(video_path: str, detector: SafetyDetector, out_dir: str):
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_id = 0
    detections_by_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_id / fps, 3)
        # Primary YOLOv8 detection
        dets = detector.detect_frame(frame, frame_id, timestamp)

        # Enhance person detections with PPE classification
        h, w = frame.shape[:2]
        serializable_dets = []
        for d in dets:
            d_dict = d.to_dict()
            if d.class_name == "person":
                x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                has_h, h_conf, has_v, v_conf = PPEInspector.inspect_person_crop(crop)
                d_dict["has_helmet"] = has_h
                d_dict["helmet_conf"] = round(h_conf, 3)
                d_dict["has_vest"] = has_v
                d_dict["vest_conf"] = round(v_conf, 3)

            serializable_dets.append(d_dict)

        detections_by_frame.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "detections": serializable_dets
        })
        frame_id += 1

    cap.release()

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{clip_name}_detections.json")
    with open(out_file, "w") as f:
        json.dump({
            "clip_id": clip_name,
            "fps": fps,
            "total_frames": frame_id,
            "frames": detections_by_frame
        }, f, indent=2)

    print(f"Exported perception detections: {out_file} ({frame_id} frames)")


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 perception on construction videos")
    parser.add_argument("--video-dir", default="data/raw", help="Path to video directory")
    parser.add_argument("--out-dir", default="data/detections", help="Output directory for detections")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    detector = SafetyDetector(conf_threshold=args.conf)
    videos = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    if not videos:
        print(f"No videos found in {args.video_dir}")
        return

    print(f"Found {len(videos)} video(s). Running perception...")
    for v in sorted(videos):
        run_perception_on_clip(v, detector, args.out_dir)


if __name__ == "__main__":
    main()
