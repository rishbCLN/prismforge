import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import argparse
from src.tracking.tracker import WorkerTrack
from src.features.feature_extractor import FeatureExtractor, HazardFeatures


def reconstruct_worker_track(w_dict: dict) -> WorkerTrack:
    track = WorkerTrack(
        track_id=w_dict["track_id"],
        first_seen=w_dict["first_seen"],
        last_seen=w_dict["last_seen"],
        frames_seen=w_dict["frames_seen"],
        current_position=tuple(w_dict["current_position"]),
        bbox=tuple(w_dict["bbox"]),
        associated_ppe=w_dict["associated_ppe"],
        associated_machine=w_dict.get("associated_machine"),
        current_risk_state=w_dict.get("current_risk_state", "NONE"),
        violation_start_time=w_dict.get("violation_start_time"),
        consecutive_violating_frames=w_dict.get("consecutive_violating_frames", 0),
        total_violating_frames=w_dict.get("total_violating_frames", 0)
    )
    return track


def build_features_for_clip(track_file: str, label_file: str, extractor: FeatureExtractor) -> list:
    with open(track_file, "r") as f:
        track_data = json.load(f)

    labels_map = {}
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lbl_data = json.load(f)
            for f_info in lbl_data["frames"]:
                fid = f_info["frame_id"]
                labels_map[fid] = {w["worker_id"]: w for w in f_info["workers"]}

    clip_id = track_data["clip_id"]
    samples = []

    for frame_data in track_data["frames"]:
        frame_id = frame_data["frame_id"]
        timestamp = frame_data["timestamp"]
        raw_workers = frame_data["workers"]
        machines = frame_data["machinery"]

        # Reconstruct WorkerTrack objects
        worker_objs = [reconstruct_worker_track(w) for w in raw_workers]

        for w_obj in worker_objs:
            feat = extractor.extract_features(
                worker=w_obj,
                all_workers=worker_objs,
                all_machines=machines,
                frame_id=frame_id,
                timestamp=timestamp,
                frame_shape=(360, 640)
            )

            # Match with ground truth
            gt_info = labels_map.get(frame_id, {}).get(w_obj.track_id)
            if gt_info is None and labels_map.get(frame_id):
                # Fallback to first available ground truth if track ID mapped
                gt_info = list(labels_map[frame_id].values())[0]

            severity_label = gt_info.get("severity", "NONE") if gt_info else "NONE"
            risk_label = gt_info.get("risk_score", 0.0) if gt_info else 0.0
            priority_label = gt_info.get("priority", "MONITOR") if gt_info else "MONITOR"

            sample = {
                "clip_id": clip_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "worker_id": w_obj.track_id,
                "features": feat.to_dict(),
                "feature_vector": feat.to_vector().tolist(),
                "ground_truth": {
                    "severity": severity_label,
                    "risk_score": risk_label,
                    "priority": priority_label
                }
            }
            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract structured hazard features from tracks")
    parser.add_argument("--tracks-dir", default="data/tracks", help="Input tracks directory")
    parser.add_argument("--labels-dir", default="data/labels", help="Labels directory")
    parser.add_argument("--out-dir", default="data/features", help="Output features directory")
    args = parser.parse_args()

    extractor = FeatureExtractor()
    manifest_path = os.path.join(args.labels_dir, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest {manifest_path} not found. Run generator first.")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    splits = manifest.get("splits", {})
    split_samples = {"train": [], "val": [], "holdout": []}

    os.makedirs(args.out_dir, exist_ok=True)

    for clip_info in manifest["clips"]:
        clip_id = clip_info["clip_id"]
        split = clip_info["split"]
        track_file = os.path.join(args.tracks_dir, f"{clip_id}_tracks.json")
        label_file = os.path.join(args.labels_dir, f"{clip_id}_labels.json")

        if not os.path.exists(track_file):
            print(f"Warning: Track file {track_file} not found. Skipping.")
            continue

        clip_samples = build_features_for_clip(track_file, label_file, extractor)
        split_samples[split].extend(clip_samples)

        # Save clip features
        clip_out = os.path.join(args.out_dir, f"{clip_id}_features.json")
        with open(clip_out, "w") as f:
            json.dump({
                "clip_id": clip_id,
                "split": split,
                "sample_count": len(clip_samples),
                "samples": clip_samples
            }, f, indent=2)

    # Save split datasets
    for split_name, s_list in split_samples.items():
        out_split_file = os.path.join(args.out_dir, f"{split_name}_features.json")
        with open(out_split_file, "w") as f:
            json.dump({
                "split": split_name,
                "sample_count": len(s_list),
                "feature_names": HazardFeatures.FEATURE_NAMES,
                "samples": s_list
            }, f, indent=2)
        print(f"Exported {split_name} split: {out_split_file} ({len(s_list)} samples)")


if __name__ == "__main__":
    main()
