"""Extracts and compiles 16 kinematic features across all warehouse scenarios.
Splits clips into:
- train (clips 1-6)
- val (clips 7-8)
- test/holdout (clips 9-10)
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features.warehouse_features import WarehouseKinematicFeatureExtractor


def build_all_warehouse_features(
    labels_dir: str = "data/labels",
    tracks_dir: str = "data/tracks",
    output_dir: str = "data/features"
):
    os.makedirs(output_dir, exist_ok=True)
    manifest_file = os.path.join(labels_dir, "warehouse_manifest.json")
    if not os.path.exists(manifest_file):
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    extractor = WarehouseKinematicFeatureExtractor()

    train_samples = []
    val_samples = []
    holdout_samples = []

    for sc in manifest["scenarios"]:
        clip_id = sc["id"]
        split = sc["split"]
        labels_file = os.path.join(labels_dir, f"{clip_id}_labels.json")
        tracks_file = os.path.join(tracks_dir, f"{clip_id}_tracks.json")

        if not os.path.exists(labels_file) or not os.path.exists(tracks_file):
            print(f"Skipping {clip_id}, files missing")
            continue

        with open(labels_file, "r") as f:
            labels_data = json.load(f)
        with open(tracks_file, "r") as f:
            tracks_data = json.load(f)

        extractor.reset()
        clip_samples = []

        labels_list = labels_data["frames"]
        tracks_list = tracks_data["tracks"]

        for i, (lbl, trk) in enumerate(zip(labels_list, tracks_list)):
            carton = trk.get("carton", {})
            operator = trk.get("operator", {})
            trolley = trk.get("trolley", None)

            feats = extractor.extract_features(
                carton_track=carton,
                operator_track=operator,
                trolley_track=trolley,
                carton_id=100 + (i % 10),
                confidence=0.96
            )

            sample = {
                "clip_id": clip_id,
                "frame_idx": lbl["frame_idx"],
                "timestamp_sec": lbl["timestamp_sec"],
                "features": feats,
                "feature_names": WarehouseKinematicFeatureExtractor.FEATURE_NAMES,
                "target_risk_score": lbl["risk_score"],
                "target_severity": lbl["severity"],
                "target_priority": lbl["priority"],
                "is_hazardous": lbl["is_hazardous"],
                "behavior": lbl["behavior"],
                "active_factors": lbl["active_factors"]
            }
            clip_samples.append(sample)

        # Save clip level features
        clip_feat_path = os.path.join(output_dir, f"{clip_id}_features.json")
        with open(clip_feat_path, "w") as f:
            json.dump({"clip_id": clip_id, "split": split, "samples": clip_samples}, f, indent=2)

        if split == "train":
            train_samples.extend(clip_samples)
        elif split == "val":
            val_samples.extend(clip_samples)
        elif split == "test":
            holdout_samples.extend(clip_samples)

        print(f"Processed features for {clip_id} ({len(clip_samples)} frames) -> {split}")

    # Save split datasets
    with open(os.path.join(output_dir, "warehouse_train_features.json"), "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(os.path.join(output_dir, "warehouse_val_features.json"), "w") as f:
        json.dump(val_samples, f, indent=2)
    with open(os.path.join(output_dir, "warehouse_holdout_features.json"), "w") as f:
        json.dump(holdout_samples, f, indent=2)

    print(f"\nFeature compilation complete:")
    print(f"  Train samples:   {len(train_samples)}")
    print(f"  Val samples:     {len(val_samples)}")
    print(f"  Holdout samples: {len(holdout_samples)}")


if __name__ == "__main__":
    build_all_warehouse_features()
