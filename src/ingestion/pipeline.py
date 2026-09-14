"""HazardMesh Ingestion Pipeline Orchestrator.
Chains dataset acquisition (Kaggle or Local), video normalization, perception,
tracking, feature engineering, dataset manifest registration, and optional retraining.
"""
import os
import sys
import json
from typing import List, Dict, Any, Optional

from src.ingestion.kaggle_ingestor import KaggleIngestor
from src.ingestion.local_ingestor import LocalIngestor
from src.perception.detector import SafetyDetector, PPEInspector
from src.tracking.tracker import HazardTracker
from src.features.feature_extractor import FeatureExtractor, HazardFeatures
from scripts.run_perception import run_perception_on_clip
from scripts.run_tracking import run_tracking_on_clip
from scripts.build_features import build_features_for_clip
from scripts.train_model import train_v3_model


class IngestionPipeline:
    """End-to-end ingestion and dataset registration pipeline."""

    def __init__(self):
        self.kaggle_ingestor = KaggleIngestor()
        self.local_ingestor = LocalIngestor()
        self.detector = SafetyDetector(conf_threshold=0.25)
        self.extractor = FeatureExtractor()

    def ingest_from_kaggle(self, dataset_slug: str, target_split: str = "train", auto_train: bool = False, max_clips: Optional[int] = 15) -> Dict[str, Any]:
        """Downloads a Kaggle dataset, processes videos/images, and registers them into HazardMesh."""
        extracted_path = self.kaggle_ingestor.download_dataset(dataset_slug)

        # Look for video files in the extracted dataset
        found_videos = []
        for root, _, files in os.walk(extracted_path):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    found_videos.append(os.path.join(root, f))

        slug_prefix = dataset_slug.split("/")[-1]
        processed_clips = []

        if found_videos:
            for v in found_videos:
                clips = self.local_ingestor.standardize_and_chunk_video(v, clip_prefix=f"kaggle_{slug_prefix}")
                processed_clips.extend(clips)
        else:
            # If dataset consists of image folders (e.g. Roboflow image dataset),
            # compile image sequences into standardized 4-second video clips
            images = []
            for root, _, files in os.walk(extracted_path):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        images.append(os.path.join(root, f))
            if images:
                print(f"Found {len(images)} images in {dataset_slug}. Compiling into video sequence...")
                clip_path = self._compile_images_to_video(images[:100], f"kaggle_{slug_prefix}_seq01")
                if clip_path:
                    processed_clips.append(clip_path)

        if max_clips and len(processed_clips) > max_clips:
            print(f"Selecting top {max_clips} clips out of {len(processed_clips)} for ingestion...")
            processed_clips = processed_clips[:max_clips]

        return self._process_and_register_clips(processed_clips, target_split=target_split, source_tag=f"kaggle:{dataset_slug}", auto_train=auto_train)

    def ingest_from_local(self, file_or_dir: str, target_split: str = "train", auto_train: bool = False) -> Dict[str, Any]:
        """Ingests a local video file, directory, or ZIP archive."""
        if not os.path.exists(file_or_dir):
            raise FileNotFoundError(f"Input path does not exist: {file_or_dir}")

        processed_clips = []
        if os.path.isdir(file_or_dir):
            for root, _, files in os.walk(file_or_dir):
                for f in files:
                    if f.lower().endswith((".mp4", ".avi", ".mov")):
                        clips = self.local_ingestor.standardize_and_chunk_video(os.path.join(root, f))
                        processed_clips.extend(clips)
        elif file_or_dir.lower().endswith(".zip"):
            clips = self.local_ingestor.ingest_archive(file_or_dir)
            processed_clips.extend(clips)
        elif file_or_dir.lower().endswith((".mp4", ".avi", ".mov")):
            clips = self.local_ingestor.standardize_and_chunk_video(file_or_dir)
            processed_clips.extend(clips)
        else:
            raise ValueError(f"Unsupported file format: {file_or_dir}")

        return self._process_and_register_clips(processed_clips, target_split=target_split, source_tag="local_upload", auto_train=auto_train)

    def _compile_images_to_video(self, image_paths: List[str], clip_name: str, fps: float = 25.0) -> Optional[str]:
        """Compiles a list of image files into a 640x360 @ 25fps MP4 video clip."""
        import cv2
        if not image_paths:
            return None
        out_path = os.path.join("data/raw", f"{clip_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (640, 360))
        for img_p in image_paths:
            img = cv2.imread(img_p)
            if img is not None:
                img_resized = cv2.resize(img, (640, 360))
                out.write(img_resized)
        out.release()
        return out_path

    def _process_and_register_clips(self, clip_paths: List[str], target_split: str, source_tag: str, auto_train: bool) -> Dict[str, Any]:
        """Runs perception, tracking, feature extraction, and registers clips into the dataset manifest."""
        manifest_path = "data/labels/dataset_manifest.json"
        manifest = {"clips": [], "splits": {"train": [], "val": [], "holdout": []}}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

        ingested_records = []

        for clip_path in clip_paths:
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]
            print(f"Processing ingested clip: {clip_name}...")

            # 1. Perception
            run_perception_on_clip(clip_path, self.detector, "data/detections")

            # 2. Tracking
            det_file = os.path.join("data/detections", f"{clip_name}_detections.json")
            run_tracking_on_clip(det_file, "data/tracks")

            # 3. Generate baseline / auto-annotations for the new clip
            label_file = self._generate_pseudo_ground_truth(clip_name, target_split, source_tag)

            # 4. Feature Extraction
            track_file = os.path.join("data/tracks", f"{clip_name}_tracks.json")
            clip_samples = build_features_for_clip(track_file, label_file, self.extractor)

            # Save clip features
            clip_out = os.path.join("data/features", f"{clip_name}_features.json")
            with open(clip_out, "w") as f:
                json.dump({
                    "clip_id": clip_name,
                    "split": target_split,
                    "sample_count": len(clip_samples),
                    "samples": clip_samples
                }, f, indent=2)

            # 5. Register in Manifest
            manifest["clips"] = [c for c in manifest["clips"] if c["clip_id"] != clip_name]
            manifest["clips"].append({
                "clip_id": clip_name,
                "split": target_split,
                "video_file": os.path.basename(clip_path),
                "label_file": f"{clip_name}_labels.json",
                "frames": len(clip_samples),
                "severity": "INGESTED",
                "source": source_tag
            })
            if clip_name not in manifest["splits"][target_split]:
                manifest["splits"][target_split].append(clip_name)

            ingested_records.append({
                "clip_id": clip_name,
                "sample_count": len(clip_samples),
                "video_path": clip_path
            })

        # Save updated manifest
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # 6. Rebuild split features file (e.g. train_features.json)
        self._refresh_split_features(target_split)

        # 7. Optional Retraining
        retrain_result = None
        if auto_train:
            print("Auto-training requested. Retraining V3 PyTorch model...")
            train_v3_model(features_dir="data/features", out_model_dir="models/v3", epochs=60, lr=0.005)
            retrain_result = "Trained V3 successfully on updated dataset."

        return {
            "status": "success",
            "source": source_tag,
            "clips_count": len(ingested_records),
            "clips": ingested_records,
            "target_split": target_split,
            "retrain_status": retrain_result
        }

    def _generate_pseudo_ground_truth(self, clip_name: str, split: str, source_tag: str) -> str:
        """Generates baseline risk labels based on perception detections and proximity rules."""
        det_file = os.path.join("data/detections", f"{clip_name}_detections.json")
        with open(det_file, "r") as f:
            det_data = json.load(f)

        frame_labels = []
        for f in det_data["frames"]:
            fid = f["frame_id"]
            t = f["timestamp"]
            persons = [d for d in f["detections"] if d["class_name"] == "person"]
            machines = [d for d in f["detections"] if d["class_name"] == "machinery"]

            workers_in_frame = []
            for idx, p in enumerate(persons):
                wid = idx + 1
                has_h = p.get("has_helmet", True)
                has_v = p.get("has_vest", True)

                # Measure proximity to nearest machine
                min_norm_dist = 1.0
                if machines:
                    for m in machines:
                        dx = p["center_x"] - m["center_x"]
                        dy = p["center_y"] - m["center_y"]
                        dist = (dx**2 + dy**2)**0.5 / ((640**2 + 360**2)**0.5)
                        if dist < min_norm_dist:
                            min_norm_dist = dist

                # Assign severity
                if min_norm_dist < 0.20 and (not has_h or not has_v):
                    sev = "HIGH"
                    score = 0.90
                    prio = "IMMEDIATE"
                elif not has_h or not has_v:
                    sev = "LOW"
                    score = 0.35
                    prio = "REVIEW"
                else:
                    sev = "NONE"
                    score = 0.05
                    prio = "MONITOR"

                workers_in_frame.append({
                    "worker_id": wid,
                    "frame_id": fid,
                    "timestamp": t,
                    "bbox": [p["x1"], p["y1"], p["x2"], p["y2"]],
                    "has_helmet": has_h,
                    "has_vest": has_v,
                    "distance_to_machine": round(min_norm_dist, 4),
                    "severity": sev,
                    "risk_score": score,
                    "priority": prio
                })

            frame_labels.append({
                "frame_id": fid,
                "timestamp": t,
                "workers": workers_in_frame
            })

        out_label_path = os.path.join("data/labels", f"{clip_name}_labels.json")
        with open(out_label_path, "w") as f:
            json.dump({
                "clip_id": clip_name,
                "split": split,
                "source": source_tag,
                "overall_severity": "INGESTED",
                "overall_risk": 0.5,
                "frames": frame_labels
            }, f, indent=2)

        return out_label_path

    def _refresh_split_features(self, split: str):
        """Aggregates all clip feature files for the given split."""
        manifest_path = "data/labels/dataset_manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        clip_ids = manifest.get("splits", {}).get(split, [])
        all_samples = []

        for cid in clip_ids:
            feat_file = os.path.join("data/features", f"{cid}_features.json")
            if os.path.exists(feat_file):
                with open(feat_file, "r") as f:
                    data = json.load(f)
                    all_samples.extend(data.get("samples", []))

        split_out_file = os.path.join("data/features", f"{split}_features.json")
        with open(split_out_file, "w") as f:
            json.dump({
                "split": split,
                "sample_count": len(all_samples),
                "feature_names": HazardFeatures.FEATURE_NAMES,
                "samples": all_samples
            }, f, indent=2)
        print(f"Refreshed {split} split features: {split_out_file} ({len(all_samples)} total samples).")
