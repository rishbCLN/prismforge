"""HazardMesh Construction Video & Ground-Truth Dataset Generator.
Synthesizes realistic construction video clips and ground-truth risk labels
for training, validation, and held-out evaluation with strict clip-level partitioning.
"""
import os
import json
import math
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple


class SyntheticConstructionScenario:
    """Generates synthetic construction video clips with realistic geometry and ground truth."""

    WIDTH = 640
    HEIGHT = 360
    FPS = 25

    def __init__(self, output_dir: str = "data/raw", labels_dir: str = "data/labels"):
        self.output_dir = output_dir
        self.labels_dir = labels_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

    @staticmethod
    def draw_construction_background(w: int, h: int) -> np.ndarray:
        """Draws realistic construction site environment with textures and safety zones."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Ground: earthy gravel / dirt gradient
        for y in range(h):
            factor = y / float(h)
            b = int(70 + 20 * factor)
            g = int(95 + 30 * factor)
            r = int(120 + 40 * factor)
            img[y, :] = (b, g, r)

        # Concrete slab / foundation zone on the left
        cv2.rectangle(img, (20, 40), (220, h - 40), (140, 140, 145), -1)
        # Concrete grid lines
        for y in range(60, h - 40, 40):
            cv2.line(img, (20, y), (220, y), (110, 110, 115), 1)

        # Safety corridor / green walkway
        walkway_pts = np.array([[20, h - 60], [220, h - 60], [220, h - 25], [20, h - 25]])
        cv2.fillPoly(img, [walkway_pts], (60, 120, 60))
        cv2.putText(img, "SAFE WALKWAY", (35, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

        # Hazard stripes along top barrier
        for x in range(0, w, 30):
            pts = np.array([[x, 0], [x + 15, 0], [x, 20], [x - 15, 20]], dtype=np.int32)
            cv2.fillPoly(img, [pts], (20, 200, 240)) # Yellow

        return img

    @staticmethod
    def draw_worker(img: np.ndarray, x: int, y: int, has_helmet: bool, has_vest: bool, worker_id: int):
        """Draws a worker with realistic body, hardhat, and high-visibility vest."""
        # Body (torso + legs)
        # Legs
        cv2.line(img, (x - 6, y + 25), (x - 8, y + 50), (40, 40, 80), 4) # Denim pants
        cv2.line(img, (x + 6, y + 25), (x + 8, y + 50), (40, 40, 80), 4)

        # Torso / Vest
        torso_color = (30, 200, 230) if has_vest else (80, 80, 80) # High-vis orange/yellow vs dark shirt
        cv2.rectangle(img, (x - 12, y - 5), (x + 12, y + 25), torso_color, -1)
        if has_vest:
            # Reflective silver stripes on vest
            cv2.line(img, (x - 12, y + 5), (x + 12, y + 5), (230, 230, 230), 2)
            cv2.line(img, (x - 12, y + 15), (x + 12, y + 15), (230, 230, 230), 2)

        # Arms
        cv2.line(img, (x - 12, y), (x - 18, y + 20), (50, 60, 90), 3)
        cv2.line(img, (x + 12, y), (x + 18, y + 20), (50, 60, 90), 3)

        # Head / Helmet
        head_y = y - 16
        cv2.circle(img, (x, head_y), 9, (140, 175, 210), -1) # Skin tone
        if has_helmet:
            # Bright yellow hardhat with brim
            cv2.ellipse(img, (x, head_y - 3), (11, 8), 0, 180, 360, (20, 220, 245), -1)
            cv2.line(img, (x - 12, head_y - 2), (x + 12, head_y - 2), (20, 220, 245), 2)

        # ID tag
        cv2.putText(img, f"W#{worker_id}", (x - 15, y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    @staticmethod
    def draw_excavator(img: np.ndarray, x: int, y: int, boom_angle_deg: float):
        """Draws heavy machinery (excavator) with rotating boom and danger proximity radius."""
        # Danger proximity perimeter (subtle translucent red zone)
        overlay = img.copy()
        danger_radius = 110
        cv2.circle(overlay, (x + 30, y + 10), danger_radius, (0, 0, 180), 2)
        cv2.circle(overlay, (x + 30, y + 10), danger_radius, (0, 0, 80), -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        # Excavator tracks (base)
        cv2.rectangle(img, (x - 45, y + 25), (x + 55, y + 45), (35, 35, 40), -1)
        for tx in range(x - 40, x + 50, 15):
            cv2.circle(img, (tx, y + 35), 6, (70, 70, 75), -1)

        # Main Cabin (Yellow construction body)
        cv2.rectangle(img, (x - 30, y - 20), (x + 35, y + 25), (10, 180, 230), -1)
        # Operator Glass
        cv2.rectangle(img, (x + 5, y - 15), (x + 30, y + 10), (190, 180, 140), -1)

        # Articulated Boom / Arm
        rad = math.radians(boom_angle_deg)
        boom_len = 65
        arm_len = 50
        joint_x = x + 30
        joint_y = y - 5

        elbow_x = int(joint_x + boom_len * math.cos(rad))
        elbow_y = int(joint_y - boom_len * math.sin(rad))

        bucket_x = int(elbow_x + arm_len * math.cos(rad - 0.7))
        bucket_y = int(elbow_y + arm_len * math.sin(rad - 0.7))

        cv2.line(img, (joint_x, joint_y), (elbow_x, elbow_y), (10, 170, 220), 8)
        cv2.line(img, (elbow_x, elbow_y), (bucket_x, bucket_y), (10, 170, 220), 6)

        # Bucket
        cv2.line(img, (bucket_x, bucket_y), (bucket_x + 15, bucket_y + 10), (40, 40, 45), 6)
        cv2.line(img, (bucket_x + 15, bucket_y + 10), (bucket_x + 5, bucket_y + 20), (40, 40, 45), 4)

        cv2.putText(img, "EXCAVATOR #1", (x - 35, y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 200, 255), 1)

    def generate_all_clips(self) -> Dict[str, Any]:
        """Generates the full clip library with strict train / val / holdout splits."""
        clip_definitions = [
            # 1. Train clips (~70%)
            {
                "clip_id": "clip_01_safe_walkway",
                "split": "train",
                "duration_s": 4.0,
                "description": "Safe workers walking inside green corridor with full PPE.",
                "ground_truth_severity": "NONE",
                "ground_truth_risk": 0.05,
                "ground_truth_priority": "MONITOR",
                "workers": [
                    {"id": 1, "start": (60, 280), "end": (160, 280), "helmet": True, "vest": True},
                    {"id": 2, "start": (110, 290), "end": (200, 290), "helmet": True, "vest": True}
                ],
                "machinery": {"pos": (460, 180), "boom_swing": 10.0}
            },
            {
                "clip_id": "clip_02_missing_helmet_safe_zone",
                "split": "train",
                "duration_s": 4.0,
                "description": "Worker missing helmet in safe walkway far from equipment.",
                "ground_truth_severity": "LOW",
                "ground_truth_risk": 0.28,
                "ground_truth_priority": "REVIEW",
                "workers": [
                    {"id": 3, "start": (50, 275), "end": (180, 275), "helmet": False, "vest": True}
                ],
                "machinery": {"pos": (480, 160), "boom_swing": 5.0}
            },
            {
                "clip_id": "clip_03_critical_danger_zone",
                "split": "train",
                "duration_s": 4.0,
                "description": "Worker missing helmet and vest lingering inside excavator swing radius.",
                "ground_truth_severity": "HIGH",
                "ground_truth_risk": 0.92,
                "ground_truth_priority": "IMMEDIATE",
                "workers": [
                    {"id": 4, "start": (410, 195), "end": (430, 205), "helmet": False, "vest": False}
                ],
                "machinery": {"pos": (440, 170), "boom_swing": 25.0}
            },
            {
                "clip_id": "clip_04_transient_crossing",
                "split": "train",
                "duration_s": 4.0,
                "description": "Worker with missing helmet briefly cuts across machinery edge for 0.4s then exits.",
                "ground_truth_severity": "LOW",
                "ground_truth_risk": 0.32,
                "ground_truth_priority": "REVIEW",
                "workers": [
                    {"id": 5, "start": (250, 140), "end": (580, 290), "helmet": False, "vest": True} # Brisk diagonal pass
                ],
                "machinery": {"pos": (380, 170), "boom_swing": 15.0}
            },
            {
                "clip_id": "clip_05_multi_worker_danger",
                "split": "train",
                "duration_s": 4.0,
                "description": "Multiple workers around active excavator without vests and prolonged exposure.",
                "ground_truth_severity": "HIGH",
                "ground_truth_risk": 0.88,
                "ground_truth_priority": "IMMEDIATE",
                "workers": [
                    {"id": 6, "start": (360, 210), "end": (380, 220), "helmet": True, "vest": False},
                    {"id": 7, "start": (410, 225), "end": (405, 230), "helmet": False, "vest": False}
                ],
                "machinery": {"pos": (420, 160), "boom_swing": 30.0}
            },
            {
                "clip_id": "clip_06_safe_team_perimeter",
                "split": "train",
                "duration_s": 3.5,
                "description": "Full PPE crew surveying perimeter at safe distance.",
                "ground_truth_severity": "NONE",
                "ground_truth_risk": 0.08,
                "ground_truth_priority": "MONITOR",
                "workers": [
                    {"id": 8, "start": (70, 160), "end": (150, 160), "helmet": True, "vest": True},
                    {"id": 9, "start": (110, 140), "end": (190, 140), "helmet": True, "vest": True}
                ],
                "machinery": {"pos": (500, 180), "boom_swing": 8.0}
            },
            {
                "clip_id": "clip_07_vest_missing_moderate_dist",
                "split": "train",
                "duration_s": 4.0,
                "description": "Worker missing vest at moderate distance from operating excavator.",
                "ground_truth_severity": "MEDIUM",
                "ground_truth_risk": 0.54,
                "ground_truth_priority": "REVIEW",
                "workers": [
                    {"id": 10, "start": (270, 190), "end": (290, 200), "helmet": True, "vest": False}
                ],
                "machinery": {"pos": (460, 170), "boom_swing": 20.0}
            },

            # 2. Validation clips (~15%)
            {
                "clip_id": "clip_08_val_safe_with_noise",
                "split": "val",
                "duration_s": 4.0,
                "description": "Safe workers with full PPE operating in designated corridor with occasional occlusion.",
                "ground_truth_severity": "NONE",
                "ground_truth_risk": 0.09,
                "ground_truth_priority": "MONITOR",
                "workers": [
                    {"id": 11, "start": (80, 270), "end": (170, 270), "helmet": True, "vest": True}
                ],
                "machinery": {"pos": (470, 170), "boom_swing": 10.0}
            },
            {
                "clip_id": "clip_09_val_prolonged_vest_hazard",
                "split": "val",
                "duration_s": 4.0,
                "description": "Worker without vest lingering close to machinery perimeter.",
                "ground_truth_severity": "MEDIUM",
                "ground_truth_risk": 0.62,
                "ground_truth_priority": "REVIEW",
                "workers": [
                    {"id": 12, "start": (310, 180), "end": (320, 185), "helmet": True, "vest": False}
                ],
                "machinery": {"pos": (450, 160), "boom_swing": 22.0}
            },

            # 3. Final Held-Out Test clips (~15%)
            {
                "clip_id": "clip_10_test_severe_hazard",
                "split": "holdout",
                "duration_s": 4.5,
                "description": "Held-out test: Unprotected worker inside excavator swing radius with active movement.",
                "ground_truth_severity": "HIGH",
                "ground_truth_risk": 0.94,
                "ground_truth_priority": "IMMEDIATE",
                "workers": [
                    {"id": 13, "start": (390, 190), "end": (400, 200), "helmet": False, "vest": False}
                ],
                "machinery": {"pos": (440, 160), "boom_swing": 35.0}
            },
            {
                "clip_id": "clip_11_test_transient_near_miss",
                "split": "holdout",
                "duration_s": 4.0,
                "description": "Held-out test: Worker briskly crossing machine fringe (transient exposure, not sustained).",
                "ground_truth_severity": "LOW",
                "ground_truth_risk": 0.34,
                "ground_truth_priority": "REVIEW",
                "workers": [
                    {"id": 14, "start": (220, 130), "end": (540, 280), "helmet": False, "vest": True}
                ],
                "machinery": {"pos": (370, 180), "boom_swing": 12.0}
            },
            {
                "clip_id": "clip_12_test_safe_perimeter",
                "split": "holdout",
                "duration_s": 4.0,
                "description": "Held-out test: Fully compliant worker walking safe path.",
                "ground_truth_severity": "NONE",
                "ground_truth_risk": 0.04,
                "ground_truth_priority": "MONITOR",
                "workers": [
                    {"id": 15, "start": (60, 285), "end": (160, 285), "helmet": True, "vest": True}
                ],
                "machinery": {"pos": (480, 180), "boom_swing": 10.0}
            }
        ]

        metadata = {"clips": [], "splits": {"train": [], "val": [], "holdout": []}}

        for cdef in clip_definitions:
            clip_id = cdef["clip_id"]
            split = cdef["split"]
            metadata["splits"][split].append(clip_id)

            video_filename = f"{clip_id}.mp4"
            video_path = os.path.join(self.output_dir, video_filename)

            # Generate video file
            total_frames = int(cdef["duration_s"] * self.FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, self.FPS, (self.WIDTH, self.HEIGHT))

            clip_labels = []

            for frame_idx in range(total_frames):
                t = frame_idx / float(self.FPS)
                frame = self.draw_construction_background(self.WIDTH, self.HEIGHT)

                # Draw machinery
                mach_cfg = cdef["machinery"]
                mx, my = mach_cfg["pos"]
                base_angle = 35.0
                swing_amplitude = mach_cfg["boom_swing"]
                current_angle = base_angle + swing_amplitude * math.sin(t * 2.5)
                self.draw_excavator(frame, mx, my, current_angle)

                frame_worker_labels = []

                # Draw each worker
                for wcfg in cdef["workers"]:
                    wid = wcfg["id"]
                    sx, sy = wcfg["start"]
                    ex, ey = wcfg["end"]
                    alpha = min(1.0, frame_idx / float(total_frames - 1))
                    wx = int(sx + alpha * (ex - sx))
                    wy = int(sy + alpha * (ey - sy))

                    has_h = wcfg["helmet"]
                    has_v = wcfg["vest"]
                    self.draw_worker(frame, wx, wy, has_h, has_v, wid)

                    # Compute bounding box
                    bx1 = max(0, wx - 18)
                    by1 = max(0, wy - 30)
                    bx2 = min(self.WIDTH, wx + 18)
                    by2 = min(self.HEIGHT, wy + 52)

                    # Ground truth for this worker at this frame
                    # Calculate true distance to machine center (mx + 30, my + 10)
                    dist_to_mach = math.sqrt((wx - (mx + 30))**2 + (wy - (my + 10))**2)
                    norm_dist = dist_to_mach / math.sqrt(self.WIDTH**2 + self.HEIGHT**2)

                    # Compute true severity label
                    # In safe zone: NONE or LOW
                    # In danger zone with missing PPE: HIGH if sustained, MEDIUM if transient
                    true_sev = cdef["ground_truth_severity"]
                    true_risk = cdef["ground_truth_risk"]
                    true_priority = cdef["ground_truth_priority"]

                    # For transient clip, earlier frames before entering zone are LOW, during zone is MEDIUM, after is LOW
                    if "transient" in clip_id:
                        if norm_dist < 0.22:
                            true_sev = "MEDIUM"
                            true_risk = 0.52
                            true_priority = "REVIEW"
                        else:
                            true_sev = "LOW"
                            true_risk = 0.28
                            true_priority = "REVIEW"

                    frame_worker_labels.append({
                        "worker_id": wid,
                        "frame_id": frame_idx,
                        "timestamp": round(t, 3),
                        "bbox": [bx1, by1, bx2, by2],
                        "has_helmet": has_h,
                        "has_vest": has_v,
                        "distance_to_machine": round(norm_dist, 4),
                        "severity": true_sev,
                        "risk_score": true_risk,
                        "priority": true_priority
                    })

                out.write(frame)
                clip_labels.append({
                    "frame_id": frame_idx,
                    "timestamp": round(t, 3),
                    "workers": frame_worker_labels
                })

            out.release()
            print(f"Generated {video_path} ({total_frames} frames)")

            # Save clip labels
            label_file = os.path.join(self.labels_dir, f"{clip_id}_labels.json")
            with open(label_file, "w") as f:
                json.dump({
                    "clip_id": clip_id,
                    "split": split,
                    "duration_s": cdef["duration_s"],
                    "description": cdef["description"],
                    "overall_severity": cdef["ground_truth_severity"],
                    "overall_risk": cdef["ground_truth_risk"],
                    "frames": clip_labels
                }, f, indent=2)

            metadata["clips"].append({
                "clip_id": clip_id,
                "split": split,
                "video_file": video_filename,
                "label_file": f"{clip_id}_labels.json",
                "frames": total_frames,
                "severity": cdef["ground_truth_severity"]
            })

        # Save dataset manifest
        manifest_path = os.path.join(self.labels_dir, "dataset_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Dataset manifest created at {manifest_path}")

        return metadata


if __name__ == "__main__":
    scenario = SyntheticConstructionScenario()
    scenario.generate_all_clips()
