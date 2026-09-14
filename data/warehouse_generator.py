"""DamageMesh Warehouse Video & Ground-Truth Dataset Generator.
Synthesizes realistic warehouse loading/unloading video clips and ground-truth handling
behavior labels for training, validation, and held-out evaluation.
Covers the core 5 behaviors: Dropping, Throwing, Dragging, Rough Handling, and Staging Breaches.
"""
import os
import json
import math
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple


class SyntheticWarehouseScenario:
    """Generates synthetic warehouse loading/unloading video clips with ground-truth kinematics."""

    WIDTH = 640
    HEIGHT = 360
    FPS = 25

    # Designated Safe Staging Zone Coordinates (Polygon)
    STAGING_ZONE = [
        (40, 180),
        (260, 180),
        (260, 330),
        (40, 330)
    ]

    def __init__(self, output_dir: str = "data/raw", labels_dir: str = "data/labels", tracks_dir: str = "data/tracks"):
        self.output_dir = output_dir
        self.labels_dir = labels_dir
        self.tracks_dir = tracks_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(tracks_dir, exist_ok=True)

    @staticmethod
    def draw_warehouse_background(w: int, h: int) -> np.ndarray:
        """Draws realistic warehouse loading dock environment with bay doors and staging area."""
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Upper wall: industrial grey/blue panels
        for y in range(int(h * 0.45)):
            factor = y / (h * 0.45)
            val = int(80 + 20 * factor)
            img[y, :] = (val + 5, val + 10, val + 15)

        # Lower floor: polished concrete epoxy floor
        for y in range(int(h * 0.45), h):
            factor = (y - h * 0.45) / (h * 0.55)
            b = int(120 + 35 * factor)
            g = int(125 + 35 * factor)
            r = int(130 + 35 * factor)
            img[y, :] = (b, g, r)

        # Dock Wall divider line
        cv2.line(img, (0, int(h * 0.45)), (w, int(h * 0.45)), (50, 50, 55), 2)

        # Loading Dock Truck Bay Door on the right
        bay_left = int(w * 0.62)
        bay_right = w - 20
        bay_top = 35
        bay_bottom = int(h * 0.85)

        # Roll-up door slats
        cv2.rectangle(img, (bay_left, bay_top), (bay_right, bay_bottom), (45, 45, 50), -1)
        for y in range(bay_top + 15, bay_bottom, 18):
            cv2.line(img, (bay_left, y), (bay_right, y), (65, 65, 70), 2)

        # Truck trailer interior (dark opening)
        trailer_left = bay_left + 15
        trailer_right = bay_right - 15
        trailer_top = bay_top + 30
        trailer_bottom = bay_bottom - 10
        cv2.rectangle(img, (trailer_left, trailer_top), (trailer_right, trailer_bottom), (20, 20, 25), -1)

        # Truck dock bumper rubber cushions
        cv2.rectangle(img, (trailer_left - 8, bay_bottom - 30), (trailer_left, bay_bottom), (10, 10, 15), -1)
        cv2.rectangle(img, (trailer_right, bay_bottom - 30), (trailer_right + 8, bay_bottom), (10, 10, 15), -1)

        # Signage
        cv2.rectangle(img, (bay_left + 40, 10), (bay_right - 40, 32), (30, 90, 180), -1)
        cv2.putText(img, "BAY 04 - UNLOADING", (bay_left + 50, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Designated Staging Area (Yellow border with diagonal hazard stripes)
        sz_pts = np.array([[40, 180], [260, 180], [260, 330], [40, 330]], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [sz_pts], (40, 140, 60))
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
        cv2.polylines(img, [sz_pts], True, (20, 220, 245), 2)
        cv2.putText(img, "DESIGNATED STAGING ZONE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 220, 245), 1)

        # Storage shelving racks on the top left
        for rx in range(40, 250, 50):
            cv2.rectangle(img, (rx, 50), (rx + 42, 140), (70, 75, 85), 2)
            cv2.line(img, (rx, 80), (rx + 42, 80), (80, 85, 95), 2)
            cv2.line(img, (rx, 110), (rx + 42, 110), (80, 85, 95), 2)
            # Some stored goods on shelf
            cv2.rectangle(img, (rx + 4, 85), (rx + 38, 108), (60, 100, 140), -1)

        return img

    @staticmethod
    def draw_operator(img: np.ndarray, x: int, y: int, operator_id: int, pose: str = "standing", is_carrying: bool = False):
        """Draws warehouse operator with realistic posture, safety boots, and high-visibility vest."""
        # Legs
        if pose == "standing":
            cv2.line(img, (x - 6, y + 25), (x - 8, y + 55), (60, 60, 70), 4) # Dark work pants
            cv2.line(img, (x + 6, y + 25), (x + 8, y + 55), (60, 60, 70), 4)
            # Safety boots
            cv2.rectangle(img, (x - 12, y + 52), (x - 5, y + 58), (25, 25, 30), -1)
            cv2.rectangle(img, (x + 5, y + 52), (x + 12, y + 58), (25, 25, 30), -1)
        elif pose == "bending":
            cv2.line(img, (x - 6, y + 20), (x - 14, y + 45), (60, 60, 70), 4)
            cv2.line(img, (x + 4, y + 20), (x - 2, y + 45), (60, 60, 70), 4)
            cv2.rectangle(img, (x - 18, y + 42), (x - 10, y + 48), (25, 25, 30), -1)
            cv2.rectangle(img, (x - 5, y + 42), (x + 2, y + 48), (25, 25, 30), -1)

        # Torso & High-Vis Safety Vest (Bright neon orange/yellow)
        cv2.rectangle(img, (x - 12, y - 5), (x + 12, y + 25), (25, 140, 230), -1)
        # Silver reflective stripes
        cv2.line(img, (x - 12, y + 4), (x + 12, y + 4), (220, 220, 230), 2)
        cv2.line(img, (x - 12, y + 14), (x + 12, y + 14), (220, 220, 230), 2)

        # Arms
        if is_carrying:
            # Reaching forward / holding box
            cv2.line(img, (x - 10, y), (x + 8, y + 12), (45, 55, 80), 3)
            cv2.line(img, (x + 10, y), (x + 16, y + 12), (45, 55, 80), 3)
            # Safety gloves (yellow)
            cv2.circle(img, (x + 9, y + 13), 3, (20, 200, 240), -1)
            cv2.circle(img, (x + 17, y + 13), 3, (20, 200, 240), -1)
        else:
            cv2.line(img, (x - 12, y), (x - 16, y + 22), (45, 55, 80), 3)
            cv2.line(img, (x + 12, y), (x + 16, y + 22), (45, 55, 80), 3)

        # Head & Cap
        head_y = y - 16
        cv2.circle(img, (x, head_y), 9, (140, 175, 210), -1)
        # Blue warehouse cap
        cv2.ellipse(img, (x, head_y - 2), (10, 7), 0, 180, 360, (180, 90, 40), -1)
        cv2.line(img, (x - 6, head_y - 2), (x + 12, head_y - 2), (180, 90, 40), 2)

        # Label
        cv2.putText(img, f"OP#{operator_id}", (x - 16, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    @staticmethod
    def draw_carton(img: np.ndarray, x: int, y: int, w: int, h: int, carton_id: int, tilt_deg: float = 0.0, is_damaged: bool = False):
        """Draws a corrugated cardboard shipping carton with tape seams and barcode label."""
        carton_img = np.zeros((h + 10, w + 10, 3), dtype=np.uint8)

        # Cardboard brown base
        base_color = (60, 125, 175) if not is_damaged else (40, 80, 130)
        cv2.rectangle(carton_img, (5, 5), (w + 5, h + 5), base_color, -1)
        cv2.rectangle(carton_img, (5, 5), (w + 5, h + 5), (35, 80, 115), 1)

        # Packaging tape across center
        cv2.line(carton_img, (5, int(h / 2 + 5)), (w + 5, int(h / 2 + 5)), (50, 160, 210), 3)

        # White shipping barcode label
        lbl_w, lbl_h = int(w * 0.4), int(h * 0.35)
        cv2.rectangle(carton_img, (10, 10), (10 + lbl_w, 10 + lbl_h), (240, 240, 245), -1)
        for ly in range(12, 10 + lbl_h, 3):
            cv2.line(carton_img, (12, ly), (10 + lbl_w - 2, ly), (20, 20, 25), 1)

        # Handling arrows ("THIS SIDE UP" symbol)
        cv2.arrowedLine(carton_img, (w - 10, 22), (w - 10, 10), (25, 25, 30), 1, tipLength=0.4)

        if is_damaged:
            # Draw dent / rupture crack
            cv2.line(carton_img, (w - 15, h - 5), (w - 5, h - 20), (20, 20, 20), 2)
            cv2.circle(carton_img, (w - 10, h - 12), 4, (30, 40, 60), -1)

        # Rotate if tilt_deg is specified
        if abs(tilt_deg) > 0.5:
            M = cv2.getRotationMatrix2D((w / 2 + 5, h / 2 + 5), tilt_deg, 1.0)
            carton_img = cv2.warpAffine(carton_img, M, (w + 10, h + 10))

        # Overlay onto background
        top_left_x = int(x - w / 2)
        top_left_y = int(y - h / 2)

        for cy in range(carton_img.shape[0]):
            for cx in range(carton_img.shape[1]):
                bg_y = top_left_y + cy
                bg_x = top_left_x + cx
                if 0 <= bg_y < img.shape[0] and 0 <= bg_x < img.shape[1]:
                    if np.any(carton_img[cy, cx] > 0):
                        img[bg_y, bg_x] = carton_img[cy, cx]

        cv2.putText(img, f"BOX#{carton_id}", (top_left_x, top_left_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (20, 20, 25), 1)

    @staticmethod
    def draw_trolley(img: np.ndarray, x: int, y: int):
        """Draws a warehouse hand truck / trolley with tubular frame and rubber wheels."""
        # Red tubular frame
        cv2.rectangle(img, (x - 16, y - 28), (x + 16, y + 10), (30, 30, 190), 2)
        cv2.line(img, (x - 16, y - 8), (x + 16, y - 8), (30, 30, 190), 2)
        # Handle
        cv2.line(img, (x - 14, y - 28), (x - 14, y - 40), (40, 40, 45), 3)
        cv2.line(img, (x + 14, y - 28), (x + 14, y - 40), (40, 40, 45), 3)
        cv2.line(img, (x - 15, y - 40), (x + 15, y - 40), (40, 40, 45), 3)
        # Base plate (toe plate)
        cv2.rectangle(img, (x - 18, y + 8), (x + 18, y + 14), (80, 80, 90), -1)
        # Wheels
        cv2.circle(img, (x - 15, y + 18), 7, (20, 20, 25), -1)
        cv2.circle(img, (x + 15, y + 18), 7, (20, 20, 25), -1)
        cv2.circle(img, (x - 15, y + 18), 2, (180, 180, 190), -1)
        cv2.circle(img, (x + 15, y + 18), 2, (180, 180, 190), -1)
        cv2.putText(img, "TROLLEY", (x - 18, y - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (30, 30, 190), 1)

    @staticmethod
    def draw_pallet(img: np.ndarray, x: int, y: int, w: int = 70, h: int = 24):
        """Draws a standard wooden pallet with timber deckboards and stringer notches."""
        # Top deckboard
        cv2.rectangle(img, (x - w // 2, y), (x + w // 2, y + 6), (50, 100, 150), -1)
        # Stringers (blocks)
        cv2.rectangle(img, (x - w // 2 + 2, y + 6), (x - w // 2 + 14, y + 18), (40, 80, 120), -1)
        cv2.rectangle(img, (x - 6, y + 6), (x + 6, y + 18), (40, 80, 120), -1)
        cv2.rectangle(img, (x + w // 2 - 14, y + 6), (x + w // 2 - 2, y + 18), (40, 80, 120), -1)
        # Bottom deckboard
        cv2.rectangle(img, (x - w // 2, y + 18), (x + w // 2, y + 24), (50, 100, 150), -1)

    def generate_all_warehouse_scenarios(self) -> Dict[str, Any]:
        """Synthesizes all 10 warehouse video scenarios and exports frame-by-frame annotations."""
        scenarios_meta = [
            {
                "id": "wh_clip_01_safe_trolley_unloading",
                "split": "train",
                "behavior": "SAFE_HANDLING",
                "risk_level": "LOW",
                "description": "Operator places carton onto hand trolley and wheels it gently into designated staging zone."
            },
            {
                "id": "wh_clip_02_carton_drop_impact",
                "split": "train",
                "behavior": "DROPPING",
                "risk_level": "HIGH",
                "description": "Operator loses grip on carton at 1.1m height; carton drops with gravitational acceleration and hits floor with severe impact."
            },
            {
                "id": "wh_clip_03_carton_throwing_truck",
                "split": "train",
                "behavior": "THROWING",
                "risk_level": "CRITICAL",
                "description": "Operator tosses carton across 3.5 meters into truck bed without support; ballistic flight trajectory."
            },
            {
                "id": "wh_clip_04_carton_floor_dragging",
                "split": "train",
                "behavior": "DRAGGING",
                "risk_level": "HIGH",
                "description": "Operator drags heavy 45-pound carton across concrete epoxy floor without a trolley."
            },
            {
                "id": "wh_clip_05_rough_rapid_handling",
                "split": "train",
                "behavior": "ROUGH_HANDLING",
                "risk_level": "HIGH",
                "description": "Operator violently slams carton down onto pallet with excessive kinetic jerk and velocity shock."
            },
            {
                "id": "wh_clip_06_staging_zone_breach",
                "split": "train",
                "behavior": "ZONE_BREACH",
                "risk_level": "MEDIUM",
                "description": "Carton staged outside designated yellow perimeter in an active traffic corridor."
            },
            {
                "id": "wh_clip_07_val_mixed_handling",
                "split": "val",
                "behavior": "DROPPING_TRANSIENT",
                "risk_level": "HIGH",
                "description": "Validation clip: Operator handles two cartons; second carton slips and drops during palletizing."
            },
            {
                "id": "wh_clip_08_val_prolonged_drag",
                "split": "val",
                "behavior": "DRAGGING",
                "risk_level": "HIGH",
                "description": "Validation clip: Prolonged continuous dragging of furniture carton across dock floor."
            },
            {
                "id": "wh_clip_09_test_severe_toss_drop",
                "split": "test",
                "behavior": "THROWING_AND_DROP",
                "risk_level": "CRITICAL",
                "description": "Held-out test clip: Operator tosses fragile package which impacts wall and crashes to floor."
            },
            {
                "id": "wh_clip_10_test_safe_dock_operations",
                "split": "test",
                "behavior": "SAFE_HANDLING",
                "risk_level": "LOW",
                "description": "Held-out test clip: Two operators executing controlled, compliant team-lift and pallet staging."
            }
        ]

        manifest = {"scenarios": scenarios_meta, "total_clips": len(scenarios_meta)}

        for sc in scenarios_meta:
            clip_id = sc["id"]
            video_path = os.path.join(self.output_dir, f"{clip_id}.mp4")
            labels_path = os.path.join(self.labels_dir, f"{clip_id}_labels.json")
            tracks_path = os.path.join(self.tracks_dir, f"{clip_id}_tracks.json")

            frames, labels, tracks = self._render_scenario(sc)

            # Write MP4
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, self.FPS, (self.WIDTH, self.HEIGHT))
            for f in frames:
                out.write(f)
            out.release()

            # Write Labels
            with open(labels_path, "w") as f:
                json.dump(labels, f, indent=2)

            # Write Tracks
            with open(tracks_path, "w") as f:
                json.dump(tracks, f, indent=2)

            print(f"Generated warehouse scenario: {clip_id} ({len(frames)} frames)")

        # Save manifest
        manifest_path = os.path.join(self.labels_dir, "warehouse_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def _render_scenario(self, sc: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """Renders video frames and synthesizes ground-truth tracks and behavior annotations."""
        clip_id = sc["id"]
        behavior = sc["behavior"]
        num_frames = 100 # 4.0 seconds @ 25 FPS
        frames = []
        frame_labels = []
        frame_tracks = []

        floor_y = 290
        box_w, box_h = 38, 30

        for t in range(num_frames):
            img = self.draw_warehouse_background(self.WIDTH, self.HEIGHT)
            operator_x, operator_y = 380, 240
            box_x, box_y = 400, 240
            trolley_x, trolley_y = 150, 250
            tilt = 0.0
            is_carrying = True
            is_damaged = False
            pose = "standing"

            # Default safe tracking state
            drop_vel = 0.0
            throw_vel = 0.0
            drag_vel = 0.0
            jerk = 0.0
            outside_staging = False
            risk_score = 0.05
            severity = "LOW"
            priority = "P3_INFORMATIONAL"
            active_factors = []

            # --- Scenario-specific physics & kinematics ---
            if behavior == "SAFE_HANDLING":
                # Operator moves with trolley smoothly into staging area
                progress = t / float(num_frames)
                trolley_x = int(420 - 280 * progress)
                operator_x = trolley_x + 35
                box_x = trolley_x
                box_y = trolley_y - 15
                is_carrying = False
                self.draw_trolley(img, trolley_x, trolley_y)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 101, tilt)
                self.draw_operator(img, operator_x, operator_y, 1, pose, is_carrying)
                risk_score = 0.08
                severity = "LOW"
                priority = "P3_INFORMATIONAL"

            elif behavior == "DROPPING":
                # Operator carries carton, loses grip at frame 30, falls, hits floor at frame 45
                self.draw_pallet(img, 150, 290)
                if t < 30:
                    operator_x = 350 - t * 2
                    box_x = operator_x - 15
                    box_y = 240
                    is_carrying = True
                    risk_score = 0.12
                    severity = "LOW"
                elif t < 46:
                    # Free fall acceleration
                    dt = (t - 30)
                    operator_x = 290
                    box_x = 275
                    box_y = int(240 + 0.5 * 0.7 * (dt ** 2)) # Quadratic descent
                    drop_vel = float(dt * 0.7)
                    is_carrying = False
                    tilt = dt * 2.5
                    risk_score = 0.78
                    severity = "HIGH"
                    priority = "P1_IMMEDIATE"
                    active_factors = ["RAPID_VERTICAL_DESCENT", "WORKER_DISENGAGEMENT"]
                else:
                    # On floor after impact
                    operator_x = 290
                    box_x = 275
                    box_y = floor_y
                    is_carrying = False
                    is_damaged = True
                    tilt = 12.0
                    risk_score = 0.88
                    severity = "CRITICAL"
                    priority = "P1_IMMEDIATE"
                    active_factors = ["IMPACT_SHOCK", "UNCONTROLLED_DROP", "STATIONARY_POST_IMPACT"]

                self.draw_operator(img, operator_x, operator_y, 1, pose, is_carrying)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 102, tilt, is_damaged)

            elif behavior == "THROWING":
                # Operator in truck throws box across the dock
                operator_x = 520
                if t < 25:
                    box_x = 505
                    box_y = 230
                    is_carrying = True
                    risk_score = 0.15
                    severity = "LOW"
                elif t < 55:
                    # Ballistic arc
                    ft = (t - 25)
                    box_x = int(505 - 8.5 * ft) # Rapid leftward horizontal velocity
                    box_y = int(230 - 3.5 * ft + 0.18 * (ft ** 2)) # Parabolic
                    throw_vel = 8.5
                    is_carrying = False
                    tilt = ft * 7.0
                    risk_score = 0.94
                    severity = "CRITICAL"
                    priority = "P1_IMMEDIATE"
                    active_factors = ["BALLISTIC_THROW", "HIGH_HORIZONTAL_VELOCITY", "AIRBORNE_UNSUPPORTED"]
                else:
                    # Crashed onto dock floor
                    box_x = 250
                    box_y = floor_y
                    is_carrying = False
                    is_damaged = True
                    tilt = 25.0
                    risk_score = 0.90
                    severity = "CRITICAL"
                    priority = "P1_IMMEDIATE"
                    active_factors = ["CRITICAL_IMPACT_DAMAGE", "ROUGH_HANDLING_THROW"]

                self.draw_operator(img, operator_x, operator_y, 2, pose, is_carrying)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 103, tilt, is_damaged)

            elif behavior == "DRAGGING":
                # Operator bends and drags heavy carton along the floor without lifting or trolley
                progress = min(1.0, t / 80.0)
                box_x = int(480 - 240 * progress)
                box_y = floor_y
                operator_x = box_x - 22
                operator_y = floor_y - 25
                pose = "bending"
                is_carrying = True
                drag_vel = 3.0 if t < 80 else 0.0
                risk_score = 0.72 if t > 15 else 0.35
                severity = "HIGH" if t > 15 else "MEDIUM"
                priority = "P2_CORRECTIVE"
                active_factors = ["FLOOR_FRICTION_DRAG", "MISSING_HANDLING_EQUIPMENT", "ERGONOMIC_STRAIN"]

                self.draw_operator(img, operator_x, operator_y, 3, pose, is_carrying)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 104, 0.0, False)

            elif behavior == "ROUGH_HANDLING":
                # Operator yanks box violently and slams down with high jerk
                operator_x = 320
                if t < 30:
                    box_x = 300
                    box_y = 230
                    is_carrying = True
                    risk_score = 0.20
                    severity = "LOW"
                elif t < 60:
                    # Severe oscillatory jerk / shaking
                    jerk_amp = math.sin(t * 0.9) * 22
                    box_x = int(300 + jerk_amp)
                    box_y = int(220 + math.cos(t * 0.9) * 15)
                    jerk = 18.5
                    is_carrying = True
                    risk_score = 0.82
                    severity = "HIGH"
                    priority = "P1_IMMEDIATE"
                    active_factors = ["EXCESSIVE_JERK", "ROUGH_SLAM_MOTION", "UNSTABLE_FORCE"]
                else:
                    # Slammed down
                    box_x = 300
                    box_y = floor_y - 10
                    is_carrying = False
                    is_damaged = True
                    risk_score = 0.75
                    severity = "HIGH"
                    priority = "P2_CORRECTIVE"
                    active_factors = ["SLAM_IMPACT", "FRAGILE_RISK"]

                self.draw_operator(img, operator_x, operator_y, 1, pose, is_carrying)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 105, 0.0, is_damaged)

            elif behavior == "ZONE_BREACH":
                # Box placed in forbidden zone outside yellow staging boundary
                operator_x = 360
                box_x = 380 # Far outside staging area (40-260)
                box_y = 280
                outside_staging = True
                is_carrying = False
                risk_score = 0.58
                severity = "MEDIUM"
                priority = "P2_CORRECTIVE"
                active_factors = ["OUTSIDE_DESIGNATED_ZONE", "CORRIDOR_OBSTRUCTION", "UNSTAGED_PRODUCT"]

                self.draw_operator(img, operator_x, operator_y, 1, pose, is_carrying)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 106, 0.0)

            elif behavior == "DROPPING_TRANSIENT": # val clip 7
                operator_x = 300
                if t < 40:
                    box_x = 285
                    box_y = 230
                    is_carrying = True
                    risk_score = 0.15
                    severity = "LOW"
                elif t < 55:
                    box_x = 285
                    box_y = int(230 + 0.4 * ((t - 40) ** 2))
                    drop_vel = float((t - 40) * 0.6)
                    is_carrying = False
                    risk_score = 0.76
                    severity = "HIGH"
                    priority = "P1_IMMEDIATE"
                    active_factors = ["UNINTENDED_DROP", "RAPID_DESCENT"]
                else:
                    box_x = 285
                    box_y = floor_y
                    is_carrying = False
                    risk_score = 0.65
                    severity = "HIGH"
                    priority = "P2_CORRECTIVE"
                    active_factors = ["FLOOR_IMPACT"]

                self.draw_operator(img, operator_x, operator_y, 4, pose, is_carrying)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 107, 0.0)

            elif behavior == "DRAGGING" and sc["id"] == "wh_clip_08_val_prolonged_drag":
                progress = min(1.0, t / 90.0)
                box_x = int(500 - 300 * progress)
                box_y = floor_y
                operator_x = box_x - 20
                operator_y = floor_y - 25
                pose = "bending"
                is_carrying = True
                drag_vel = 3.3
                risk_score = 0.75
                severity = "HIGH"
                priority = "P2_CORRECTIVE"
                active_factors = ["PROLONGED_FLOOR_DRAGGING", "HIGH_ABRASION_WEAR"]

                self.draw_operator(img, operator_x, operator_y, 5, pose, is_carrying)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 108, 0.0)

            elif behavior == "THROWING_AND_DROP": # test clip 9
                operator_x = 480
                if t < 20:
                    box_x = 460
                    box_y = 220
                    is_carrying = True
                    risk_score = 0.20
                    severity = "LOW"
                elif t < 45:
                    ft = t - 20
                    box_x = int(460 - 8.0 * ft)
                    box_y = int(220 - 2.5 * ft + 0.15 * (ft ** 2))
                    throw_vel = 8.0
                    is_carrying = False
                    tilt = ft * 6.0
                    risk_score = 0.95
                    severity = "CRITICAL"
                    priority = "P1_IMMEDIATE"
                    active_factors = ["BALLISTIC_THROW_VIOLATION"]
                else:
                    box_x = 260
                    box_y = floor_y
                    is_carrying = False
                    is_damaged = True
                    tilt = 30.0
                    risk_score = 0.92
                    severity = "CRITICAL"
                    priority = "P1_IMMEDIATE"
                    active_factors = ["SEVERE_DROP_IMPACT", "STRUCTURAL_DAMAGE"]

                self.draw_operator(img, operator_x, operator_y, 1, pose, is_carrying)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 109, tilt, is_damaged)

            elif behavior == "SAFE_HANDLING" and sc["id"] == "wh_clip_10_test_safe_dock_operations":
                # Two operators team lifting carton and placing on pallet inside staging zone
                self.draw_pallet(img, 150, 290)
                progress = min(1.0, t / 80.0)
                box_x = int(320 - 170 * progress)
                box_y = int(250 + 25 * math.sin(progress * math.pi))
                op1_x = box_x - 25
                op2_x = box_x + 25
                self.draw_operator(img, op1_x, 250, 1, "standing", True)
                self.draw_operator(img, op2_x, 250, 2, "standing", True)
                self.draw_carton(img, box_x, box_y, box_w, box_h, 110, 0.0)
                risk_score = 0.06
                severity = "LOW"
                priority = "P3_INFORMATIONAL"

            # Overlay HUD Telemetry on video frame
            hud = img.copy()
            cv2.rectangle(hud, (10, 10), (220, 75), (20, 20, 25), -1)
            cv2.addWeighted(hud, 0.75, img, 0.25, 0, img)
            cv2.putText(img, f"PRISM DamageMesh: {clip_id}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 220, 245), 1)
            cv2.putText(img, f"Frame: {t:03d}/100 | Risk: {risk_score:.2f} [{severity}]", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(img, f"Behavior: {behavior}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 220, 120) if severity == 'LOW' else (40, 100, 240), 1)

            frames.append(img)

            # Ground-truth frame label
            frame_labels.append({
                "frame_idx": t,
                "timestamp_sec": round(t / float(self.FPS), 2),
                "behavior": behavior,
                "risk_score": round(risk_score, 3),
                "severity": severity,
                "priority": priority,
                "active_factors": active_factors,
                "is_hazardous": severity in ["HIGH", "CRITICAL"]
            })

            # Ground-truth entity track
            frame_tracks.append({
                "frame_idx": t,
                "operator": {"x": operator_x, "y": operator_y, "pose": pose, "is_carrying": is_carrying},
                "carton": {"x": box_x, "y": box_y, "w": box_w, "h": box_h, "tilt": tilt, "is_damaged": is_damaged},
                "kinematics": {
                    "drop_velocity": round(drop_vel, 2),
                    "throw_velocity": round(throw_vel, 2),
                    "drag_velocity": round(drag_vel, 2),
                    "jerk": round(jerk, 2),
                    "outside_staging": outside_staging
                }
            })

        labels_doc = {
            "clip_id": clip_id,
            "split": sc["split"],
            "behavior": behavior,
            "total_frames": num_frames,
            "fps": self.FPS,
            "ground_truth_risk": sc["risk_level"],
            "frames": frame_labels
        }

        tracks_doc = {
            "clip_id": clip_id,
            "total_frames": num_frames,
            "tracks": frame_tracks
        }

        return frames, labels_doc, tracks_doc


if __name__ == "__main__":
    generator = SyntheticWarehouseScenario()
    manifest = generator.generate_all_warehouse_scenarios()
    print("Warehouse scenario generation completed successfully.")
