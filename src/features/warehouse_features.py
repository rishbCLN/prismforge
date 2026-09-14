"""DamageMesh Warehouse Kinematic Feature Extractor.
Extracts 16 physics-based spatio-temporal and kinematic features from tracked
cartons, operators, and handling equipment across warehouse loading/unloading clips.
"""
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class WarehouseKinematicFeatureExtractor:
    """Extracts 16 kinematic behavior features from tracked entities in warehouse video."""

    FEATURE_NAMES = [
        "drop_velocity",              # 0: Downward velocity component vy(t)
        "impact_deceleration",        # 1: Rapid deceleration spike after downward motion
        "throw_horizontal_velocity",  # 2: Ballistic horizontal speed vx(t) away from operator
        "separation_rate",            # 3: Rate of distance increase between operator and carton
        "drag_floor_contact",         # 4: Floor contact proximity indicator [0, 1]
        "drag_velocity",              # 5: Lateral velocity while in floor contact
        "trolley_overlap_iou",        # 6: IoU overlap between carton and mechanical trolley
        "kinematic_jerk",             # 7: Acceleration variance / 3rd derivative (rough motion)
        "staging_zone_distance",      # 8: Distance to designated staging zone boundary
        "is_outside_staging",         # 9: Binary indicator (1.0 if outside staging perimeter)
        "operator_distance",          # 10: Euclidean distance to nearest operator
        "handling_duration",          # 11: Cumulative frames carton has been in motion
        "package_area_ratio",         # 12: Bounding box scale proxy for package volume
        "velocity_variance",          # 13: Rolling speed variance over window
        "stationary_after_impact",    # 14: Persistence of zero velocity post-impact
        "tracking_confidence"         # 15: Detection and tracking confidence score
    ]

    STAGING_X_MIN = 40
    STAGING_X_MAX = 260
    STAGING_Y_MIN = 180
    STAGING_Y_MAX = 330
    FLOOR_Y = 290.0

    def __init__(self, window_size: int = 8, frame_w: int = 640, frame_h: int = 360):
        self.window_size = window_size
        self.frame_w = frame_w
        self.frame_h = frame_h
        # History buffers: key = carton_id
        self.history: Dict[int, List[Dict[str, Any]]] = {}

    def reset(self):
        """Clears sliding window history."""
        self.history.clear()

    def extract_features(
        self,
        carton_track: Dict[str, Any],
        operator_track: Optional[Dict[str, Any]] = None,
        trolley_track: Optional[Dict[str, Any]] = None,
        carton_id: int = 101,
        confidence: float = 0.95
    ) -> List[float]:
        """Computes 16 kinematic features for a carton at the current frame."""
        bx = float(carton_track.get("x", 0.0))
        by = float(carton_track.get("y", 0.0))
        bw = float(carton_track.get("w", 38.0))
        bh = float(carton_track.get("h", 30.0))

        op_x = float(operator_track.get("x", bx)) if operator_track else bx
        op_y = float(operator_track.get("y", by)) if operator_track else by

        # Append to carton history
        if carton_id not in self.history:
            self.history[carton_id] = []

        entry = {
            "x": bx,
            "y": by,
            "w": bw,
            "h": bh,
            "op_x": op_x,
            "op_y": op_y
        }
        self.history[carton_id].append(entry)
        if len(self.history[carton_id]) > self.window_size * 2:
            self.history[carton_id].pop(0)

        hist = self.history[carton_id]
        n_hist = len(hist)

        # 1. Drop velocity vy(t)
        if n_hist >= 2:
            dy = (hist[-1]["y"] - hist[-2]["y"])
            drop_vel = max(0.0, dy / (self.frame_h * 0.05)) # Scaled
        else:
            drop_vel = 0.0

        # 2. Impact Deceleration spike
        if n_hist >= 3:
            prev_dy = hist[-2]["y"] - hist[-3]["y"]
            curr_dy = hist[-1]["y"] - hist[-2]["y"]
            if prev_dy > 4.0 and curr_dy < 1.0 and by >= (self.FLOOR_Y - 10):
                impact_decel = min(1.0, prev_dy / 10.0)
            else:
                impact_decel = 0.0
        else:
            impact_decel = 0.0

        # 3. Ballistic Throw horizontal speed vx(t)
        if n_hist >= 2:
            dx = abs(hist[-1]["x"] - hist[-2]["x"])
            throw_vel = min(1.0, dx / (self.frame_w * 0.025))
        else:
            throw_vel = 0.0

        # 4. Separation rate from operator
        if n_hist >= 2:
            dist_curr = math.hypot(bx - op_x, by - op_y)
            dist_prev = math.hypot(hist[-2]["x"] - hist[-2]["op_x"], hist[-2]["y"] - hist[-2]["op_y"])
            sep_rate = min(1.0, max(0.0, (dist_curr - dist_prev) / 25.0))
        else:
            sep_rate = 0.0

        # 5. Drag Floor contact indicator
        floor_dist = abs(by - self.FLOOR_Y)
        drag_floor_contact = 1.0 if floor_dist <= 15.0 else max(0.0, 1.0 - (floor_dist / 60.0))

        # 6. Drag velocity
        if n_hist >= 2 and drag_floor_contact > 0.6:
            dx = abs(hist[-1]["x"] - hist[-2]["x"])
            drag_vel = min(1.0, (dx / 15.0) * drag_floor_contact)
        else:
            drag_vel = 0.0

        # 7. Trolley IoU Overlap
        if trolley_track:
            tx = float(trolley_track.get("x", -999))
            ty = float(trolley_track.get("y", -999))
            tw, th = 32.0, 48.0
            trolley_iou = self._compute_iou(
                (bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2),
                (tx - tw / 2, ty - th / 2, tx + tw / 2, ty + th / 2)
            )
        else:
            trolley_iou = 0.0

        # 8. Kinematic Jerk (acceleration variance / roughness)
        if n_hist >= 4:
            accels = []
            for i in range(1, len(hist) - 1):
                vx1 = hist[i]["x"] - hist[i - 1]["x"]
                vy1 = hist[i]["y"] - hist[i - 1]["y"]
                vx2 = hist[i + 1]["x"] - hist[i]["x"]
                vy2 = hist[i + 1]["y"] - hist[i]["y"]
                ax = vx2 - vx1
                ay = vy2 - vy1
                accels.append(math.hypot(ax, ay))
            jerk = min(1.0, float(np.std(accels)) / 8.0) if len(accels) > 1 else 0.0
        else:
            jerk = 0.0

        # 9 & 10. Staging Zone containment & distance
        is_outside = not (self.STAGING_X_MIN <= bx <= self.STAGING_X_MAX and self.STAGING_Y_MIN <= by <= self.STAGING_Y_MAX)
        outside_staging = 1.0 if is_outside else 0.0

        if is_outside:
            dx_zone = max(0.0, self.STAGING_X_MIN - bx, bx - self.STAGING_X_MAX)
            dy_zone = max(0.0, self.STAGING_Y_MIN - by, by - self.STAGING_Y_MAX)
            zone_dist = min(1.0, math.hypot(dx_zone, dy_zone) / 200.0)
        else:
            zone_dist = 0.0

        # 11. Operator distance (normalized)
        op_dist = min(1.0, math.hypot(bx - op_x, by - op_y) / 300.0)

        # 12. Handling motion duration
        speeds = []
        for i in range(1, len(hist)):
            sp = math.hypot(hist[i]["x"] - hist[i - 1]["x"], hist[i]["y"] - hist[i - 1]["y"])
            speeds.append(sp)
        active_motion_frames = sum(1 for s in speeds if s > 1.2)
        handling_duration = min(1.0, active_motion_frames / float(self.window_size))

        # 13. Package Area Ratio
        pkg_area = (bw * bh) / float(self.frame_w * self.frame_h)
        area_ratio = min(1.0, pkg_area / 0.015)

        # 14. Velocity variance
        vel_var = min(1.0, float(np.var(speeds)) / 25.0) if speeds else 0.0

        # 15. Stationary after impact
        if n_hist >= 5 and impact_decel > 0.4:
            recent_speeds = speeds[-3:] if len(speeds) >= 3 else speeds
            is_stopped = all(s < 1.0 for s in recent_speeds)
            stat_post_impact = 1.0 if is_stopped else 0.0
        else:
            stat_post_impact = 0.0

        # 16. Tracking confidence
        track_conf = float(confidence)

        return [
            round(min(1.0, max(0.0, drop_vel)), 4),
            round(min(1.0, max(0.0, impact_decel)), 4),
            round(min(1.0, max(0.0, throw_vel)), 4),
            round(min(1.0, max(0.0, sep_rate)), 4),
            round(min(1.0, max(0.0, drag_floor_contact)), 4),
            round(min(1.0, max(0.0, drag_vel)), 4),
            round(min(1.0, max(0.0, trolley_iou)), 4),
            round(min(1.0, max(0.0, jerk)), 4),
            round(min(1.0, max(0.0, zone_dist)), 4),
            round(outside_staging, 4),
            round(min(1.0, max(0.0, op_dist)), 4),
            round(min(1.0, max(0.0, handling_duration)), 4),
            round(min(1.0, max(0.0, area_ratio)), 4),
            round(min(1.0, max(0.0, vel_var)), 4),
            round(stat_post_impact, 4),
            round(track_conf, 4)
        ]

    @staticmethod
    def _compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """Calculates Intersection over Union between two axis-aligned bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h
        if inter_area == 0.0:
            return 0.0

        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = boxA_area + boxB_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
