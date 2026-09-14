"""VigiAI Warehouse Object Tracking Layer.
Maintains persistent identities across video frames using ByteTrack multi-stage association,
manages sliding-window trajectory histories, and infers spatial worker/equipment associations.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.perception.yolo import Detection
from src.tracking.history import TrackHistory, TrackObservation


def compute_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])
    union_area = area1 + area2 - intersection_area

    if union_area <= 0.0:
        return 0.0
    return float(intersection_area / union_area)


@dataclass
class WarehouseTrack:
    """Maintains state and persistent history for a tracked warehouse entity."""
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    frames_seen: int
    current_bbox: Tuple[float, float, float, float]
    current_position: Tuple[float, float]
    current_position_norm: Tuple[float, float]
    confidence: float
    time_lost: int = 0
    previous_bbox: Optional[Tuple[float, float, float, float]] = None
    previous_position: Optional[Tuple[float, float]] = None
    history: TrackHistory = field(default_factory=lambda: TrackHistory(track_id=0))
    # Spatial associations
    associated_worker_id: Optional[int] = None
    worker_distance_norm: Optional[float] = None
    associated_equipment_id: Optional[int] = None
    equipment_overlap_iou: float = 0.0

    def update(
        self,
        frame_id: int,
        timestamp: float,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        frame_w: int,
        frame_h: int
    ):
        """Updates track with a new detection observation."""
        self.previous_bbox = self.current_bbox
        self.previous_position = self.current_position
        self.current_bbox = bbox
        self.confidence = confidence
        self.last_seen = timestamp
        self.frames_seen += 1
        self.time_lost = 0

        x1, y1, x2, y2 = bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        self.current_position = (cx, cy)

        w_div = max(1.0, float(frame_w))
        h_div = max(1.0, float(frame_h))
        cx_norm = cx / w_div
        cy_norm = cy / h_div
        w_norm = w / w_div
        h_norm = h / h_div
        self.current_position_norm = (cx_norm, cy_norm)

        self.history.add_observation(
            frame_id=frame_id,
            timestamp=timestamp,
            center_x=cx,
            center_y=cy,
            width=w,
            height=h,
            confidence=confidence,
            bbox=bbox,
            center_x_norm=cx_norm,
            center_y_norm=cy_norm,
            width_norm=w_norm,
            height_norm=h_norm
        )

    def mark_missed(self):
        """Increments missed frame counter."""
        self.time_lost += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serializes track state for machine-readable JSON output."""
        latest_obs = self.history.get_latest_observation()
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "frames_seen": self.frames_seen,
            "time_lost": self.time_lost,
            "bbox": [round(v, 2) for v in self.current_bbox],
            "position": [round(v, 2) for v in self.current_position],
            "position_norm": [round(v, 4) for v in self.current_position_norm],
            "confidence": round(self.confidence, 4),
            "velocity": [latest_obs.velocity_x, latest_obs.velocity_y] if latest_obs else [0.0, 0.0],
            "speed": latest_obs.speed if latest_obs else 0.0,
            "acceleration": [latest_obs.acceleration_x, latest_obs.acceleration_y] if latest_obs else [0.0, 0.0],
            "associated_worker_id": self.associated_worker_id,
            "worker_distance_norm": round(self.worker_distance_norm, 4) if self.worker_distance_norm is not None else None,
            "associated_equipment_id": self.associated_equipment_id,
            "equipment_overlap_iou": round(self.equipment_overlap_iou, 4)
        }


class WarehouseTracker:
    """ByteTrack-inspired multi-stage association tracker for warehouse material handling."""

    def __init__(
        self,
        high_conf_thresh: float = 0.50,
        low_conf_thresh: float = 0.15,
        iou_threshold: float = 0.25,
        max_lost_frames: int = 15,
        history_window_size: int = 16
    ):
        """Initializes warehouse tracker.

        Args:
            high_conf_thresh: Detection confidence threshold for primary matching.
            low_conf_thresh: Detection confidence threshold for recovery matching.
            iou_threshold: Minimum IoU overlap required to match a detection to a track.
            max_lost_frames: Maximum frames a track can be missed before termination.
            history_window_size: Max length of trajectory history window.
        """
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.history_window_size = history_window_size

        self.next_track_id: int = 1
        self.active_tracks: Dict[int, WarehouseTrack] = {}

    def reset(self):
        """Resets tracker state and ID counter."""
        self.next_track_id = 1
        self.active_tracks.clear()

    def update(
        self,
        frame_id: int,
        timestamp: float,
        detections: List[Detection],
        frame_w: int = 640,
        frame_h: int = 360
    ) -> List[WarehouseTrack]:
        """Updates tracks with detections from the current frame using two-stage ByteTrack matching.

        Args:
            frame_id: Current frame index.
            timestamp: Current timestamp in seconds.
            detections: List of Detection objects from perception.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            List of currently active and updated WarehouseTrack objects.
        """
        # Partition detections into high and low confidence sets
        high_dets: List[Detection] = []
        low_dets: List[Detection] = []

        for d in detections:
            if d.confidence >= self.high_conf_thresh:
                high_dets.append(d)
            elif d.confidence >= self.low_conf_thresh:
                low_dets.append(d)

        # Separate tracks by status
        track_ids = list(self.active_tracks.keys())
        matched_track_ids: Set[int] = set()
        matched_det_indices: Set[int] = set()

        # -------------------------------------------------------------
        # Stage 1: Match high-confidence detections to active tracks
        # -------------------------------------------------------------
        if track_ids and high_dets:
            iou_matrix = np.zeros((len(track_ids), len(high_dets)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                t = self.active_tracks[tid]
                for j, d in enumerate(high_dets):
                    # Enforce class consistency where practical
                    if t.class_name == d.class_name:
                        iou_matrix[i, j] = compute_iou(t.current_bbox, (d.x1, d.y1, d.x2, d.y2))
                    else:
                        iou_matrix[i, j] = 0.0

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    tid = track_ids[r]
                    det = high_dets[c]
                    self.active_tracks[tid].update(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        bbox=(det.x1, det.y1, det.x2, det.y2),
                        confidence=det.confidence,
                        frame_w=frame_w,
                        frame_h=frame_h
                    )
                    det.track_id = tid
                    matched_track_ids.add(tid)
                    matched_det_indices.add(c)

        unmatched_high_dets = [d for j, d in enumerate(high_dets) if j not in matched_det_indices]
        unmatched_track_ids = [tid for tid in track_ids if tid not in matched_track_ids]

        # -------------------------------------------------------------
        # Stage 2: Match remaining unmatched tracks to low-confidence detections
        # -------------------------------------------------------------
        if unmatched_track_ids and low_dets:
            low_iou_matrix = np.zeros((len(unmatched_track_ids), len(low_dets)), dtype=np.float32)
            for i, tid in enumerate(unmatched_track_ids):
                t = self.active_tracks[tid]
                for j, d in enumerate(low_dets):
                    if t.class_name == d.class_name:
                        low_iou_matrix[i, j] = compute_iou(t.current_bbox, (d.x1, d.y1, d.x2, d.y2))
                    else:
                        low_iou_matrix[i, j] = 0.0

            row_ind, col_ind = linear_sum_assignment(-low_iou_matrix)
            matched_low_track_ids: Set[int] = set()
            for r, c in zip(row_ind, col_ind):
                if low_iou_matrix[r, c] >= self.iou_threshold:
                    tid = unmatched_track_ids[r]
                    det = low_dets[c]
                    self.active_tracks[tid].update(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        bbox=(det.x1, det.y1, det.x2, det.y2),
                        confidence=det.confidence,
                        frame_w=frame_w,
                        frame_h=frame_h
                    )
                    det.track_id = tid
                    matched_low_track_ids.add(tid)

            unmatched_track_ids = [tid for tid in unmatched_track_ids if tid not in matched_low_track_ids]

        # -------------------------------------------------------------
        # Stage 3: Age unmatched tracks and remove lost ones
        # -------------------------------------------------------------
        for tid in unmatched_track_ids:
            self.active_tracks[tid].mark_missed()

        # Remove dead tracks
        dead_track_ids = [tid for tid, t in self.active_tracks.items() if t.time_lost > self.max_lost_frames]
        for tid in dead_track_ids:
            del self.active_tracks[tid]

        # -------------------------------------------------------------
        # Stage 4: Initialize new tracks for unmatched high-confidence detections
        # -------------------------------------------------------------
        for det in unmatched_high_dets:
            tid = self.next_track_id
            self.next_track_id += 1
            det.track_id = tid

            track = WarehouseTrack(
                track_id=tid,
                class_name=det.class_name,
                first_seen=timestamp,
                last_seen=timestamp,
                frames_seen=0,
                current_bbox=(det.x1, det.y1, det.x2, det.y2),
                current_position=(det.center_x, det.center_y),
                current_position_norm=(det.center_x_norm, det.center_y_norm),
                confidence=det.confidence,
                history=TrackHistory(track_id=tid, max_window_size=self.history_window_size)
            )
            track.update(
                frame_id=frame_id,
                timestamp=timestamp,
                bbox=(det.x1, det.y1, det.x2, det.y2),
                confidence=det.confidence,
                frame_w=frame_w,
                frame_h=frame_h
            )
            self.active_tracks[tid] = track

        # -------------------------------------------------------------
        # Stage 5: Infer spatial interactions (Worker-Carton, Equipment-Carton)
        # -------------------------------------------------------------
        self._infer_spatial_associations()

        # Return active tracks visible in current frame
        return [t for t in self.active_tracks.values() if t.time_lost == 0]

    def _infer_spatial_associations(self):
        """Infers nearest worker and overlapping equipment for all active cartons."""
        workers = [t for t in self.active_tracks.values() if t.class_name == "person" and t.time_lost == 0]
        equipment = [t for t in self.active_tracks.values() if t.class_name in {"pallet", "trolley"} and t.time_lost == 0]
        cartons = [t for t in self.active_tracks.values() if t.class_name == "carton" and t.time_lost == 0]

        for c in cartons:
            # Nearest worker association
            best_w_dist = float("inf")
            best_w_id = None
            cx, cy = c.current_position_norm
            for w in workers:
                wx, wy = w.current_position_norm
                dist = float(np.sqrt((cx - wx) ** 2 + (cy - wy) ** 2))
                if dist < best_w_dist:
                    best_w_dist = dist
                    best_w_id = w.track_id

            c.associated_worker_id = best_w_id
            c.worker_distance_norm = best_w_dist if best_w_id is not None else None

            # Overlapping equipment association
            best_eq_iou = 0.0
            best_eq_id = None
            for eq in equipment:
                iou = compute_iou(c.current_bbox, eq.current_bbox)
                if iou > best_eq_iou:
                    best_eq_iou = iou
                    best_eq_id = eq.track_id

            c.associated_equipment_id = best_eq_id
            c.equipment_overlap_iou = best_eq_iou
