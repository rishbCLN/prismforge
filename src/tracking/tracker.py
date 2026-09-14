"""HazardMesh Tracking Layer.
Maintains persistent worker and machinery identities across video frames,
tracking their spatial movement, PPE status history, and hazard timestamps.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from src.perception.detector import Detection


@dataclass
class WorkerTrack:
    track_id: int
    first_seen: float            # timestamp in seconds
    last_seen: float             # timestamp in seconds
    frames_seen: int
    current_position: Tuple[float, float]   # (center_x, center_y)
    previous_position: Optional[Tuple[float, float]] = None
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0) # (x1, y1, x2, y2)
    associated_ppe: Dict[str, Any] = field(default_factory=lambda: {
        "helmet_missing": False,
        "vest_missing": False,
        "helmet_conf": 0.9,
        "vest_conf": 0.9
    })
    associated_machine: Optional[Dict[str, Any]] = None
    current_risk_state: str = "NONE"
    violation_start_time: Optional[float] = None
    consecutive_violating_frames: int = 0
    total_violating_frames: int = 0
    confidence_history: List[float] = field(default_factory=list)
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    time_lost: int = 0           # consecutive frames missed

    def update_position(self, pos: Tuple[float, float], bbox: Tuple[float, float, float, float], timestamp: float):
        self.previous_position = self.current_position
        self.current_position = pos
        self.bbox = bbox
        self.last_seen = timestamp
        self.frames_seen += 1
        self.time_lost = 0
        self.position_history.append(pos)
        if len(self.position_history) > 60:
            self.position_history.pop(0)

    def update_ppe(self, helmet_missing: bool, vest_missing: bool, helmet_conf: float, vest_conf: float, timestamp: float):
        self.associated_ppe = {
            "helmet_missing": helmet_missing,
            "vest_missing": vest_missing,
            "helmet_conf": helmet_conf,
            "vest_conf": vest_conf
        }
        is_violating = helmet_missing or vest_missing
        if is_violating:
            if self.violation_start_time is None:
                self.violation_start_time = timestamp
            self.consecutive_violating_frames += 1
            self.total_violating_frames += 1
        else:
            self.consecutive_violating_frames = 0
            # If safe for more than 1.0s, reset violation_start_time
            if self.violation_start_time is not None and (timestamp - self.last_seen) > 1.0:
                self.violation_start_time = None

    def get_violation_duration(self, current_time: float) -> float:
        if self.violation_start_time is None or self.consecutive_violating_frames == 0:
            return 0.0
        return max(0.0, current_time - self.violation_start_time)

    def get_persistence_ratio(self) -> float:
        if self.frames_seen == 0:
            return 0.0
        return float(self.total_violating_frames) / float(self.frames_seen)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "frames_seen": self.frames_seen,
            "current_position": self.current_position,
            "bbox": self.bbox,
            "associated_ppe": self.associated_ppe,
            "associated_machine": self.associated_machine,
            "current_risk_state": self.current_risk_state,
            "violation_start_time": self.violation_start_time,
            "consecutive_violating_frames": self.consecutive_violating_frames,
            "total_violating_frames": self.total_violating_frames,
            "violation_duration": self.get_violation_duration(self.last_seen),
            "persistence_ratio": self.get_persistence_ratio()
        }


class HazardTracker:
    """Robust spatio-temporal tracker for workers and machinery."""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.worker_tracks: Dict[int, WorkerTrack] = {}
        self.machinery_tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return float(iou)

    def update(self, detections: List[Detection], timestamp: float) -> Tuple[List[WorkerTrack], List[Dict[str, Any]]]:
        """Updates tracks with new detections from a frame."""
        person_dets = [d for d in detections if d.class_name == "person"]
        machinery_dets = [d for d in detections if d.class_name == "machinery"]

        # 1. Update Person Tracks
        active_worker_ids = set(self.worker_tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        # Match using IoU greedy assignment
        if active_worker_ids and person_dets:
            candidates = []
            for tid in active_worker_ids:
                track = self.worker_tracks[tid]
                for didx, det in enumerate(person_dets):
                    det_box = (det.x1, det.y1, det.x2, det.y2)
                    iou = self.compute_iou(track.bbox, det_box)
                    if iou >= self.iou_threshold:
                        candidates.append((iou, tid, didx))

            # Sort by highest IoU
            candidates.sort(key=lambda x: x[0], reverse=True)
            for iou, tid, didx in candidates:
                if tid not in matched_tracks and didx not in matched_dets:
                    matched_tracks.add(tid)
                    matched_dets.add(didx)
                    det = person_dets[didx]
                    det.track_id = tid
                    track = self.worker_tracks[tid]
                    track.update_position((det.center_x, det.center_y), (det.x1, det.y1, det.x2, det.y2), timestamp)
                    track.confidence_history.append(det.confidence)

        # Unmatched detections -> spawn new tracks
        for didx, det in enumerate(person_dets):
            if didx not in matched_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                det.track_id = tid
                new_track = WorkerTrack(
                    track_id=tid,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    frames_seen=1,
                    current_position=(det.center_x, det.center_y),
                    bbox=(det.x1, det.y1, det.x2, det.y2),
                    confidence_history=[det.confidence]
                )
                self.worker_tracks[tid] = new_track
                matched_tracks.add(tid)

        # Unmatched existing tracks -> increment time lost
        dead_tracks = []
        for tid in list(self.worker_tracks.keys()):
            if tid not in matched_tracks:
                track = self.worker_tracks[tid]
                track.time_lost += 1
                if track.time_lost > self.max_lost_frames:
                    dead_tracks.append(tid)

        for tid in dead_tracks:
            del self.worker_tracks[tid]

        # 2. Update Machinery
        active_machines = []
        for midx, m_det in enumerate(machinery_dets):
            m_dict = {
                "machine_id": midx + 1,
                "bbox": (m_det.x1, m_det.y1, m_det.x2, m_det.y2),
                "center": (m_det.center_x, m_det.center_y),
                "class_name": "excavator", # default machinery subtype
                "confidence": m_det.confidence
            }
            active_machines.append(m_dict)

        # 3. Associate each worker with nearest machine
        for track in self.worker_tracks.values():
            if active_machines:
                nearest_m = None
                min_dist = float("inf")
                wx, wy = track.current_position
                for m in active_machines:
                    mx, my = m["center"]
                    dist = math.sqrt((wx - mx)**2 + (wy - my)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_m = m
                track.associated_machine = {
                    "machine_id": nearest_m["machine_id"],
                    "distance_px": min_dist,
                    "machine_bbox": nearest_m["bbox"],
                    "class_name": nearest_m["class_name"]
                }
            else:
                track.associated_machine = None

        return list(self.worker_tracks.values()), active_machines
