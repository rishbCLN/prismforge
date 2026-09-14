"""VigiAI Warehouse Track History Layer.
Maintains a sliding temporal window of observations and motion kinematics for each tracked object.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class TrackObservation:
    """Single-frame observation entry for track history."""
    frame_id: int
    timestamp: float
    center_x: float
    center_y: float
    center_x_norm: float
    center_y_norm: float
    width: float
    height: float
    width_norm: float
    height_norm: float
    confidence: float
    bbox: Tuple[float, float, float, float] # (x1, y1, x2, y2)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    jerk: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrackHistory:
    """Sliding-window trajectory and kinematic history container for a tracked entity."""

    def __init__(self, track_id: int, max_window_size: int = 30):
        """Initializes track history buffer.

        Args:
            track_id: Unique integer identifier of the track.
            max_window_size: Maximum sliding window length (frames).
        """
        self.track_id = track_id
        self.max_window_size = max(2, max_window_size)
        self.observations: List[TrackObservation] = []

    def __len__(self) -> int:
        return len(self.observations)

    def clear(self):
        """Clears all stored observations."""
        self.observations.clear()

    def add_observation(
        self,
        frame_id: int,
        timestamp: float,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        center_x_norm: float = 0.0,
        center_y_norm: float = 0.0,
        width_norm: float = 0.0,
        height_norm: float = 0.0
    ) -> TrackObservation:
        """Appends a new observation and computes differential kinematics (velocity, acceleration, jerk).

        Args:
            frame_id: Frame index.
            timestamp: Timestamp in seconds.
            center_x: Pixel center x.
            center_y: Pixel center y.
            width: Bounding box width.
            height: Bounding box height.
            confidence: Detector confidence.
            bbox: Bounding box (x1, y1, x2, y2).
            center_x_norm: Normalized center x in [0, 1].
            center_y_norm: Normalized center y in [0, 1].
            width_norm: Normalized width in [0, 1].
            height_norm: Normalized height in [0, 1].

        Returns:
            Computed TrackObservation record.
        """
        vel_x = 0.0
        vel_y = 0.0
        speed = 0.0
        acc_x = 0.0
        acc_y = 0.0
        jerk = 0.0

        if len(self.observations) >= 1:
            prev = self.observations[-1]
            dt = timestamp - prev.timestamp
            # If dt is zero or negative (duplicate timestamp), fallback to 1 frame interval proxy
            if dt <= 1e-5:
                dt = 1.0 / 25.0

            # Normalized coordinate velocities for scale-invariant kinematics
            vel_x = (center_x_norm - prev.center_x_norm) / dt
            vel_y = (center_y_norm - prev.center_y_norm) / dt
            speed = float(np.sqrt(vel_x ** 2 + vel_y ** 2))

            if len(self.observations) >= 2:
                prev_acc_x = prev.acceleration_x
                prev_acc_y = prev.acceleration_y
                acc_x = (vel_x - prev.velocity_x) / dt
                acc_y = (vel_y - prev.velocity_y) / dt

                if len(self.observations) >= 3:
                    jerk_x = (acc_x - prev_acc_x) / dt
                    jerk_y = (acc_y - prev_acc_y) / dt
                    jerk = float(np.sqrt(jerk_x ** 2 + jerk_y ** 2))

        obs = TrackObservation(
            frame_id=frame_id,
            timestamp=round(timestamp, 4),
            center_x=round(center_x, 2),
            center_y=round(center_y, 2),
            center_x_norm=round(center_x_norm, 4),
            center_y_norm=round(center_y_norm, 4),
            width=round(width, 2),
            height=round(height, 2),
            width_norm=round(width_norm, 4),
            height_norm=round(height_norm, 4),
            confidence=round(confidence, 4),
            bbox=tuple(round(v, 2) for v in bbox),
            velocity_x=round(vel_x, 4),
            velocity_y=round(vel_y, 4),
            speed=round(speed, 4),
            acceleration_x=round(acc_x, 4),
            acceleration_y=round(acc_y, 4),
            jerk=round(jerk, 4)
        )

        self.observations.append(obs)
        if len(self.observations) > self.max_window_size:
            self.observations.pop(0)

        return obs

    def get_latest_observation(self) -> Optional[TrackObservation]:
        """Returns the most recent observation in history."""
        return self.observations[-1] if self.observations else None

    def get_trajectory(self, max_points: Optional[int] = None) -> List[Tuple[float, float]]:
        """Returns list of pixel center points (x, y) over recent history."""
        pts = [(o.center_x, o.center_y) for o in self.observations]
        if max_points is not None:
            return pts[-max_points:]
        return pts

    def get_normalized_trajectory(self, max_points: Optional[int] = None) -> List[Tuple[float, float]]:
        """Returns list of normalized center points (x_norm, y_norm) over recent history."""
        pts = [(o.center_x_norm, o.center_y_norm) for o in self.observations]
        if max_points is not None:
            return pts[-max_points:]
        return pts

    def get_speed_history(self, max_points: Optional[int] = None) -> List[float]:
        """Returns list of computed speed values over recent history."""
        speeds = [o.speed for o in self.observations]
        if max_points is not None:
            return speeds[-max_points:]
        return speeds

    def get_mean_speed(self, window: int = 5) -> float:
        """Returns mean speed over the last N observations."""
        if not self.observations:
            return 0.0
        recent = self.observations[-window:]
        return float(np.mean([o.speed for o in recent]))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes track history to machine-readable dictionary."""
        return {
            "track_id": self.track_id,
            "window_size": len(self.observations),
            "max_window_size": self.max_window_size,
            "observations": [o.to_dict() for o in self.observations]
        }
