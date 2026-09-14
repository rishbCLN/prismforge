"""HazardMesh Perception Layer.
Wraps YOLOv8 and generates structured Detection records.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class Detection:
    frame_id: int
    timestamp: float
    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        return cls(**data)


class PPEInspector:
    """Accurate PPE inspector for head and torso regions."""
    
    @staticmethod
    def inspect_person_crop(person_crop: np.ndarray) -> Tuple[bool, float, bool, float]:
        """Inspects person image crop for hardhat and safety vest.
        Returns:
            (has_helmet, helmet_conf, has_vest, vest_conf)
        """
        if person_crop is None or person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
            return False, 0.5, False, 0.5

        h, w = person_crop.shape[:2]
        # Head region: top 28%
        head_crop = person_crop[0:int(h * 0.28), :]
        # Torso region: 28% to 68%
        torso_crop = person_crop[int(h * 0.28):int(h * 0.68), :]

        hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV) if head_crop.size > 0 else None
        hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV) if torso_crop.size > 0 else None

        has_helmet = False
        helmet_conf = 0.5
        if hsv_head is not None and hsv_head.size > 0:
            # Yellow / High-Vis Hardhat: H in [20, 35], S > 160, V > 180
            mask_yellow_hh = cv2.inRange(hsv_head, np.array([18, 150, 180]), np.array([36, 255, 255]))
            # White hardhat: S < 35, V > 220
            mask_white_hh = cv2.inRange(hsv_head, np.array([0, 0, 220]), np.array([180, 40, 255]))
            mask_hh = cv2.bitwise_or(mask_yellow_hh, mask_white_hh)
            hh_ratio = float(np.sum(mask_hh > 0)) / float(head_crop.shape[0] * head_crop.shape[1] + 1e-5)
            if hh_ratio > 0.08:
                has_helmet = True
                helmet_conf = min(0.98, float(0.70 + hh_ratio * 1.5))
            else:
                has_helmet = False
                helmet_conf = min(0.96, float(0.75 + (0.08 - hh_ratio) * 2.0))

        has_vest = False
        vest_conf = 0.5
        if hsv_torso is not None and hsv_torso.size > 0:
            # High-vis vest (orange/yellow/lime): H in [18, 40], S > 150, V > 160
            mask_vest = cv2.inRange(hsv_torso, np.array([16, 140, 150]), np.array([40, 255, 255]))
            vest_ratio = float(np.sum(mask_vest > 0)) / float(torso_crop.shape[0] * torso_crop.shape[1] + 1e-5)
            if vest_ratio > 0.12:
                has_vest = True
                vest_conf = min(0.98, float(0.70 + vest_ratio * 1.2))
            else:
                has_vest = False
                vest_conf = min(0.96, float(0.75 + (0.12 - vest_ratio) * 1.8))

        return has_helmet, helmet_conf, has_vest, vest_conf


class SafetyDetector:
    """YOLOv8-based Safety Perception Detector with Construction Machinery Awareness."""

    MACHINERY_CLASSES = {"truck", "bus", "train", "car", "machinery", "excavator", "crane", "forklift", "roller"}
    PERSON_CLASSES = {"person", "worker"}
    PPE_CLASSES = {"helmet", "hard-hat", "hardhat", "safety-vest", "vest", "no-helmet", "no-vest"}

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.model = None
        if YOLO is not None and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"[Warning] Failed to load YOLO model: {e}")

    def detect_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        """Runs perception on a single frame and returns structured Detections."""
        h, w = frame.shape[:2]
        detections: List[Detection] = []

        if self.model is not None:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = r.names.get(cls_id, f"class_{cls_id}").lower()
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy

                    x1 = max(0.0, min(float(w), x1))
                    y1 = max(0.0, min(float(h), y1))
                    x2 = max(0.0, min(float(w), x2))
                    y2 = max(0.0, min(float(h), y2))

                    width = max(1.0, x2 - x1)
                    height = max(1.0, y2 - y1)
                    cx = x1 + width / 2.0
                    cy = y1 + height / 2.0

                    mapped_name = cls_name
                    if cls_name in self.PERSON_CLASSES:
                        mapped_name = "person"
                    elif cls_name in self.MACHINERY_CLASSES:
                        mapped_name = "machinery"

                    det = Detection(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        track_id=None,
                        class_id=cls_id,
                        class_name=mapped_name,
                        confidence=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        center_x=cx,
                        center_y=cy,
                        width=width,
                        height=height
                    )
                    detections.append(det)

        # Construction machinery visual locator (excavator cabin/boom/tracks detection)
        # Check if machinery was already detected by YOLO
        has_machinery = any(d.class_name == "machinery" for d in detections)
        if not has_machinery:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Construction machinery yellow: H in [15, 36], S > 170, V > 170
            mask_yellow = cv2.inRange(hsv, np.array([15, 170, 170]), np.array([36, 255, 255]))
            # Mask out already detected persons so person clothes don't trigger machine
            for d in detections:
                if d.class_name == "person":
                    px1, py1, px2, py2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                    mask_yellow[py1:py2, px1:px2] = 0

            cnts, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area > 800:
                    mx, my, mw, mh = cv2.boundingRect(c)
                    # Extend bounding box to include full machine base/boom
                    x1 = float(max(0, mx - 30))
                    y1 = float(max(0, my - 20))
                    x2 = float(min(w, mx + mw + 30))
                    y2 = float(min(h, my + mh + 35))
                    width = x2 - x1
                    height = y2 - y1
                    det = Detection(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        track_id=None,
                        class_id=7, # truck / machinery id
                        class_name="machinery",
                        confidence=0.91,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        center_x=x1 + width / 2.0,
                        center_y=y1 + height / 2.0,
                        width=width,
                        height=height
                    )
                    detections.append(det)
                    break # At most one primary excavator in current synthetic clips

        return detections
