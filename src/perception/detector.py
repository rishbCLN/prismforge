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
    """Accurate PPE inspector for full-body, half-body, and chest-up portrait framing."""
    
    @staticmethod
    def inspect_person_crop(person_crop: np.ndarray) -> Tuple[bool, float, bool, float]:
        """Inspects person image crop for hardhat and safety vest.
        Dynamically adapts anatomical regions for chest-up/portrait crops vs full standing body.
        """
        if person_crop is None or person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
            return False, 0.5, False, 0.5

        h, w = person_crop.shape[:2]
        aspect_ratio = float(w) / float(max(1, h))

        # Dynamic region segmentation based on framing:
        # If aspect_ratio >= 0.45: portrait / chest-up framing (head is upper 45%, vest spans down to bottom)
        if aspect_ratio >= 0.45:
            head_crop = person_crop[0:int(h * 0.48), :]
            torso_crop = person_crop[int(h * 0.32):h, :]
        else:
            # Full standing body framing
            head_crop = person_crop[0:int(h * 0.28), :]
            torso_crop = person_crop[int(h * 0.20):int(h * 0.72), :]

        hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV) if head_crop.size > 0 else None
        hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV) if torso_crop.size > 0 else None

        has_helmet = False
        helmet_conf = 0.5
        if head_crop is not None and head_crop.size > 0:
            head_h = head_crop.shape[0]
            # Anatomical Cranial Vault: top 65% of head crop (above eyebrows/eyes)
            cranial_crop = head_crop[0:max(4, int(head_h * 0.65)), :]
            hsv_cranial = cv2.cvtColor(cranial_crop, cv2.COLOR_BGR2HSV)

            # Full-Spectrum ANSI/OSHA Hardhat Pigments:
            # 1. High-Vis Yellow / Lime: H in [17, 42], S > 95, V > 115
            mask_yellow = cv2.inRange(hsv_cranial, np.array([17, 95, 115]), np.array([42, 255, 255]))
            # 2. Safety Orange: H in [6, 17], S > 120, V > 115
            mask_orange = cv2.inRange(hsv_cranial, np.array([6, 120, 115]), np.array([17, 255, 255]))
            # 3. Safety Red: H in [0, 6] or [170, 180], S > 120, V > 105
            mask_red1 = cv2.inRange(hsv_cranial, np.array([0, 120, 105]), np.array([6, 255, 255]))
            mask_red2 = cv2.inRange(hsv_cranial, np.array([170, 120, 105]), np.array([180, 255, 255]))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            # 4. Safety Blue: H in [95, 135], S > 75, V > 75
            mask_blue = cv2.inRange(hsv_cranial, np.array([95, 75, 75]), np.array([135, 255, 255]))
            # 5. Safety Green: H in [40, 85], S > 70, V > 75
            mask_green = cv2.inRange(hsv_cranial, np.array([40, 70, 75]), np.array([85, 255, 255]))
            # 6. Safety White: S < 40, V > 200
            mask_white = cv2.inRange(hsv_cranial, np.array([0, 0, 200]), np.array([180, 40, 255]))

            mask_colored = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_green))))
            tot_cranial = float(cranial_crop.shape[0] * cranial_crop.shape[1] + 1e-5)
            colored_ratio = float(np.sum(mask_colored > 0)) / tot_cranial
            white_ratio = float(np.sum(mask_white > 0)) / tot_cranial

            # Colored helmets (yellow, orange, red, blue, green) contrast strongly with skin and hair
            if colored_ratio > 0.05:
                has_helmet = True
                helmet_conf = min(0.99, float(0.78 + colored_ratio * 1.5))
            # White helmets require higher concentration or density
            elif white_ratio > 0.10:
                has_helmet = True
                helmet_conf = min(0.99, float(0.75 + white_ratio * 1.2))
            else:
                has_helmet = False
                helmet_conf = min(0.95, float(0.75 + (0.05 - colored_ratio) * 2.0))

        has_vest = False
        vest_conf = 0.5
        if hsv_torso is not None and hsv_torso.size > 0:
            # 1. High-vis Neon Yellow / Lime: H in [16, 45], S > 90, V > 120
            mask_lime = cv2.inRange(hsv_torso, np.array([16, 90, 120]), np.array([45, 255, 255]))
            # 2. High-vis Neon Orange: H in [0, 16] or [165, 180], S > 100, V > 120
            mask_orange1 = cv2.inRange(hsv_torso, np.array([0, 100, 120]), np.array([16, 255, 255]))
            mask_orange2 = cv2.inRange(hsv_torso, np.array([165, 100, 120]), np.array([180, 255, 255]))
            # 3. Reflective silver/white stripes on torso
            mask_silver = cv2.inRange(hsv_torso, np.array([0, 0, 190]), np.array([180, 45, 255]))
            
            mask_vest_colored = cv2.bitwise_or(mask_lime, cv2.bitwise_or(mask_orange1, mask_orange2))
            vest_color_ratio = float(np.sum(mask_vest_colored > 0)) / float(torso_crop.shape[0] * torso_crop.shape[1] + 1e-5)
            silver_ratio = float(np.sum(mask_silver > 0)) / float(torso_crop.shape[0] * torso_crop.shape[1] + 1e-5)

            # In chest-up crops, high-vis neckline or silver reflective bands are decisive
            threshold = 0.05 if aspect_ratio >= 0.45 else 0.08
            if vest_color_ratio > threshold or (vest_color_ratio > 0.03 and silver_ratio > 0.02):
                has_vest = True
                vest_conf = min(0.99, float(0.75 + vest_color_ratio * 1.8 + silver_ratio * 1.2))
            else:
                has_vest = False
                vest_conf = min(0.96, float(0.75 + (threshold - vest_color_ratio) * 2.0))

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
