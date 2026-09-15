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


class FaceCranialDetector:
    """Dedicated Face and Cranial Vault detector for construction personnel.
    Combines OpenCV Haar cascades with adaptive morphological skin-tone & cranial profiling.
    """
    _frontal_cascade = None
    _profile_cascade = None

    @classmethod
    def _get_cascades(cls):
        if cls._frontal_cascade is None:
            frontal_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            profile_path = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
            if os.path.exists(frontal_path):
                cls._frontal_cascade = cv2.CascadeClassifier(frontal_path)
            if os.path.exists(profile_path):
                cls._profile_cascade = cv2.CascadeClassifier(profile_path)
        return cls._frontal_cascade, cls._profile_cascade

    @classmethod
    def detect_face(
        cls,
        person_crop: np.ndarray,
        person_box: Optional[List[float]] = None
    ) -> Tuple[bool, float, Optional[List[float]], str]:
        """Detects the human face in a person crop with multi-stage fallback.

        Args:
            person_crop: BGR image crop of the person.
            person_box: Optional [px1, py1, px2, py2] in absolute pixel coordinates.

        Returns:
            Tuple of (has_face: bool, confidence: float, face_box: [fx1, fy1, fx2, fy2] or None, method: str)
            Coordinates are absolute if person_box is provided, else relative to person_crop.
        """
        if person_crop is None or person_crop.size == 0 or person_crop.shape[0] < 12 or person_crop.shape[1] < 12:
            return False, 0.0, None, "none"

        ph, pw = person_crop.shape[:2]
        px1, py1 = (person_box[0], person_box[1]) if person_box else (0.0, 0.0)
        aspect_ratio = float(pw) / float(max(1, ph))

        # Upper head/face region to search: upper 60% for portrait/chest-up, upper 38% for full body
        search_h = int(ph * 0.60) if aspect_ratio >= 0.45 else int(ph * 0.38)
        search_h = max(10, min(ph, search_h))
        head_crop = person_crop[0:search_h, :]

        frontal_casc, profile_casc = cls._get_cascades()

        # Stage 1: Frontal Haar Cascade
        if frontal_casc is not None and not frontal_casc.empty():
            gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray_head)
            min_dim = max(14, int(min(pw, search_h) * 0.18))
            faces = frontal_casc.detectMultiScale(
                gray_eq,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(min_dim, min_dim)
            )
            if len(faces) > 0:
                # Pick most central and substantial face
                best_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = best_face
                abs_box = [float(px1 + fx), float(py1 + fy), float(px1 + fx + fw), float(py1 + fy + fh)]
                return True, 0.94, abs_box, "haar_frontal"

        # Stage 2: Profile Haar Cascade (for turned heads)
        if profile_casc is not None and not profile_casc.empty():
            gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
            faces = profile_casc.detectMultiScale(
                gray_head,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(14, 14)
            )
            if len(faces) > 0:
                best_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = best_face
                abs_box = [float(px1 + fx), float(py1 + fy), float(px1 + fx + fw), float(py1 + fy + fh)]
                return True, 0.88, abs_box, "haar_profile"

        # Stage 3: Adaptive Skin Tone & Cranial Morphology Fallback
        # Converts to YCrCb (OSHA standard skin tone chromaticity: Cr in [133, 173], Cb in [77, 127])
        ycrcb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2YCrCb)
        mask_skin = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        # Morphological opening to remove salt noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_score = 0.0
        tot_head_area = float(pw * search_h + 1e-5)
        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)

        for c in cnts:
            c_area = cv2.contourArea(c)
            fx, fy, fw, fh = cv2.boundingRect(c)
            box_area = float(fw * fh)
            area_ratio = box_area / tot_head_area
            # Face must be a distinct cranial element: between 4% and 65% of head crop
            # (rejects massive ambient washes that span whole image, and tiny specks)
            if area_ratio < 0.04 or area_ratio > 0.65:
                continue
            # Plausible face aspect ratio
            ratio = fw / float(max(1, fh))
            if not (0.45 <= ratio <= 1.85):
                continue
            # Must not bleed to all borders simultaneously
            if fw >= 0.92 * pw and fh >= 0.92 * search_h:
                continue
            # Skin fill density inside bounding box
            c_crop_mask = mask_skin[fy:fy+fh, fx:fx+fw]
            density = np.count_nonzero(c_crop_mask) / max(1.0, box_area)
            if density < 0.20:
                continue

            score = c_area * density
            if score > best_score:
                best_score = score
                best_candidate = [float(px1 + fx), float(py1 + fy), float(px1 + fx + fw), float(py1 + fy + fh)]

        if best_candidate is not None:
            return True, 0.80, best_candidate, "skin_cranial_morph"

        # If neither Haar cascades nor validated cranial morphology finds a face, return False
        return False, 0.0, None, "none"


class PPEInspector:
    """Accurate PPE inspector for full-body, half-body, and chest-up portrait framing.
    Anchors hardhat verification strictly to anatomical cranial relationship with the human face.
    """
    
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

            # Saturated ANSI/OSHA Hardhat Pigments (Yellow, Safety Orange, Red, Blue, Green)
            # High-saturation ensures white walls, blonde/grey hair, and daylight are rejected
            # 1. High-Vis Yellow / Lime: H in [17, 42], S > 95, V > 115
            mask_yellow = cv2.inRange(hsv_cranial, np.array([17, 95, 115]), np.array([42, 255, 255]))
            # 2. Safety Orange: H in [8, 17], S > 120, V > 115
            mask_orange = cv2.inRange(hsv_cranial, np.array([8, 120, 115]), np.array([17, 255, 255]))
            # 3. Safety Red: H in [0, 7], S > 130, V > 110 (strictly red, not pink/magenta)
            mask_red = cv2.inRange(hsv_cranial, np.array([0, 130, 110]), np.array([7, 255, 255]))
            # 4. Safety Blue: H in [95, 135], S > 85, V > 85
            mask_blue = cv2.inRange(hsv_cranial, np.array([95, 85, 85]), np.array([135, 255, 255]))
            # 5. Safety Green: H in [40, 85], S > 80, V > 80
            mask_green = cv2.inRange(hsv_cranial, np.array([40, 80, 80]), np.array([85, 255, 255]))

            mask_colored = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange, cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_green))))
            tot_cranial = float(cranial_crop.shape[0] * cranial_crop.shape[1] + 1e-5)
            colored_ratio = float(np.sum(mask_colored > 0)) / tot_cranial

            # Saturated colored helmets contrast strongly with skin, hair, and white background walls
            if colored_ratio > 0.08:
                has_helmet = True
                helmet_conf = min(0.99, float(0.78 + colored_ratio * 1.5))
            else:
                has_helmet = False
                helmet_conf = min(0.95, float(0.75 + (0.08 - colored_ratio) * 2.0))

        has_vest = False
        vest_conf = 0.5
        if hsv_torso is not None and hsv_torso.size > 0:
            # High-Vis Fluorescent Neon Safety Vest Pigments:
            # 1. High-vis Neon Yellow / Lime: H in [18, 42], S > 110, V > 135
            mask_lime = cv2.inRange(hsv_torso, np.array([18, 110, 135]), np.array([42, 255, 255]))
            # 2. High-vis Neon Orange: H in [9, 16], S > 150, V > 140 (strictly neon fluorescent orange; excludes clay bricks and dark plaid)
            mask_orange = cv2.inRange(hsv_torso, np.array([9, 150, 140]), np.array([16, 255, 255]))
            # 3. Retro-reflective silver/white stripes on torso (paired with neon fabric)
            mask_silver = cv2.inRange(hsv_torso, np.array([0, 0, 215]), np.array([180, 25, 255]))
            
            mask_vest_colored = cv2.bitwise_or(mask_lime, mask_orange)
            vest_color_ratio = float(np.sum(mask_vest_colored > 0)) / float(torso_crop.shape[0] * torso_crop.shape[1] + 1e-5)
            silver_ratio = float(np.sum(mask_silver > 0)) / float(torso_crop.shape[0] * torso_crop.shape[1] + 1e-5)

            # In safety vests, high-vis neon fluorescent fabric must be prominent, or paired with silver bands
            threshold = 0.12 if aspect_ratio >= 0.45 else 0.14
            if (vest_color_ratio > threshold) or (vest_color_ratio > 0.06 and silver_ratio > 0.02):
                has_vest = True
                vest_conf = min(0.99, float(0.75 + vest_color_ratio * 1.8 + silver_ratio * 1.2))
            else:
                has_vest = False
                vest_conf = min(0.96, float(0.75 + (threshold - vest_color_ratio) * 2.0))

        return has_helmet, helmet_conf, has_vest, vest_conf

    @staticmethod
    def verify_hardhat_on_face(
        candidate_box: List[float],
        face_box: List[float],
        person_box: Optional[List[float]] = None
    ) -> Tuple[bool, float, Optional[List[float]], Dict[str, Any]]:
        """Strictly verifies if a detected hardhat sits directly on top of the worker's face.

        Defines the 'Good Detect' rule:
          1. Vertical alignment: hardhat center is strictly above face center (hardhat_cy < face_cy).
             The bottom of the hardhat (hy2) rests in the transition zone near the forehead/eyebrows.
          2. Horizontal alignment: hardhat is horizontally centered with the face.
          3. Proportional scale: hardhat width is anatomically compatible with face width.
          4. Returns an exact, jitter-free square bounding box for the hardhat anchored to the face.

        Args:
            candidate_box: [hx1, hy1, hx2, hy2] raw candidate hardhat box in pixel coordinates.
            face_box: [fx1, fy1, fx2, fy2] face box in pixel coordinates.
            person_box: Optional [px1, py1, px2, py2] worker body box.

        Returns:
            Tuple of:
              - is_good_detect (bool): True if hardhat sits on top of face.
              - confidence (float): Hardhat spatial verification confidence.
              - square_box (List[float] or None): Clean, stabilized square bounding box [hx1, hy1, hx2, hy2].
              - metrics (Dict[str, Any]): Detailed spatial alignment diagnostics.
        """
        if not candidate_box or not face_box:
            return False, 0.0, None, {"reason": "missing_inputs"}

        hx1, hy1, hx2, hy2 = candidate_box
        fx1, fy1, fx2, fy2 = face_box

        hw = max(1.0, float(hx2 - hx1))
        hh = max(1.0, float(hy2 - hy1))
        hcx = (hx1 + hx2) / 2.0
        hcy = (hy1 + hy2) / 2.0

        fw = max(1.0, float(fx2 - fx1))
        fh = max(1.0, float(fy2 - fy1))
        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0

        # Metric 1: Vertical positioning: Hardhat must be above the face
        vertical_delta = fcy - hcy  # Must be positive (hardhat is higher up in image coordinates)
        is_above_face = vertical_delta > 0.0

        # Hardhat bottom (hy2) should rest at forehead level (near fy1)
        # Permissible range: from slightly above top of face to upper 40% of face (forehead/brow)
        bottom_to_face_top = hy2 - fy1
        valid_vertical_seating = (-0.45 * fh <= bottom_to_face_top <= 0.45 * fh) or (hcy < fy1)

        # Metric 2: Horizontal alignment: Hardhat must align horizontally with the face
        horiz_offset = abs(hcx - fcx)
        max_allowed_h_offset = 0.55 * max(hw, fw)
        is_horiz_aligned = horiz_offset <= max_allowed_h_offset

        # Metric 3: Proportional scale: Hardhat width must be anatomically compatible with face width
        scale_ratio = hw / fw
        is_scale_valid = (0.60 <= scale_ratio <= 2.50)

        # Good Detect evaluation
        is_good_detect = bool(is_above_face and valid_vertical_seating and is_horiz_aligned and is_scale_valid)

        metrics = {
            "is_above_face": is_above_face,
            "vertical_delta_px": round(vertical_delta, 1),
            "bottom_to_face_top_px": round(bottom_to_face_top, 1),
            "valid_vertical_seating": valid_vertical_seating,
            "horiz_offset_px": round(horiz_offset, 1),
            "max_allowed_h_offset": round(max_allowed_h_offset, 1),
            "is_horiz_aligned": is_horiz_aligned,
            "scale_ratio": round(scale_ratio, 2),
            "is_scale_valid": is_scale_valid,
            "is_good_detect": is_good_detect
        }

        if not is_good_detect:
            return False, 0.20, None, metrics

        # Compute Rock-Solid, Jitter-Free Square Bounding Box anchored directly on top of the face:
        # Standard hardhat is 1.10x - 1.25x the width of the face
        side = int(round(max(hw, hh, fw * 1.15)))
        # Center horizontally: weighted blend between face center and raw hardhat center
        hcx_stable = int(round(0.65 * fcx + 0.35 * hcx))
        # Anchor bottom to forehead/eyebrow level
        hy2_sq = int(round(fy1 + 0.12 * fh))
        hy1_sq = hy2_sq - side
        hx1_sq = hcx_stable - side // 2
        hx2_sq = hx1_sq + side

        # Ensure inside positive image coordinates
        if hx1_sq < 0:
            hx2_sq += (-hx1_sq)
            hx1_sq = 0
        if hy1_sq < 0:
            hy1_sq = 0
            hy2_sq = side

        square_box = [float(hx1_sq), float(hy1_sq), float(hx2_sq), float(hy2_sq)]
        conf = float(min(0.99, 0.82 + 0.15 * (1.0 - min(1.0, horiz_offset / max_allowed_h_offset))))

        return True, conf, square_box, metrics

    @staticmethod
    def locate_cranial_hardhat_square(
        person_crop: np.ndarray,
        person_box: List[float],
        face_box: Optional[List[float]] = None
    ) -> Tuple[bool, float, Optional[List[float]], bool]:
        """Locates the dedicated square bounding box for a hardhat on the cranial dome,
        anchored directly to the detected face for anatomical consistency.

        Args:
            person_crop: BGR image crop of the person.
            person_box: [px1, py1, px2, py2] absolute pixel coordinates.
            face_box: Optional [fx1, fy1, fx2, fy2] face box in absolute pixel coordinates.

        Returns:
            Tuple: (has_hardhat: bool, confidence: float, square_box: [hx1, hy1, hx2, hy2] or None, is_good_detect: bool)
        """
        if person_crop is None or person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
            return False, 0.5, None, False

        px1, py1, px2, py2 = person_box
        ph, pw = person_crop.shape[:2]
        aspect_ratio = float(pw) / float(max(1, ph))

        # If face_box is not provided, detect face first
        if face_box is None:
            has_face, face_conf, face_box, method = FaceCranialDetector.detect_face(person_crop, person_box)
        else:
            has_face = True

        # Cranial dome region situated directly on top of the face
        if has_face and face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            fw = max(1.0, fx2 - fx1)
            fh = max(1.0, fy2 - fy1)
            fcx = (fx1 + fx2) / 2.0
            # Local coordinates of face relative to person_crop
            local_fx1 = max(0, int(fx1 - px1))
            local_fy1 = max(0, int(fy1 - py1))
            local_fx2 = min(pw, int(fx2 - px1))
            local_fy2 = min(ph, int(fy2 - py1))

            # Cranial search dome is directly above the face
            cranial_top = max(0, int(local_fy1 - 1.2 * fh))
            cranial_bottom = min(ph, int(local_fy1 + 0.18 * fh))
            cranial_left = max(0, int(local_fx1 - 0.25 * fw))
            cranial_right = min(pw, int(local_fx2 + 0.25 * fw))

            if cranial_bottom > cranial_top + 4 and cranial_right > cranial_left + 4:
                cranial_crop = person_crop[cranial_top:cranial_bottom, cranial_left:cranial_right]
            else:
                cranial_crop = person_crop[0:max(6, int(ph * 0.30)), :]
        else:
            cranial_h_ratio = 0.42 if aspect_ratio >= 0.45 else 0.24
            cranial_h = max(6, int(ph * cranial_h_ratio))
            cranial_crop = person_crop[0:cranial_h, :]
            fw = pw * 0.45
            fh = cranial_h * 0.8
            fcx = px1 + pw / 2.0
            fy1 = py1 + cranial_h * 0.65

        hsv_cranial = cv2.cvtColor(cranial_crop, cv2.COLOR_BGR2HSV)

        # Saturated ANSI/OSHA Hardhat Pigments (Yellow, Safety Orange, Red, Blue, Green)
        # Saturated pigments ensure white walls, blonde/grey hair, and daylight are rejected
        mask_yellow = cv2.inRange(hsv_cranial, np.array([17, 95, 115]), np.array([42, 255, 255]))
        mask_orange = cv2.inRange(hsv_cranial, np.array([8, 120, 115]), np.array([17, 255, 255]))
        mask_red = cv2.inRange(hsv_cranial, np.array([0, 130, 110]), np.array([7, 255, 255]))
        mask_blue = cv2.inRange(hsv_cranial, np.array([95, 85, 85]), np.array([135, 255, 255]))
        mask_green = cv2.inRange(hsv_cranial, np.array([40, 80, 80]), np.array([85, 255, 255]))

        mask_hh = cv2.bitwise_or(
            cv2.bitwise_or(mask_yellow, mask_orange),
            cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_green))
        )

        tot_area = float(cranial_crop.shape[0] * cranial_crop.shape[1] + 1e-5)
        hh_pixels = float(np.sum(mask_hh > 0))
        ratio = hh_pixels / tot_area

        if ratio < 0.08:
            return False, 0.20, None, False

        # Build consistent square box anchored on top of face
        side = int(round(max(pw * 0.70, fw * 1.15, cranial_crop.shape[0] * 1.10)))
        hcx = int(round(fcx))
        hy2_sq = int(round(fy1 + 0.10 * fh))
        hy1_sq = hy2_sq - side
        hx1_sq = hcx - side // 2
        hx2_sq = hx1_sq + side

        if hx1_sq < 0:
            hx1_sq = 0
            hx2_sq = side
        if hy1_sq < 0:
            hy1_sq = 0
            hy2_sq = side

        conf = float(min(0.98, 0.75 + ratio * 1.8))
        return True, conf, [float(hx1_sq), float(hy1_sq), float(hx2_sq), float(hy2_sq)], True



class SafetyDetector:
    """YOLOv8-based Safety Perception Detector with Dedicated Hardhat & PPE Awareness."""

    HARDHAT_CLASSES = {"hardhat", "hard-hat", "helmet", "safety-helmet", "safety_helmet"}
    NO_HARDHAT_CLASSES = {"no-hardhat", "no_hardhat", "no-helmet", "no_helmet", "head", "without-helmet"}
    VEST_CLASSES = {"safety vest", "safety-vest", "safety_vest", "vest", "hi-vis", "reflective-vest"}
    NO_VEST_CLASSES = {"no-vest", "no_vest", "without-vest", "no-safety vest", "no_safety_vest"}
    PERSON_CLASSES = {"person", "worker", "human", "pedestrian"}
    MACHINERY_CLASSES = {"truck", "bus", "train", "car", "machinery", "excavator", "crane", "forklift", "roller", "vehicle"}
    PPE_CLASSES = HARDHAT_CLASSES | NO_HARDHAT_CLASSES | VEST_CLASSES | NO_VEST_CLASSES

    DEFAULT_PPE_WEIGHTS = "models/yolov8_ppe_best.pt"

    def __init__(self, model_path: str = "models/yolov8_ppe_best.pt", conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        # Check if specialized PPE model is available, otherwise fallback
        if not os.path.exists(model_path):
            if os.path.exists(self.DEFAULT_PPE_WEIGHTS):
                model_path = self.DEFAULT_PPE_WEIGHTS
            elif os.path.exists("yolov8n.pt"):
                model_path = "yolov8n.pt"

        self.model_path = model_path
        self.model = None
        if YOLO is not None and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"[Warning] Failed to load YOLO model from {model_path}: {e}")

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
                    raw_name = r.names.get(cls_id, f"class_{cls_id}").lower().strip()
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

                    # Standardize class names
                    clean_name = raw_name.replace("_", "-").replace(" ", "-")
                    if clean_name in self.HARDHAT_CLASSES or raw_name in self.HARDHAT_CLASSES:
                        mapped_name = "hardhat"
                    elif clean_name in self.NO_HARDHAT_CLASSES or raw_name in self.NO_HARDHAT_CLASSES:
                        mapped_name = "no_hardhat"
                    elif clean_name in self.VEST_CLASSES or raw_name in self.VEST_CLASSES:
                        mapped_name = "vest"
                    elif clean_name in self.NO_VEST_CLASSES or raw_name in self.NO_VEST_CLASSES:
                        mapped_name = "no_vest"
                    elif clean_name in self.PERSON_CLASSES or raw_name in self.PERSON_CLASSES:
                        mapped_name = "person"
                    elif clean_name in self.MACHINERY_CLASSES or raw_name in self.MACHINERY_CLASSES:
                        mapped_name = "machinery"
                    else:
                        mapped_name = raw_name

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
