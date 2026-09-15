"""End-to-End PPE Inference Engine.
Connects YOLOv8 perception with custom PPEReasonerNet neural network to analyze
uploaded test images and output safety compliance conclusions and visual annotations.
"""
import os
import cv2
import time
import base64
import threading
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.risk_model.ppe_reasoner_net import PPEReasonerNet, extract_ppe_neural_features, _calc_iou
from src.perception.detector import SafetyDetector, PPEInspector, Detection, FaceCranialDetector
from src.prism.client import PrismClient


class PPEInferenceEngine:
    """Inference engine combining YOLOv8 detection with PPEReasonerNet decision intelligence."""

    VIOLATION_NAMES = {
        0: "COMPLIANT",
        1: "MISSING_VEST",
        2: "MISSING_HARDHAT",
        3: "CRITICAL_NO_PPE"
    }

    VIOLATION_COLORS = {
        0: (46, 204, 113),   # Emerald Green (BGR: 113, 204, 46)
        1: (34, 126, 230),   # Orange (BGR: 34, 126, 230)
        2: (15, 196, 241),   # Yellow (BGR: 15, 196, 241)
        3: (60, 76, 231)     # Red (BGR: 60, 76, 231)
    }

    FILES_USED = [
        "models/ppe_reasoner.pt (Trained PyTorch Neural Weights from datasets/ppe_master_folder)",
        "models/yolov8_ppe_best.pt (Custom YOLOv8 PPE detector with Hardhat/Person classes)",
        "src/risk_model/ppe_reasoner_net.py (PPEReasonerNet Architecture & 16-dim Feature Extractor)",
        "src/perception/detector.py (YOLOv8 + PPE Visual Perception Inspector)",
        "src/features/ppe_inference.py (Inference Engine & Decision Pipeline)",
        "web/server.py (FastAPI Endpoint /api/ppe/analyze)",
        "web/index.html & web/static/js/app.js (PPE Analyser Interactive UI & Canvas)"
    ]

    def __init__(self, model_path: str = "models/ppe_reasoner.pt", yolo_path: str = "models/yolov8_ppe_best.pt",
                 use_tta: bool = True, temperature: float = 1.5, tta_noise_std: float = 0.015,
                 tta_passes: int = 5):
        self.model_path = model_path
        self.yolo_path = yolo_path
        self.use_tta = use_tta
        self.temperature = temperature  # Temperature scaling for calibrated logits
        self.tta_noise_std = tta_noise_std
        self.tta_passes = tta_passes

        # Load PPEReasonerNet (V3 with backward compat for V2)
        self.device = torch.device("cpu")
        self.reasoner = None
        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                in_dim = ckpt.get("input_dim", 16)
                hid_dim = ckpt.get("hidden_dim", 64)
                self.reasoner = PPEReasonerNet(input_dim=in_dim, hidden_dim=hid_dim)
                self.reasoner.load_state_dict(ckpt["state_dict"], strict=False)
                self.reasoner.eval()
            except Exception as e:
                print(f"[Warning] Failed to load PPEReasonerNet: {e}")

        # Try loading EMA model if available (even more stable weights)
        ema_path = model_path.replace(".pt", "_ema.pt")
        self.reasoner_ema = None
        if os.path.exists(ema_path):
            try:
                ema_ckpt = torch.load(ema_path, map_location=self.device, weights_only=False)
                in_dim = ema_ckpt.get("input_dim", 16)
                hid_dim = ema_ckpt.get("hidden_dim", 64)
                self.reasoner_ema = PPEReasonerNet(input_dim=in_dim, hidden_dim=hid_dim)
                self.reasoner_ema.load_state_dict(ema_ckpt["state_dict"], strict=False)
                self.reasoner_ema.eval()
            except Exception:
                pass

        # Load YOLOv8 (prefer specialized PPE model if present)
        chosen_yolo = yolo_path if os.path.exists(yolo_path) else ("models/yolov8_ppe_best.pt" if os.path.exists("models/yolov8_ppe_best.pt") else "yolov8n.pt")
        self.detector = SafetyDetector(model_path=chosen_yolo, conf_threshold=0.25)
        self.prism_client = PrismClient()

    @staticmethod
    def _box_metrics(b1: List[float], b2: List[float]) -> Tuple[float, float]:
        """Calculates IoU and Containment ratio between two boxes [x1, y1, x2, y2]."""
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        iarea = iw * ih
        a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        iou = iarea / max(1.0, a1 + a2 - iarea)
        containment = iarea / min(a1, a2)
        return iou, containment

    @classmethod
    def filter_valid_workers(
        cls,
        person_dets: List[Detection],
        img: np.ndarray,
        hardhat_dets: List[Detection],
        vest_dets: List[Detection]
    ) -> List[Detection]:
        """Suppresses duplicate and architectural false-positive worker proposals.

        Rules:
          1. Extreme Aspect Ratio Gating: Rejects narrow architectural columns (pw/ph < 0.15)
             or ultra-wide horizontal bands (pw/ph > 2.5).
          2. Containment & Overlap Suppression (NMS): If a candidate is heavily contained
             (containment > 0.45 or IoU > 0.40) inside an accepted higher-confidence worker,
             it is suppressed unless both have separate, distinct verified human faces.
          3. Verification Requirement for Low/Moderate Confidence:
             If candidate confidence < 0.65, the candidate must have at least one verified
             human anchor: a verified face, a verified hardhat in the cranial zone,
             or a verified safety vest on the torso. Unverified background objects (pillars,
             walls, equipment) are rejected.
        """
        if not person_dets:
            return []

        h, w = img.shape[:2]
        sorted_persons = sorted(person_dets, key=lambda d: d.confidence, reverse=True)
        accepted: List[Detection] = []
        accepted_face_boxes: List[Optional[List[float]]] = []

        for p in sorted_persons:
            px1 = max(0, int(round(p.x1)))
            py1 = max(0, int(round(p.y1)))
            px2 = min(w, int(round(p.x2)))
            py2 = min(h, int(round(p.y2)))
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)
            aspect = float(pw) / float(ph)

            # 1. Reject impossible aspect ratios for standing/sitting workers
            if aspect < 0.15 or aspect > 2.5:
                continue

            person_crop = img[py1:py2, px1:px2]
            if person_crop.size == 0:
                continue

            # Check if this candidate has a verified face
            has_face, face_conf, face_box, _ = FaceCranialDetector.detect_face(
                person_crop, [float(px1), float(py1), float(px2), float(py2)]
            )

            # Check if this candidate has an overlapping hardhat near the head
            head_ymax = py1 + ph * 0.45
            has_cranial_hh = any(
                hd.y1 <= head_ymax and (px1 - pw * 0.20 <= hd.center_x <= px2 + pw * 0.20)
                for hd in hardhat_dets
            ) or PPEInspector.locate_cranial_hardhat_square(person_crop, [float(px1), float(py1), float(px2), float(py2)], face_box)[0]

            # Check if this candidate has an overlapping vest near torso
            torso_ymin = py1 + ph * 0.20
            has_torso_vest = any(
                vd.y1 >= torso_ymin and (px1 - pw * 0.20 <= vd.center_x <= px2 + pw * 0.20)
                for vd in vest_dets
            ) or PPEInspector.inspect_person_crop(person_crop)[2]

            # 2. Minimum Verification for low/moderate confidence
            if p.confidence < 0.65:
                if not (has_face or has_cranial_hh or has_torso_vest):
                    # Inanimate architectural element (e.g. concrete pillar, door frame)
                    continue

            # 3. Containment & NMS Suppression against already accepted workers
            is_duplicate = False
            p_box = [float(px1), float(py1), float(px2), float(py2)]

            for acc, acc_fb in zip(accepted, accepted_face_boxes):
                acc_box = [float(acc.x1), float(acc.y1), float(acc.x2), float(acc.y2)]
                iou, containment = cls._box_metrics(p_box, acc_box)

                if containment > 0.45 or iou > 0.40:
                    # Overlap is significant. Check if they have distinct verified human faces
                    if has_face and face_box is not None and acc_fb is not None:
                        p_fcx = (face_box[0] + face_box[2]) / 2.0
                        acc_fcx = (acc_fb[0] + acc_fb[2]) / 2.0
                        face_sep = abs(p_fcx - acc_fcx)
                        min_sep = 0.25 * max(pw, acc.width)
                        if face_sep >= min_sep:
                            # Two distinct people side-by-side
                            continue

                    # Otherwise, candidate is a duplicate / shadow / pillar overlapping the real worker
                    is_duplicate = True
                    break

            if not is_duplicate:
                accepted.append(p)
                accepted_face_boxes.append(face_box if has_face else None)

        return accepted

    def analyze_image_bytes(self, image_bytes: bytes, filename: str = "test.jpg") -> Dict[str, Any]:
        """Analyzes raw image bytes, runs perception and PPEReasonerNet, and returns full metrics."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode uploaded image bytes.")

        return self.analyze_cv2_image(img, filename=filename)

    def analyze_cv2_image(self, img: np.ndarray, filename: str = "test.jpg") -> Dict[str, Any]:
        """Runs full perception and reasoning pipeline on a BGR OpenCV image."""
        start_time = time.time()
        h, w = img.shape[:2]
        detections = self.detector.detect_frame(img, frame_id=0, timestamp=0.0)

        # Separate detections across whole image
        raw_person_dets = [d for d in detections if d.class_name == "person"]
        hardhat_dets = [d for d in detections if d.class_name == "hardhat"]
        no_hardhat_dets = [d for d in detections if d.class_name == "no_hardhat"]
        vest_dets = [d for d in detections if d.class_name == "vest"]
        machine_dets = [d for d in detections if d.class_name == "machinery"]

        # Filter out background architectural elements (pillars, beams) and suppressed duplicates
        person_dets = self.filter_valid_workers(raw_person_dets, img, hardhat_dets, vest_dets)

        matched_hh_indices = set()
        workers_analysis = []
        annotated_img = img.copy()

        for idx, p in enumerate(person_dets, start=1):
            px1, py1, px2, py2 = int(p.x1), int(p.y1), int(p.x2), int(p.y2)
            person_crop = img[py1:py2, px1:px2]
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)
            aspect = float(pw) / float(ph)

            # Step 1: Detect Face within the person crop (Haar cascade + morphology)
            has_face, face_conf, face_box, face_method = FaceCranialDetector.detect_face(
                person_crop, [float(px1), float(py1), float(px2), float(py2)]
            )
            if face_box is not None:
                fx1, fy1, fx2, fy2 = [int(round(v)) for v in face_box]
            else:
                # Fallback cranial anchor
                fx1 = px1 + int(pw * 0.25)
                fx2 = px1 + int(pw * 0.75)
                fy1 = py1 + int(ph * 0.08)
                fy2 = py1 + int(ph * 0.28)
                face_box = [float(fx1), float(fy1), float(fx2), float(fy2)]

            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)
            fcx = (fx1 + fx2) / 2.0
            fcy = (fy1 + fy2) / 2.0

            # Step 2: Strict Face-Anchored Hardhat Search
            # Find candidate hardhats in YOLO detections that sit directly on top of the face
            best_hh_match = None
            best_hh_metrics = None
            best_hh_square = None
            best_hh_idx = -1
            best_hh_score = 0.0

            for h_idx, hd in enumerate(hardhat_dets):
                raw_hd_box = [hd.x1, hd.y1, hd.x2, hd.y2]
                is_gd, v_conf, sq_box, metrics = PPEInspector.verify_hardhat_on_face(
                    raw_hd_box, face_box, [px1, py1, px2, py2]
                )
                if is_gd:
                    # Calculate combined score: confidence + horizontal alignment
                    align_score = v_conf * (1.0 - min(1.0, metrics.get("horiz_offset_px", 0.0) / max(1.0, metrics.get("max_allowed_h_offset", 1.0))))
                    if align_score > best_hh_score:
                        best_hh_score = align_score
                        best_hh_match = hd
                        best_hh_metrics = metrics
                        best_hh_square = sq_box
                        best_hh_idx = h_idx

            # Explicit check for NO-Hardhat detection near head
            cranial_rect = [float(fx1 - 0.20 * fw), float(max(0, fy1 - 1.25 * fh)), float(fx2 + 0.20 * fw), float(fy1 + 0.20 * fh)]
            has_no_hh_det = any(
                _calc_iou(cranial_rect, [nh.x1, nh.y1, nh.x2, nh.y2]) > 0.08
                for nh in no_hardhat_dets
            )

            hardhat_box_pixel = None
            from_yolo_det = False
            is_good_detect = False

            if best_hh_match is not None and best_hh_square is not None and not has_no_hh_det:
                has_helmet = True
                h_conf = best_hh_match.confidence
                matched_hh_indices.add(best_hh_idx)
                # Mark any other candidate hardhat boxes that overlap this worker's head as matched
                for other_idx, other_hd in enumerate(hardhat_dets):
                    if other_idx != best_hh_idx and other_idx not in matched_hh_indices:
                        o_iou, o_cont = self._box_metrics([other_hd.x1, other_hd.y1, other_hd.x2, other_hd.y2], best_hh_square)
                        if o_iou > 0.15 or o_cont > 0.35 or (best_hh_square[0] - 20 <= other_hd.center_x <= best_hh_square[2] + 20 and other_hd.y1 <= fy2):
                            matched_hh_indices.add(other_idx)
                from_yolo_det = True
                is_good_detect = True
                hardhat_box_pixel = [int(round(v)) for v in best_hh_square]
            else:
                # Step 3: Fallback to Face-Guided Cranial Locator in PPEInspector
                has_cranial_hh, cranial_conf, cranial_square, cranial_is_gd = PPEInspector.locate_cranial_hardhat_square(
                    person_crop, [px1, py1, px2, py2], face_box
                )
                if has_cranial_hh and cranial_square is not None and not has_no_hh_det:
                    has_helmet = True
                    h_conf = cranial_conf
                    is_good_detect = cranial_is_gd
                    hardhat_box_pixel = [int(round(v)) for v in cranial_square]
                    best_hh_metrics = {"reason": "cranial_pigment_on_face", "is_good_detect": True, "scale_ratio": 1.15}
                    for other_idx, other_hd in enumerate(hardhat_dets):
                        if other_idx not in matched_hh_indices:
                            o_iou, o_cont = self._box_metrics([other_hd.x1, other_hd.y1, other_hd.x2, other_hd.y2], cranial_square)
                            if o_iou > 0.15 or o_cont > 0.35:
                                matched_hh_indices.add(other_idx)
                else:
                    has_helmet = False
                    h_conf = 0.10
                    is_good_detect = False
                    hardhat_box_pixel = None
                    best_hh_metrics = {"reason": "no_hardhat_on_face", "is_good_detect": False}

            # Inspect person crop for vest
            _, _, has_vest, v_conf = PPEInspector.inspect_person_crop(person_crop)

            # Build bounding box coordinates normalized to 0.0 - 1.0
            p_box_norm = [p.x1 / w, p.y1 / h, p.x2 / w, p.y2 / h]
            if aspect >= 0.45:
                # Chest-up crop: vest spans from 32% down to bottom of crop
                v_box_norm = [p.x1 / w, (p.y1 + 0.32 * ph) / h, p.x2 / w, p.y2 / h] if has_vest else None
            else:
                v_box_norm = [p.x1 / w, (p.y1 + 0.20 * ph) / h, p.x2 / w, (p.y1 + 0.72 * ph) / h] if has_vest else None

            if hardhat_box_pixel is not None:
                h_box_norm = [hardhat_box_pixel[0] / w, hardhat_box_pixel[1] / h, hardhat_box_pixel[2] / w, hardhat_box_pixel[3] / h]
            else:
                h_box_norm = None

            # Extract 16-dim spatial vector with face-cranial awareness
            f_box_norm = [face_box[0] / w, face_box[1] / h, face_box[2] / w, face_box[3] / h] if face_box else None
            feats = extract_ppe_neural_features(
                worker_box=p_box_norm,
                vest_box=v_box_norm,
                hardhat_box=h_box_norm,
                worker_conf=p.confidence,
                vest_conf=v_conf if has_vest else 0.0,
                hardhat_conf=h_conf if has_helmet else 0.0,
                face_box=f_box_norm,
                is_good_detect=is_good_detect
            )

            # Pass through PPEReasonerNet with optional TTA
            if self.reasoner is not None:
                with torch.no_grad():
                    inp = torch.tensor([feats], dtype=torch.float32)

                    if self.use_tta and self.tta_passes > 1:
                        # Test-Time Augmentation: multiple passes with noise, average predictions
                        vest_scores = []
                        hh_scores = []
                        viol_logits_list = []
                        risk_scores_list = []

                        # Original pass (clean)
                        out0 = self.reasoner(inp)
                        vest_scores.append(out0["vest_compliance"][0, 0].item())
                        hh_scores.append(out0["hardhat_compliance"][0, 0].item())
                        viol_logits_list.append(out0["violation_logits"][0])
                        risk_scores_list.append(out0["risk_score"][0, 0].item())

                        # EMA model pass (if available)
                        if self.reasoner_ema is not None:
                            out_ema = self.reasoner_ema(inp)
                            vest_scores.append(out_ema["vest_compliance"][0, 0].item())
                            hh_scores.append(out_ema["hardhat_compliance"][0, 0].item())
                            viol_logits_list.append(out_ema["violation_logits"][0])
                            risk_scores_list.append(out_ema["risk_score"][0, 0].item())

                        # Noisy passes
                        for _ in range(self.tta_passes - 1):
                            noise = torch.randn_like(inp) * self.tta_noise_std
                            inp_noisy = inp + noise
                            out_n = self.reasoner(inp_noisy)
                            vest_scores.append(out_n["vest_compliance"][0, 0].item())
                            hh_scores.append(out_n["hardhat_compliance"][0, 0].item())
                            viol_logits_list.append(out_n["violation_logits"][0])
                            risk_scores_list.append(out_n["risk_score"][0, 0].item())

                        vest_comp = float(np.mean(vest_scores))
                        hh_comp = float(np.mean(hh_scores))
                        risk_score = float(np.mean(risk_scores_list))

                        # Average logits and apply temperature scaling
                        avg_logits = torch.stack(viol_logits_list).mean(dim=0)
                        scaled_logits = avg_logits / self.temperature
                        viol_class_idx = int(scaled_logits.argmax().item())
                    else:
                        out = self.reasoner(inp)
                        vest_comp = float(out["vest_compliance"][0, 0].item())
                        hh_comp = float(out["hardhat_compliance"][0, 0].item())
                        scaled_logits = out["violation_logits"][0] / self.temperature
                        viol_class_idx = int(scaled_logits.argmax().item())
                        risk_score = float(out["risk_score"][0, 0].item())
            else:
                # Fallback heuristic if reasoner weights unavailable
                vest_comp = 0.95 if has_vest else 0.05
                hh_comp = 0.95 if (has_helmet and is_good_detect) else 0.05
                if has_vest and has_helmet and is_good_detect:
                    viol_class_idx = 0
                    risk_score = 0.05
                elif not has_vest and has_helmet and is_good_detect:
                    viol_class_idx = 1
                    risk_score = 0.45
                elif has_vest and (not has_helmet or not is_good_detect):
                    viol_class_idx = 2
                    risk_score = 0.55
                else:
                    viol_class_idx = 3
                    risk_score = 0.90

            # Override hardhat compliance if no helmet was detected or not sitting on top of face
            if not has_helmet or not is_good_detect:
                hh_comp = min(hh_comp, 0.20)
                if viol_class_idx == 0:
                    viol_class_idx = 2  # MISSING_HARDHAT
                elif viol_class_idx == 1:
                    viol_class_idx = 3  # CRITICAL_NO_PPE

            viol_name = self.VIOLATION_NAMES.get(viol_class_idx, "UNKNOWN")
            color_bgr = self.VIOLATION_COLORS.get(viol_class_idx, (0, 255, 0))

            # Risk level and recommendation
            if risk_score >= 0.70:
                risk_level = "CRITICAL"
                action = "P0: Immediate Halt. Worker lacks vital cranial PPE in active perimeter."
            elif risk_score >= 0.40:
                risk_level = "HIGH RISK"
                action = "P1: Supervisor Intervention. Missing required PPE (Vest or Hardhat)."
            elif risk_score >= 0.20:
                risk_level = "MODERATE"
                action = "P2: Issue Warning. Confirm proper fitment of safety gear."
            else:
                risk_level = "SAFE"
                action = "Compliant. Worker adheres to OSHA/ANSI PPE protocol."

            has_hardhat_final = bool(hh_comp >= 0.5 and has_helmet and is_good_detect)
            worker_info = {
                "worker_id": f"Worker #{idx}",
                "confidence": round(p.confidence, 3),
                "box": [px1, py1, px2, py2],
                "box_norm": [round(c, 4) for c in p_box_norm],
                "has_vest": bool(vest_comp >= 0.5),
                "has_hardhat": has_hardhat_final,
                "vest_compliance_pct": round(vest_comp * 100.0, 1),
                "hardhat_compliance_pct": round(hh_comp * 100.0, 1),
                "vest_decision_margin": round(abs(vest_comp - 0.5), 4),
                "hardhat_decision_margin": round(abs(hh_comp - 0.5), 4),
                "violation_class": viol_name,
                "violation_id": viol_class_idx,
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level,
                "action": action,
                "inference_method": "TTA" if (self.use_tta and self.reasoner is not None) else "single_pass",
                "face_box": [fx1, fy1, fx2, fy2] if has_face else None,
                "face_box_norm": [round(fx1 / w, 4), round(fy1 / h, 4), round(fx2 / w, 4), round(fy2 / h, 4)] if has_face else None,
                "face_confidence": round(face_conf, 3),
                "face_method": face_method,
                "has_face": bool(has_face),
                "is_good_detect": bool(has_hardhat_final and is_good_detect),
                "sits_on_top_of_face": bool(has_hardhat_final and is_good_detect),
                "hardhat_metrics": best_hh_metrics,
                "hardhat_box": [int(round(v)) for v in hardhat_box_pixel] if (has_hardhat_final and hardhat_box_pixel) else None,
                "hardhat_box_norm": [round(hardhat_box_pixel[0] / w, 4), round(hardhat_box_pixel[1] / h, 4), round(hardhat_box_pixel[2] / w, 4), round(hardhat_box_pixel[3] / h, 4)] if (has_hardhat_final and hardhat_box_pixel) else None,
                "has_hardhat_square": bool(has_hardhat_final and hardhat_box_pixel is not None),
                "hardhat_square_dims": f"{hardhat_box_pixel[2]-hardhat_box_pixel[0]}x{hardhat_box_pixel[3]-hardhat_box_pixel[1]}px" if (has_hardhat_final and hardhat_box_pixel) else None,
                "from_yolo_det": from_yolo_det
            }
            workers_analysis.append(worker_info)

            # Draw visual annotations:
            # 1. Main worker body bounding box
            cv2.rectangle(annotated_img, (px1, py1), (px2, py2), color_bgr, 2)

            # 2. Worker label banner
            label_text = f"#{idx} {viol_name} | Risk: {risk_score:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            banner_y1 = max(0, py1 - text_h - 10)
            banner_y2 = py1
            cv2.rectangle(annotated_img, (px1, banner_y1), (px1 + text_w + 12, banner_y2), color_bgr, -1)
            cv2.putText(
                annotated_img,
                label_text,
                (px1 + 6, py1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # 3. DETECTED FACE BOX
            if has_face and face_box is not None:
                face_color = (255, 180, 0)  # Sky blue / cyan in BGR
                cv2.rectangle(annotated_img, (fx1, fy1), (fx2, fy2), face_color, 1, cv2.LINE_AA)
                face_lbl = f"FACE {int(face_conf * 100)}%"
                (fl_w, fl_h), _ = cv2.getTextSize(face_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(annotated_img, (fx1, fy2), (fx1 + fl_w + 4, fy2 + fl_h + 4), face_color, -1)
                cv2.putText(annotated_img, face_lbl, (fx1 + 2, fy2 + fl_h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

            # 4. DEDICATED HARDHAT SQUARE BOX (GOOD DETECT)
            if has_hardhat_final and hardhat_box_pixel is not None and is_good_detect:
                hx1, hy1, hx2, hy2 = [int(round(v)) for v in hardhat_box_pixel]
                # High-visibility Gold square for hardhat
                hh_color = (0, 235, 255)  # Bright gold BGR
                cv2.rectangle(annotated_img, (hx1, hy1), (hx2, hy2), hh_color, 2, cv2.LINE_AA)
                # Hardhat tag above square
                hh_tag = f"HARDHAT [GOOD DETECT] {int(hh_comp * 100)}%"
                (tw, th), _ = cv2.getTextSize(hh_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                tag_y1 = max(0, hy1 - th - 6)
                cv2.rectangle(annotated_img, (hx1, tag_y1), (hx1 + tw + 6, hy1), hh_color, -1)
                cv2.putText(annotated_img, hh_tag, (hx1 + 3, hy1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
                # Visual link from hardhat bottom to face top (sits on top)
                cv2.line(annotated_img, (int((hx1 + hx2) / 2), hy2), (int((fx1 + fx2) / 2), fy1), hh_color, 1, cv2.LINE_AA)
            else:
                # Warning square on missing hardhat cranial zone directly on top of face
                c_side = max(fw, int(fh * 1.15))
                hx1_sq = max(0, int(fcx - c_side // 2))
                hx2_sq = min(w, hx1_sq + c_side)
                hy2_sq = fy1 + int(0.10 * fh)
                hy1_sq = max(0, hy2_sq - c_side)
                no_hh_color = (60, 76, 231)  # Crimson red
                cv2.rectangle(annotated_img, (hx1_sq, hy1_sq), (hx2_sq, hy2_sq), no_hh_color, 2, cv2.LINE_AA)
                no_hh_tag = "MISSING HARDHAT (HEAD EXPOSED)"
                (tw, th), _ = cv2.getTextSize(no_hh_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                tag_y1 = max(0, hy1_sq - th - 6)
                cv2.rectangle(annotated_img, (hx1_sq, tag_y1), (hx1_sq + tw + 6, hy1_sq), no_hh_color, -1)
                cv2.putText(annotated_img, no_hh_tag, (hx1_sq + 3, hy1_sq - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Standalone / Unattended hardhats in whole image
        unattended_hardhats = []
        for h_idx, hd in enumerate(hardhat_dets):
            if h_idx not in matched_hh_indices and hd.confidence >= 0.55:
                inside_worker = any(
                    w["box"][0] - 15 <= hd.center_x <= w["box"][2] + 15 and w["box"][1] - 15 <= hd.center_y <= w["box"][3] + 15
                    for w in workers_analysis
                )
                if inside_worker:
                    continue
                hx1, hy1, hx2, hy2 = int(hd.x1), int(hd.y1), int(hd.x2), int(hd.y2)
                side = max(hx2 - hx1, hy2 - hy1)
                hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                hx1_sq = max(0, hcx - side // 2)
                hy1_sq = max(0, hcy - side // 2)
                hx2_sq = min(w, hx1_sq + side)
                hy2_sq = min(h, hy1_sq + side)

                cv2.rectangle(annotated_img, (hx1_sq, hy1_sq), (hx2_sq, hy2_sq), (0, 165, 255), 2)
                unatt_tag = f"HARDHAT (STANDALONE) {int(hd.confidence * 100)}%"
                (tw, th), _ = cv2.getTextSize(unatt_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                tag_y1 = max(0, hy1_sq - th - 6)
                cv2.rectangle(annotated_img, (hx1_sq, tag_y1), (hx1_sq + tw + 6, hy1_sq), (0, 165, 255), -1)
                cv2.putText(annotated_img, unatt_tag, (hx1_sq + 3, hy1_sq - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
                unattended_hardhats.append({
                    "box": [hx1_sq, hy1_sq, hx2_sq, hy2_sq],
                    "box_norm": [round(hx1_sq / w, 4), round(hy1_sq / h, 4), round(hx2_sq / w, 4), round(hy2_sq / h, 4)],
                    "confidence": round(hd.confidence, 3),
                    "status": "STANDALONE"
                })

        # If no workers were detected, check if image has person
        if not person_dets:
            cv2.putText(
                annotated_img,
                "No Workers Detected in Image",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

        # Encode annotated image to Base64 JPEG for web rendering
        success, encoded_jpg = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        base64_image = base64.b64encode(encoded_jpg).decode("utf-8") if success else ""

        # Aggregate summary metrics
        total_workers = len(workers_analysis)
        compliant_count = sum(1 for w in workers_analysis if w["violation_id"] == 0)
        violation_count = total_workers - compliant_count

        vest_compliance_rate = round(
            (sum(1 for w in workers_analysis if w["has_vest"]) / max(1, total_workers)) * 100.0, 1
        )
        hardhat_compliance_rate = round(
            (sum(1 for w in workers_analysis if w["has_hardhat"]) / max(1, total_workers)) * 100.0, 1
        )
        avg_risk = round(
            sum(w["risk_score"] for w in workers_analysis) / max(1, total_workers), 3
        ) if total_workers > 0 else 0.0

        if total_workers == 0:
            site_status = "NO_WORKERS"
        elif violation_count == 0:
            site_status = "FULL_COMPLIANCE"
        elif any(w["violation_id"] == 3 for w in workers_analysis):
            site_status = "CRITICAL_VIOLATIONS"
        else:
            site_status = "PARTIAL_NON_COMPLIANCE"

        # Dispatch real-time telemetry trace to PRISM
        try:
            worker_details = ", ".join([f"{w['worker_id']}: {w['violation_class']} (Risk {w['risk_score']:.2f})" for w in workers_analysis]) or "No workers detected"
            compliance_risk = round(float(avg_risk), 3)
            quality_score = round(max(0.0, min(1.0, 1.0 - compliance_risk)), 3)
            response_quality = 0.98 if total_workers > 0 else 0.92
            compliance_score = round((vest_compliance_rate + hardhat_compliance_rate) / 2.0, 1) if total_workers > 0 else 100.0

            output_desc = (
                f"PPE Construction Safety Inspection ({filename}):\n"
                f"• Status: {site_status}\n"
                f"• Workers Evaluated: {total_workers} (Compliant: {compliant_count} | Violations: {violation_count})\n"
                f"• Hardhat Compliance: {hardhat_compliance_rate}% (OSHA 1926.100 Cranial Protection)\n"
                f"• High-Vis Vest Compliance: {vest_compliance_rate}% (OSHA 1926.201 Standard)\n"
                f"• Compliance Risk: {compliance_risk} | Quality Score: {quality_score} | Response Quality: {response_quality}\n"
                f"• Neural Breakdown: [{worker_details}]"
            )
            latency_ms = max(1, int((time.time() - start_time) * 1000))

            def _send_trace():
                try:
                    self.prism_client.emit_trace(
                        input_text=f"PPE Analyser Inspection: {filename} ({w}x{h}, {total_workers} workers, database: datasets/ppe_master_folder)",
                        output_text=output_desc,
                        latency_ms=latency_ms,
                        agent_name="rishabh",
                        model="PPEReasonerNet-V1",
                        session_id="ppe-analyser-session",
                        metadata={
                            "quality_score": quality_score,
                            "response_quality": response_quality,
                            "compliance_risk": compliance_risk,
                            "compliance_score": compliance_score,
                            "Quality Score": quality_score,
                            "Response Quality": response_quality,
                            "Compliance Risk": compliance_risk,
                            "Compliance Score": compliance_score,
                            "filename": filename,
                            "workers_count": total_workers,
                            "compliant_count": compliant_count,
                            "violation_count": violation_count,
                            "vest_rate": vest_compliance_rate,
                            "hardhat_rate": hardhat_compliance_rate,
                            "site_risk": compliance_risk,
                            "site_status": site_status,
                            "database": "datasets/ppe_master_folder",
                            "domain": "PPE Construction Safety"
                        }
                    )
                except Exception:
                    pass

            threading.Thread(target=_send_trace, daemon=True).start()
        except Exception:
            pass

        return {
            "status": "success",
            "filename": filename,
            "image_dimensions": {"width": w, "height": h},
            "summary_metrics": {
                "total_workers": total_workers,
                "compliant_count": compliant_count,
                "violation_count": violation_count,
                "vest_compliance_rate_pct": vest_compliance_rate,
                "hardhat_compliance_rate_pct": hardhat_compliance_rate,
                "avg_risk_score": avg_risk,
                "site_status": site_status
            },
            "hardhat_summary": {
                "total_hardhats_in_scene": len(hardhat_dets) + sum(1 for w in workers_analysis if w["has_hardhat"] and not w.get("from_yolo_det")),
                "workers_with_hardhat": sum(1 for w in workers_analysis if w["has_hardhat"]),
                "workers_missing_hardhat": sum(1 for w in workers_analysis if not w["has_hardhat"]),
                "unattended_hardhats_count": len(unattended_hardhats)
            },
            "workers": workers_analysis,
            "unattended_hardhats": unattended_hardhats,
            "annotated_image_base64": f"data:image/jpeg;base64,{base64_image}",
            "files_needed_by_analyser": self.FILES_USED,
            "neural_model_info": {
                "architecture": "PPEReasonerNet V3 (Residual MLP + HardhatReasonerBlock + 4 Multi-Task Heads)",
                "input_dimension": 16,
                "hidden_dimension": 64,
                "weights_file": self.model_path,
                "perception_backbone": "YOLOv8 PPE Detector + High-Vis Cranial Vault Locator"
            }
        }
