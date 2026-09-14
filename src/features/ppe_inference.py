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

from src.risk_model.ppe_reasoner_net import PPEReasonerNet, extract_ppe_neural_features
from src.perception.detector import SafetyDetector, PPEInspector, Detection
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
        "src/risk_model/ppe_reasoner_net.py (PPEReasonerNet Architecture & 14-dim Feature Extractor)",
        "src/perception/detector.py (YOLOv8 + PPE Visual Perception Inspector)",
        "src/features/ppe_inference.py (Inference Engine & Decision Pipeline)",
        "web/server.py (FastAPI Endpoint /api/ppe/analyze)",
        "web/index.html & web/static/js/app.js (PPE Analyser Interactive UI & Canvas)"
    ]

    def __init__(self, model_path: str = "models/ppe_reasoner.pt", yolo_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.yolo_path = yolo_path

        # Load PPEReasonerNet
        self.device = torch.device("cpu")
        self.reasoner = None
        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                in_dim = ckpt.get("input_dim", 14)
                hid_dim = ckpt.get("hidden_dim", 64)
                self.reasoner = PPEReasonerNet(input_dim=in_dim, hidden_dim=hid_dim)
                self.reasoner.load_state_dict(ckpt["state_dict"])
                self.reasoner.eval()
            except Exception as e:
                print(f"[Warning] Failed to load PPEReasonerNet: {e}")

        # Load YOLOv8
        self.detector = SafetyDetector(model_path=yolo_path, conf_threshold=0.25)
        self.prism_client = PrismClient()

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

        # Separate persons from other objects
        person_dets = [d for d in detections if d.class_name == "person"]
        machine_dets = [d for d in detections if d.class_name == "machinery"]

        workers_analysis = []
        annotated_img = img.copy()

        for idx, p in enumerate(person_dets, start=1):
            px1, py1, px2, py2 = int(p.x1), int(p.y1), int(p.x2), int(p.y2)
            person_crop = img[py1:py2, px1:px2]

            # Inspect person crop for helmet & vest
            has_helmet, h_conf, has_vest, v_conf = PPEInspector.inspect_person_crop(person_crop)

            # Build bounding box coordinates normalized to 0.0 - 1.0
            p_box_norm = [p.x1 / w, p.y1 / h, p.x2 / w, p.y2 / h]
            aspect = float(p.width) / float(max(1.0, p.height))
            if aspect >= 0.45:
                # Chest-up crop: vest spans from 32% down to bottom of crop
                v_box_norm = [p.x1 / w, (p.y1 + 0.32 * p.height) / h, p.x2 / w, p.y2 / h] if has_vest else None
                h_box_norm = [p.x1 / w, p.y1 / h, p.x2 / w, (p.y1 + 0.35 * p.height) / h] if has_helmet else None
            else:
                v_box_norm = [p.x1 / w, (p.y1 + 0.20 * p.height) / h, p.x2 / w, (p.y1 + 0.72 * p.height) / h] if has_vest else None
                h_box_norm = [p.x1 / w, p.y1 / h, p.x2 / w, (p.y1 + 0.20 * p.height) / h] if has_helmet else None

            # Extract 16-dim spatial vector with crop-geometry awareness
            feats = extract_ppe_neural_features(
                worker_box=p_box_norm,
                vest_box=v_box_norm,
                hardhat_box=h_box_norm,
                worker_conf=p.confidence,
                vest_conf=v_conf if has_vest else 0.0,
                hardhat_conf=h_conf if has_helmet else 0.0
            )

            # Pass through PPEReasonerNet
            if self.reasoner is not None:
                with torch.no_grad():
                    inp = torch.tensor([feats], dtype=torch.float32)
                    out = self.reasoner(inp)
                    vest_comp = float(out["vest_compliance"][0, 0].item())
                    hh_comp = float(out["hardhat_compliance"][0, 0].item())
                    viol_class_idx = int(out["violation_logits"].argmax(dim=1)[0].item())
                    risk_score = float(out["risk_score"][0, 0].item())
            else:
                # Fallback heuristic if reasoner weights unavailable
                vest_comp = 0.95 if has_vest else 0.05
                hh_comp = 0.95 if has_helmet else 0.05
                if has_vest and has_helmet:
                    viol_class_idx = 0
                    risk_score = 0.05
                elif not has_vest and has_helmet:
                    viol_class_idx = 1
                    risk_score = 0.45
                elif has_vest and not has_helmet:
                    viol_class_idx = 2
                    risk_score = 0.55
                else:
                    viol_class_idx = 3
                    risk_score = 0.90

            viol_name = self.VIOLATION_NAMES.get(viol_class_idx, "UNKNOWN")
            color_bgr = self.VIOLATION_COLORS.get(viol_class_idx, (0, 255, 0))

            # Risk level and recommendation
            if risk_score >= 0.70:
                risk_level = "CRITICAL"
                action = "P0: Immediate Halt. Worker lacks vital PPE in active perimeter."
            elif risk_score >= 0.40:
                risk_level = "HIGH RISK"
                action = "P1: Supervisor Intervention. Missing required PPE (Vest or Hardhat)."
            elif risk_score >= 0.20:
                risk_level = "MODERATE"
                action = "P2: Issue Warning. Confirm proper fitment of safety gear."
            else:
                risk_level = "SAFE"
                action = "Compliant. Worker adheres to OSHA/ANSI PPE protocol."

            worker_info = {
                "worker_id": f"Worker #{idx}",
                "confidence": round(p.confidence, 3),
                "box": [px1, py1, px2, py2],
                "box_norm": [round(c, 4) for c in p_box_norm],
                "has_vest": bool(vest_comp >= 0.5),
                "has_hardhat": bool(hh_comp >= 0.5),
                "vest_compliance_pct": round(vest_comp * 100.0, 1),
                "hardhat_compliance_pct": round(hh_comp * 100.0, 1),
                "violation_class": viol_name,
                "violation_id": viol_class_idx,
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level,
                "action": action
            }
            workers_analysis.append(worker_info)

            # Draw visual annotation on image
            cv2.rectangle(annotated_img, (px1, py1), (px2, py2), color_bgr, 3)

            # Draw label banner
            label_text = f"#{idx} {viol_name} | Risk: {risk_score:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            banner_y1 = max(0, py1 - text_h - 10)
            banner_y2 = py1
            cv2.rectangle(annotated_img, (px1, banner_y1), (px1 + text_w + 12, banner_y2), color_bgr, -1)
            cv2.putText(
                annotated_img,
                label_text,
                (px1 + 6, py1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

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
            output_desc = f"PPEReasonerNet Status: {site_status} | Total: {total_workers} | Compliant: {compliant_count} | Violations: {violation_count} | Vest: {vest_compliance_rate}% | Hardhat: {hardhat_compliance_rate}% | Avg Risk: {avg_risk} | Breakdown: [{worker_details}]"
            latency_ms = max(1, int((time.time() - start_time) * 1000))

            def _send_trace():
                try:
                    self.prism_client.emit_trace(
                        input_text=f"PPE Analyser Inspection: {filename} ({w}x{h}, {total_workers} workers)",
                        output_text=output_desc,
                        latency_ms=latency_ms,
                        agent_name="rishabh",
                        model="PPEReasonerNet-V1",
                        session_id="ppe-analyser-session",
                        metadata={
                            "filename": filename,
                            "workers_count": total_workers,
                            "compliant_count": compliant_count,
                            "violation_count": violation_count,
                            "vest_rate": vest_compliance_rate,
                            "hardhat_rate": hardhat_compliance_rate,
                            "site_risk": avg_risk,
                            "site_status": site_status
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
            "workers": workers_analysis,
            "annotated_image_base64": f"data:image/jpeg;base64,{base64_image}",
            "files_needed_by_analyser": self.FILES_USED,
            "neural_model_info": {
                "architecture": "PPEReasonerNet V2 (Residual MLP + 4 Multi-Task Heads + Crop Geometry Invariance)",
                "input_dimension": 16,
                "hidden_dimension": 64,
                "weights_file": self.model_path,
                "perception_backbone": "YOLOv8 + High-Vis Anatomical Perception"
            }
        }
