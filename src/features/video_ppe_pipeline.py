"""Video PPE Pipeline.
Dissects uploaded construction video streams into sequential frames,
executes perception and PPEReasonerNet inference on each frame,
and compiles temporal compliance metrics and violation heatmaps for playback.
"""
import os
import cv2
import time
import base64
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.features.ppe_inference import PPEInferenceEngine


class VideoPPEPipeline:
    """End-to-end video dissection, frame-by-frame inference, and playback generator."""

    def __init__(self, ppe_engine: Optional[PPEInferenceEngine] = None):
        self.engine = ppe_engine if ppe_engine is not None else PPEInferenceEngine()

    @staticmethod
    def _encode_cv2_image_to_base64(img: np.ndarray, quality: int = 75) -> str:
        """Encodes an annotated BGR image to a compact base64 JPEG data URL."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", img, encode_params)
        if not success:
            return ""
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def process_video(
        self,
        video_path: str,
        target_fps: float = 8.0,
        max_frames: int = 120,
        jpeg_quality: int = 75
    ) -> Dict[str, Any]:
        """Dissects a video into frames and executes PPE neural inference on each frame.

        Args:
            video_path: Path to the local video file.
            target_fps: Target sampling rate (frames per second). Default 8.0.
            max_frames: Upper cap on total frames to analyze to maintain fast response.
            jpeg_quality: Compression quality for the base64 annotated frames (1-100).

        Returns:
            Dict containing video metadata, summary metrics, violation timeline,
            and frame-by-frame records with annotated base64 images and worker cards.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video stream: {video_path}")

        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0.0:
            source_fps = 25.0

        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_source_frames / source_fps if total_source_frames > 0 else 0.0

        # Determine frame skip interval to achieve target_fps
        frame_interval = max(1, int(round(source_fps / max(1.0, target_fps))))

        processed_frames: List[Dict[str, Any]] = []
        violation_timeline: List[Dict[str, Any]] = []

        total_workers_observed = 0
        total_compliant_worker_instances = 0
        total_violation_instances = 0
        peak_risk_score = 0.0
        peak_risk_frame_idx = 0
        overall_hardhat_compliant_instances = 0
        overall_vest_compliant_instances = 0

        start_time = time.time()
        source_frame_idx = 0
        extracted_frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames based on calculated interval
            if source_frame_idx % frame_interval == 0:
                timestamp_sec = round(source_frame_idx / source_fps, 2)

                # Run PPE perception & PPEReasonerNet inference
                frame_res = self.engine.analyze_cv2_image(
                    frame, filename=f"frame_{extracted_frame_idx:04d}.jpg"
                )

                annotated_img = frame_res.get("annotated_image")
                if annotated_img is not None:
                    annotated_b64 = self._encode_cv2_image_to_base64(annotated_img, quality=jpeg_quality)
                else:
                    annotated_b64 = self._encode_cv2_image_to_base64(frame, quality=jpeg_quality)

                workers = frame_res.get("workers", [])
                summary = frame_res.get("summary_metrics", {})
                frame_risk = float(summary.get("avg_risk_score", 0.0))

                # Accumulate stats
                w_count = len(workers)
                total_workers_observed += w_count
                comp_count = int(summary.get("compliant_count", 0))
                viol_count = int(summary.get("violation_count", 0))
                total_compliant_worker_instances += comp_count
                total_violation_instances += viol_count

                for w in workers:
                    if w.get("has_hardhat", False):
                        overall_hardhat_compliant_instances += 1
                    if w.get("has_vest", False):
                        overall_vest_compliant_instances += 1

                if frame_risk > peak_risk_score:
                    peak_risk_score = frame_risk
                    peak_risk_frame_idx = extracted_frame_idx

                # Timeline tick representation for scrubber heatmap:
                # 0 = Compliant (Green), 1 = Missing Vest (Yellow), 2 = Missing Helmet / Critical (Red)
                has_critical_hh_breach = any(not w.get("has_hardhat", False) for w in workers)
                has_vest_breach = any(not w.get("has_vest", False) for w in workers)
                
                if w_count == 0:
                    status_code = 0  # No workers = safe
                    tick_color = "emerald"
                elif has_critical_hh_breach:
                    status_code = 2  # Critical No Hardhat
                    tick_color = "red"
                elif has_vest_breach:
                    status_code = 1  # Moderate Missing Vest
                    tick_color = "amber"
                else:
                    status_code = 0  # Fully Compliant
                    tick_color = "emerald"

                timeline_tick = {
                    "frame_index": extracted_frame_idx,
                    "source_frame": source_frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "workers_count": w_count,
                    "risk_score": round(frame_risk, 3),
                    "status_code": status_code,
                    "tick_color": tick_color,
                    "has_critical_hh_breach": has_critical_hh_breach
                }
                violation_timeline.append(timeline_tick)

                # Store complete frame analysis payload for playback
                frame_data = {
                    "frame_index": extracted_frame_idx,
                    "source_frame_index": source_frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "time_display": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}.{int((timestamp_sec % 1) * 10)}",
                    "annotated_image_base64": annotated_b64,
                    "workers": workers,
                    "hardhat_summary": frame_res.get("hardhat_summary", {}),
                    "summary_metrics": summary,
                    "unattended_hardhats": frame_res.get("unattended_hardhats", [])
                }
                processed_frames.append(frame_data)
                extracted_frame_idx += 1

                if extracted_frame_idx >= max_frames:
                    break

            source_frame_idx += 1

        cap.release()
        total_processing_time = round(time.time() - start_time, 2)
        total_extracted = len(processed_frames)

        # Calculate clip-level aggregate metrics
        avg_risk = round(float(np.mean([t["risk_score"] for t in violation_timeline])) if violation_timeline else 0.0, 3)
        overall_compliance_rate = round(
            (total_compliant_worker_instances / max(1, total_workers_observed)) * 100.0, 1
        ) if total_workers_observed > 0 else 100.0

        overall_hh_rate = round(
            (overall_hardhat_compliant_instances / max(1, total_workers_observed)) * 100.0, 1
        ) if total_workers_observed > 0 else 100.0

        overall_vest_rate = round(
            (overall_vest_compliant_instances / max(1, total_workers_observed)) * 100.0, 1
        ) if total_workers_observed > 0 else 100.0

        if any(t["status_code"] == 2 for t in violation_timeline):
            overall_status = "CRITICAL_VIOLATIONS"
        elif any(t["status_code"] == 1 for t in violation_timeline):
            overall_status = "MODERATE_VIOLATIONS"
        else:
            overall_status = "FULL_COMPLIANCE"

        effective_playback_fps = min(float(target_fps), round(total_extracted / max(0.1, duration_sec), 1) if duration_sec > 0 else float(target_fps))

        return {
            "status": "success",
            "video_metadata": {
                "filename": os.path.basename(video_path),
                "duration_sec": round(duration_sec, 2),
                "source_fps": round(source_fps, 2),
                "playback_fps": effective_playback_fps,
                "total_source_frames": total_source_frames,
                "extracted_frames_count": total_extracted,
                "resolution": {"width": source_width, "height": source_height},
                "processing_time_sec": total_processing_time,
                "avg_frame_latency_ms": round((total_processing_time / max(1, total_extracted)) * 1000, 1)
            },
            "clip_summary": {
                "overall_status": overall_status,
                "total_workers_observed": total_workers_observed,
                "overall_compliance_rate_pct": overall_compliance_rate,
                "overall_hardhat_rate_pct": overall_hh_rate,
                "overall_vest_rate_pct": overall_vest_rate,
                "avg_risk_score": avg_risk,
                "peak_risk_score": round(peak_risk_score, 3),
                "peak_risk_frame_index": peak_risk_frame_idx,
                "critical_violation_frames_count": sum(1 for t in violation_timeline if t["status_code"] == 2),
                "moderate_violation_frames_count": sum(1 for t in violation_timeline if t["status_code"] == 1)
            },
            "violation_timeline": violation_timeline,
            "frames": processed_frames
        }
