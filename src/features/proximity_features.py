"""Proximity Feature Extraction for Large Vehicle Safety Scoring.

Extracts a 20-dimensional spatial feature vector for each (worker, vehicle)
pair in a scene. Labels (proximity_class, risk_score) are derived heuristically
from bounding box geometry since the dataset has no explicit proximity annotations.

Feature vector layout (all values normalized 0→1 unless noted):
  [0]  worker_cx         — worker center x
  [1]  worker_cy         — worker center y
  [2]  worker_w          — worker bbox width
  [3]  worker_h          — worker bbox height
  [4]  vehicle_cx        — vehicle center x
  [5]  vehicle_cy        — vehicle center y
  [6]  vehicle_w         — vehicle bbox width
  [7]  vehicle_h         — vehicle bbox height
  [8]  vehicle_class_id  — 0=tractor, 1=truck (normalized: /1.0)
  [9]  center_distance   — euclidean center-to-center distance (normalized to [0,√2])
  [10] iou               — intersection-over-union
  [11] rel_x             — worker cx relative to vehicle cx (signed, shifted +0.5)
  [12] rel_y             — worker cy relative to vehicle cy (signed, shifted +0.5)
  [13] vehicle_area      — vehicle bbox area (w*h)
  [14] worker_area       — worker bbox area (w*h)
  [15] area_ratio        — vehicle_area / (worker_area + 1e-6), clipped to [0,1]
  [16] worker_inside     — 1.0 if worker center inside vehicle bbox, else 0.0
  [17] in_danger_zone    — 1.0 if distance < DANGER_THRESH
  [18] in_supervised_zone— 1.0 if distance < SUPERVISED_THRESH
  [19] too_far_flag      — 1.0 if distance > TOO_FAR_THRESH

Proximity classes:
  0 = SAFE        (supervised_thresh < dist < too_far_thresh)
  1 = SUPERVISED  (danger_thresh < dist ≤ supervised_thresh)
  2 = DANGER      (dist ≤ danger_thresh or worker_inside)
  3 = TOO_FAR     (dist > too_far_thresh)
"""

import math
import random
from typing import List, Dict, Any, Tuple, Optional

import torch
import numpy as np

# Thresholds (normalized image coordinates, 0→1 scale)
DANGER_THRESH      = 0.15   # worker within 15% of image width from vehicle center
SUPERVISED_THRESH  = 0.30   # within 30%
TOO_FAR_THRESH     = 0.55   # beyond 55% — probably not monitoring the vehicle

VEHICLE_CLASS_MAP = {
    "tractor": 0,
    "truck": 1,
    "crane": 1,
    "excavator": 1,
    "bulldozer": 1,
    "forklift": 1,
    "vehicle": 1,
    "large vehicle": 1,
}

PROXIMITY_CLASS_NAMES = ["SAFE", "SUPERVISED", "DANGER", "TOO_FAR"]
PROXIMITY_CLASS_DANGER = 2
PROXIMITY_CLASS_TOO_FAR = 3


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def _iou(b1: List[float], b2: List[float]) -> float:
    """Compute IoU between two YOLO-format [cx, cy, w, h] boxes (normalized)."""
    x1_min = b1[0] - b1[2] / 2;  x1_max = b1[0] + b1[2] / 2
    y1_min = b1[1] - b1[3] / 2;  y1_max = b1[1] + b1[3] / 2
    x2_min = b2[0] - b2[2] / 2;  x2_max = b2[0] + b2[2] / 2
    y2_min = b2[1] - b2[3] / 2;  y2_max = b2[1] + b2[3] / 2

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h

    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / (union + 1e-6)


def _center_distance(b1: List[float], b2: List[float]) -> float:
    """Euclidean distance between two bbox centers (normalized coords)."""
    return math.sqrt((b1[0] - b2[0]) ** 2 + (b1[1] - b2[1]) ** 2)


def _worker_inside_vehicle(worker: List[float], vehicle: List[float]) -> bool:
    """True if worker center is within vehicle bbox."""
    v_xmin = vehicle[0] - vehicle[2] / 2
    v_xmax = vehicle[0] + vehicle[2] / 2
    v_ymin = vehicle[1] - vehicle[3] / 2
    v_ymax = vehicle[1] + vehicle[3] / 2
    return (v_xmin <= worker[0] <= v_xmax) and (v_ymin <= worker[1] <= v_ymax)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_proximity_feature(
    worker_bbox: List[float],
    vehicle_bbox: List[float],
    vehicle_class: str,
) -> List[float]:
    """Extract 20-dim feature vector for a single (worker, vehicle) pair.

    Args:
        worker_bbox: YOLO [cx, cy, w, h] normalized
        vehicle_bbox: YOLO [cx, cy, w, h] normalized
        vehicle_class: string class name (e.g. 'tractor', 'truck')

    Returns:
        List of 20 floats
    """
    wcx, wcy, ww, wh = worker_bbox
    vcx, vcy, vw, vh = vehicle_bbox

    cls_id = VEHICLE_CLASS_MAP.get(vehicle_class.lower(), 1)
    dist = _center_distance(worker_bbox, vehicle_bbox)
    iou = _iou(worker_bbox, vehicle_bbox)
    inside = 1.0 if _worker_inside_vehicle(worker_bbox, vehicle_bbox) else 0.0

    # Relative position (signed, shifted to [0,1] range)
    rel_x = (wcx - vcx) + 0.5
    rel_y = (wcy - vcy) + 0.5

    # Areas and ratio (capped)
    v_area = vw * vh
    w_area = ww * wh
    area_ratio = min(1.0, v_area / (w_area + 1e-6)) / 10.0  # scale down

    # Zone flags
    in_danger = 1.0 if (dist <= DANGER_THRESH or inside) else 0.0
    in_supervised = 1.0 if (dist <= SUPERVISED_THRESH) else 0.0
    too_far = 1.0 if dist > TOO_FAR_THRESH else 0.0

    # Normalize distance to [0, 1] (max diagonal of unit square = √2 ≈ 1.414)
    dist_norm = min(1.0, dist / 1.414)

    feat = [
        wcx, wcy, ww, wh,                    # 0-3: worker bbox
        vcx, vcy, vw, vh,                    # 4-7: vehicle bbox
        float(cls_id),                        # 8: vehicle class id
        dist_norm,                            # 9: center distance
        iou,                                  # 10: iou
        max(0.0, min(1.0, rel_x)),            # 11: relative x
        max(0.0, min(1.0, rel_y)),            # 12: relative y
        v_area,                               # 13: vehicle area
        w_area,                               # 14: worker area
        area_ratio,                           # 15: area ratio
        inside,                               # 16: worker inside vehicle
        in_danger,                            # 17: in danger zone
        in_supervised,                        # 18: in supervised zone
        too_far,                              # 19: too far flag
    ]
    return feat


def derive_proximity_label(
    worker_bbox: List[float],
    vehicle_bbox: List[float],
) -> Tuple[int, float]:
    """Derive heuristic proximity class and continuous risk score.

    Returns:
        (proximity_class, risk_score)
        proximity_class: 0=SAFE, 1=SUPERVISED, 2=DANGER, 3=TOO_FAR
        risk_score: float in [0, 1]
    """
    dist = _center_distance(worker_bbox, vehicle_bbox)
    inside = _worker_inside_vehicle(worker_bbox, vehicle_bbox)

    if inside or dist <= DANGER_THRESH:
        prox_class = 2  # DANGER
        # Higher risk if more inside / closer
        risk = max(0.75, 1.0 - dist / DANGER_THRESH * 0.25)
    elif dist <= SUPERVISED_THRESH:
        prox_class = 1  # SUPERVISED
        # Linear interpolation from ~0.5 (at supervised boundary) to ~0.75 (near danger)
        t = 1.0 - (dist - DANGER_THRESH) / (SUPERVISED_THRESH - DANGER_THRESH)
        risk = 0.40 + t * 0.35
    elif dist <= TOO_FAR_THRESH:
        prox_class = 0  # SAFE
        # Low risk, slight increase as approaching supervised
        t = (dist - SUPERVISED_THRESH) / (TOO_FAR_THRESH - SUPERVISED_THRESH)
        risk = 0.20 - t * 0.15
    else:
        prox_class = 3  # TOO_FAR
        risk = 0.30 + min(0.40, (dist - TOO_FAR_THRESH) * 0.5)

    return prox_class, float(max(0.0, min(1.0, risk)))


# ---------------------------------------------------------------------------
# Sample extraction from manifest entries
# ---------------------------------------------------------------------------

def extract_proximity_samples(
    manifest: Dict[str, Any],
    augment: bool = True,
    hard_negative_ratio: float = 0.20,
    synthetic_danger_ratio: float = 0.15,
    max_pairs_per_image: int = 8,
) -> List[Dict[str, Any]]:
    """Extract training samples from ingestor manifest.

    For each image entry, pairs all (worker, vehicle) combinations.
    For images with vehicles but no workers, synthesizes virtual worker positions
    to provide DANGER and TOO_FAR training signal.

    Returns list of dicts:
      {
        "features": List[float] (len=20),
        "proximity_class": int (0-3),
        "risk_score": float (0-1),
        "vehicle_class_id": int (0 or 1),
      }
    """
    samples = []

    for entry in manifest.get("entries", []):
        vehicles = entry.get("vehicles", [])
        persons = entry.get("persons", [])

        if not vehicles:
            continue  # skip entries with no vehicle — no proximity to compute

        for v in vehicles:
            v_bbox = v["bbox_norm"]
            v_class = v["class"]
            v_cls_id = VEHICLE_CLASS_MAP.get(v_class.lower(), 1)

            # Real worker–vehicle pairs
            for p in persons[:max_pairs_per_image]:
                w_bbox = p["bbox_norm"]
                feat = extract_proximity_feature(w_bbox, v_bbox, v_class)
                prox_class, risk = derive_proximity_label(w_bbox, v_bbox)
                samples.append({
                    "features": feat,
                    "proximity_class": prox_class,
                    "risk_score": risk,
                    "vehicle_class_id": v_cls_id,
                })

            # Synthesize virtual workers for images with no person annotations
            if not persons or augment:
                n_synth = 4 if not persons else 2
                synth_workers = _synthesize_workers(v_bbox, n_synth)
                for w_bbox in synth_workers:
                    feat = extract_proximity_feature(w_bbox, v_bbox, v_class)
                    prox_class, risk = derive_proximity_label(w_bbox, v_bbox)
                    samples.append({
                        "features": feat,
                        "proximity_class": prox_class,
                        "risk_score": risk,
                        "vehicle_class_id": v_cls_id,
                    })

    # Hard negatives: explicitly create DANGER zone examples
    if augment and samples:
        n_hard = int(len(samples) * hard_negative_ratio)
        vehicle_entries = [e for e in manifest.get("entries", []) if e.get("vehicles")]
        for _ in range(n_hard):
            if not vehicle_entries:
                break
            entry = random.choice(vehicle_entries)
            v = random.choice(entry["vehicles"])
            v_bbox = v["bbox_norm"]
            v_class = v["class"]
            v_cls_id = VEHICLE_CLASS_MAP.get(v_class.lower(), 1)
            # Synthesize danger-zone worker
            w_bbox = _synthesize_danger_worker(v_bbox)
            feat = extract_proximity_feature(w_bbox, v_bbox, v_class)
            prox_class, risk = derive_proximity_label(w_bbox, v_bbox)
            samples.append({
                "features": feat,
                "proximity_class": prox_class,
                "risk_score": risk,
                "vehicle_class_id": v_cls_id,
            })

    # Augmentation: feature-level noise passes
    if augment:
        samples = _augment_samples(samples)

    return samples


def _synthesize_workers(v_bbox: List[float], n: int) -> List[List[float]]:
    """Synthesize n virtual worker bboxes covering all proximity zones."""
    vcx, vcy = v_bbox[0], v_bbox[1]
    workers = []
    # Sample from different zones
    distances = [
        random.uniform(0.02, DANGER_THRESH),                       # DANGER
        random.uniform(DANGER_THRESH, SUPERVISED_THRESH),           # SUPERVISED
        random.uniform(SUPERVISED_THRESH, TOO_FAR_THRESH),          # SAFE
        random.uniform(TOO_FAR_THRESH, min(1.0, TOO_FAR_THRESH + 0.3)),  # TOO_FAR
    ]
    for i in range(n):
        d = distances[i % len(distances)]
        angle = random.uniform(0, 2 * math.pi)
        wx = max(0.02, min(0.98, vcx + d * math.cos(angle)))
        wy = max(0.02, min(0.98, vcy + d * math.sin(angle)))
        ww = random.uniform(0.02, 0.07)
        wh = random.uniform(0.04, 0.12)
        workers.append([wx, wy, ww, wh])
    return workers


def _synthesize_danger_worker(v_bbox: List[float]) -> List[float]:
    """Synthesize a worker specifically in the danger zone."""
    vcx, vcy = v_bbox[0], v_bbox[1]
    d = random.uniform(0.0, DANGER_THRESH * 0.9)
    angle = random.uniform(0, 2 * math.pi)
    wx = max(0.02, min(0.98, vcx + d * math.cos(angle)))
    wy = max(0.02, min(0.98, vcy + d * math.sin(angle)))
    ww = random.uniform(0.02, 0.06)
    wh = random.uniform(0.04, 0.10)
    return [wx, wy, ww, wh]


def _augment_samples(
    samples: List[Dict[str, Any]],
    noise_std: float = 0.015,
    n_extra_passes: int = 1,
) -> List[Dict[str, Any]]:
    """Add augmented copies of samples with small spatial jitter."""
    augmented = list(samples)
    for _ in range(n_extra_passes):
        for s in samples:
            feats = list(s["features"])
            # Jitter worker and vehicle positions (first 8 dims)
            for i in range(8):
                feats[i] = max(0.0, min(1.0, feats[i] + random.gauss(0, noise_std)))
            # Re-derive geometric features from jittered positions
            w_bbox = feats[0:4]
            v_bbox = feats[4:8]
            v_cls = "tractor" if s["vehicle_class_id"] == 0 else "truck"
            new_feat = extract_proximity_feature(w_bbox, v_bbox, v_cls)
            prox_class, risk = derive_proximity_label(w_bbox, v_bbox)
            augmented.append({
                "features": new_feat,
                "proximity_class": prox_class,
                "risk_score": risk,
                "vehicle_class_id": s["vehicle_class_id"],
            })
    return augmented


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

FEATURE_DIM = 20


def samples_to_tensors(
    samples: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sample list to PyTorch tensors.

    Returns:
        features: [N, 20] float tensor
        prox_class: [N] long tensor
        risk_score: [N] float tensor
        vehicle_class_id: [N] float tensor
    """
    feats = torch.tensor([s["features"] for s in samples], dtype=torch.float32)
    prox  = torch.tensor([s["proximity_class"] for s in samples], dtype=torch.long)
    risk  = torch.tensor([s["risk_score"] for s in samples], dtype=torch.float32)
    v_cls = torch.tensor([s["vehicle_class_id"] for s in samples], dtype=torch.float32)
    return feats, prox, risk, v_cls
