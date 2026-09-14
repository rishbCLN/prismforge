# VigiAI — Warehouse Material-Handling Intelligence Module

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Perception-00FFFF.svg)](https://ultralytics.com)
[![Pydantic Contract](https://img.shields.io/badge/Schema-Pydantic%20Contract-orange.svg)](https://docs.pydantic.dev/)
[![Hardware](https://img.shields.io/badge/Compute-CPU%20%2F%20CUDA-purple.svg)]()

> **"Proactive Video Intelligence for Safer Warehouse Operations"**  
> *Tagline: "See the Risk. Understand the Cause. Prevent the Damage."*

---

## 1. Executive Summary & Problem Context

Traditional warehouse CCTV is passive: it records footage that is only reviewed *after* goods arrive crushed, packages are reported missing, or disputes occur between carriers and distribution centers. Traditional video cameras do not understand handling kinematics, identify potential product damage risks early enough for intervention, or alert warehouse supervisors in real time.

**VigiAI** transforms warehouse video streams into proactive operational intelligence:

```text
Video Input → Perception → Tracking → Behaviour Understanding → Risk Reasoning → Incident → Intervention
```

### Proactive Decision Support vs. Damage Assumption
VigiAI operates strictly on observable behaviour and physics-informed spatio-temporal cues:
- **Observed Behaviour**: Kinematic trajectory, acceleration, velocity shock, floor proximity, zone violations.
- **Potential Risk**: Continuous risk score $[0, 1]$, categorized severity (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and priority (`P1_IMMEDIATE`, `P2_CORRECTIVE`, `P3_INFORMATIONAL`).
- **Recommended Intervention**: Actionable guidance for dock managers (e.g., immediate package inspection, ergonomic operator coaching).

> [!NOTE]  
> VigiAI does **not** claim that actual product damage definitely occurred inside an opaque box. It detects **"Potential damage-causing behaviour"** and flags **"High-risk handling events"** for inspection.

---

## 2. Scope & Architectural Boundaries

VigiAI is developed by a two-person hackathon team with strict module isolation:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           VigiAI SYSTEM                                 │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │   WAREHOUSE MATERIAL-HANDLING MODULE (THIS COMPONENT)           │   │
│   │                                                                 │   │
│   │   • Video Ingestion (MP4/AVI)                                   │   │
│   │   • YOLOv8 Perception (Worker, Carton, Pallet, Trolley)        │   │
│   │   • Lightweight Spatio-Temporal Tracking (ByteTrack / Sort)    │   │
│   │   • Sliding-Window Track History & Coordinate Normalization     │   │
│   │   • Kinematic Feature Extraction (Velocity, Jerk, Proximity)    │   │
│   │   • Warehouse Behaviour Engine (DROP, THROW, DRAG, etc.)        │   │
│   │   • Explainable Risk Scorer & Severity Matrix                   │   │
│   │   • Structured Evidence Extraction (Frames & Clips)             │   │
│   │   • Stable Machine-Readable Event Stream Contract (JSON)        │   │
│   └────────────────────────────────┬────────────────────────────────┘   │
│                                    │ Machine-Readable Event Contract    │
│                                    ▼ (JSON / REST API)                  │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │   TEAMMATE SUBSYSTEMS (EXTERNAL TO THIS MODULE)                 │   │
│   │                                                                 │   │
│   │   • Unified Team Dashboard & Web UI                             │   │
│   │   • Worker / Construction Safety Module                         │   │
│   │   • Worker PPE Monitoring (Hardhats, Safety Vests)              │   │
│   │   • Forklift / Pedestrian Site Monitoring                       │   │
│   │   • Shared Application Shell & Notification Center              │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. High-Level Architecture

The core pipeline transforms raw pixel frames into structured operational incidents through deterministic, explainable stages:

```text
                  ┌──────────────────────────────┐
                  │         Video Input          │ (MP4 / AVI / Stream)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │       Video Ingestion        │ (frame_id, timestamp, FPS, WxH)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │      YOLOv8 Perception       │ (Person, Carton, Pallet, Trolley)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │       Object Tracking        │ (Persistent Track IDs)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │        Track History         │ (Sliding Temporal Window)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   Kinematic & Spatial Features│ (Velocity, Jerk, Separation, Floor Proxy)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   Warehouse Behaviour Engine │ (State Machines: DROP, THROW, DRAG, ...)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   Explainable Risk Scorer    │ (Score [0,1], Severity, Contributors)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │     Evidence Extractor       │ (Pre/Post Frames & Incident Clips)
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   Structured Event Output    │ (Stable JSON/JSONL Event Contract)
                  └──────────────────────────────┘
```

---

## 4. The 5 Core Warehouse Behaviours

VigiAI detects 5 high-impact material-handling behaviours through multi-frame temporal reasoning rather than single-frame classification:

| Behaviour | Physical Reasoning & Sequence Pattern | Trigger Conditions |
| :--- | :--- | :--- |
| **`DROP`** | Carton near worker $\rightarrow$ rapid downward acceleration $\rightarrow$ worker-product separation $\rightarrow$ floor proximity $\rightarrow$ sudden stationary stop. | $v_y > v_{\text{drop\_thresh}}$, separation $> d_{\text{thresh}}$, $y \ge y_{\text{floor}}$, sudden deceleration. |
| **`THROW`** | Rapid horizontal ballistic motion $\rightarrow$ rapid operator separation $\rightarrow$ flight trajectory without mechanical support. | $v_x > v_{\text{throw\_thresh}}$, $\frac{\Delta d_{\text{op}}}{\Delta t} > s_{\text{thresh}}$, no pallet/trolley IoU overlap. |
| **`DRAG`** | Product in contact with floor $\rightarrow$ lateral displacement persists across time $\rightarrow$ absence of trolley or pallet support. | $y \approx y_{\text{floor}}$, $v_x > v_{\text{drag\_thresh}}$, duration $\ge N$ frames, trolley IoU $< 0.1$. |
| **`ROUGH_HANDLING`** | Erratic motion, severe kinematic jerk, violent shaking, or sudden impact slamming. | Jerk $|\frac{d^3 x}{dt^3}| > j_{\text{thresh}}$, velocity variance $\sigma_v^2 > \text{thresh}$. |
| **`ZONE_BREACH`** | Package or material staged outside designated yellow/green perimeter in traffic corridors. | Point-in-polygon test: Carton centroid $(x_{\text{norm}}, y_{\text{norm}}) \notin \text{Polygon}_{\text{staging}}$. |

---

## 5. Machine-Readable Event Contract

Downstream dashboards and consumer applications ingest a stable, strictly-typed JSON event contract. Each detected incident outputs:

```json
{
  "event_id": "EVT_20260914_1068_DROP",
  "timestamp": 42.70,
  "frame_id": 1068,
  "track_id": 7,
  "object_type": "carton",
  "behaviour": "DROP",
  "confidence": 0.91,
  "risk_score": 0.87,
  "severity": "HIGH",
  "priority": "P1_IMMEDIATE",
  "features": {
    "vertical_velocity": 0.62,
    "horizontal_velocity": 0.08,
    "worker_separation": 0.35,
    "floor_proximity": 0.95,
    "impact_deceleration": 0.84,
    "persistence_frames": 14
  },
  "contributors": [
    "High downward velocity exceeding safety threshold",
    "Rapid worker-product physical separation",
    "Sudden deceleration impact near concrete floor region",
    "Product stationary post-impact"
  ],
  "evidence": {
    "trigger_frame_path": "outputs/evidence/EVT_1068_trigger.jpg",
    "buffer_frames": [
      "outputs/evidence/EVT_1068_pre_10.jpg",
      "outputs/evidence/EVT_1068_post_10.jpg"
    ],
    "clip_path": "outputs/evidence/EVT_1068_incident.mp4"
  }
}
```

---

## 6. Risk Scoring & Explainability

The Risk Engine maps behavioural kinematics into actionable risk tiers without being an uninterpretable black box:

### Risk Tiers & SLA Matrix
* **`LOW` ($0.00 - 0.29$)** $\rightarrow$ **`P3_INFORMATIONAL`**: Safe compliant handling, controlled palletizing.
* **`MEDIUM` ($0.30 - 0.64$)** $\rightarrow$ **`P2_CORRECTIVE`**: Staging zone breach, minor rough placement.
* **`HIGH` ($0.65 - 0.84$)** $\rightarrow$ **`P2_CORRECTIVE` / `P1_IMMEDIATE`**: Prolonged dragging, heavy drop from waist height.
* **`CRITICAL` ($0.85 - 1.00$)** $\rightarrow$ **`P1_IMMEDIATE`**: High-velocity throw, ballistic impact, fragile carton drop.

### Explainable Contributing Factors
Every risk score includes a human-readable list of `contributors` so supervisors immediately understand *why* an event was flagged without analyzing raw numbers.

---

## 7. Project Structure

```text
prismforge/
├── configs/
│   └── warehouse.yaml              # Externalized thresholds, zone polygons, weights
│
├── data/
│   ├── raw/                        # Ingested MP4/AVI warehouse video clips
│   ├── processed/                  # Standardized video frames
│   ├── detections/                 # Raw and structured YOLOv8 detections
│   ├── tracks/                     # Persistent spatio-temporal trajectories
│   ├── features/                   # Extracted kinematic and spatial feature sets
│   ├── events/                     # Generated incident events (JSON/JSONL)
│   └── warehouse_generator.py      # Synthetic warehouse clip generator for verification
│
├── src/
│   ├── ingestion/
│   │   └── video.py                # Robust video reader (frame_id, timestamp, FPS)
│   │
│   ├── perception/
│   │   ├── yolo.py                 # Modular YOLOv8 detector & bounding box normalizer
│   │   └── export.py               # Structured detection export utility
│   │
│   ├── tracking/
│   │   └── tracker.py              # Lightweight persistent ID tracker (ByteTrack/SORT)
│   │
│   ├── features/
│   │   ├── motion.py               # Velocity, acceleration, kinematic jerk
│   │   ├── spatial.py              # Normalized proximity, floor proxy, polygon zones
│   │   └── temporal.py             # Sliding window history, persistence ratio
│   │
│   ├── behaviours/
│   │   ├── base.py                 # Abstract state machine behaviour detector
│   │   ├── drop.py                 # Multi-stage drop state machine
│   │   ├── drag.py                 # Floor-contact lateral motion detector
│   │   ├── throw.py                # Ballistic horizontal separation detector
│   │   ├── rough_handling.py       # Kinetic jerk & impact shock detector
│   │   └── zone_breach.py          # Point-in-polygon staging breach detector
│   │
│   ├── risk/
│   │   └── scorer.py               # Heuristic & contextual risk scoring engine
│   │
│   ├── events/
│   │   └── schema.py               # Pydantic / Dataclass event models
│   │
│   ├── evidence/
│   │   └── extractor.py            # Evidence frame buffer and incident clip creator
│   │
│   └── evaluation/
│       └── metrics.py              # Clip-level Precision, Recall, F1, Confusion Matrix
│
├── scripts/
│   ├── run_perception.py           # Phase 2 CLI: run perception on video
│   ├── run_tracking.py             # Phase 4 CLI: run tracking on detections
│   ├── build_features.py           # Phase 6 CLI: compute kinematic features
│   ├── detect_behaviours.py        # Phase 7-11 CLI: run behaviour engines
│   ├── score_risk.py               # Phase 12 CLI: score risks and export
│   └── run_pipeline.py             # End-to-end unified warehouse pipeline
│
├── tests/
│   ├── test_ingestion.py           # Video capture, corruption handling
│   ├── test_perception.py          # Detection structure and normalization
│   ├── test_tracking.py            # Persistent IDs and occlusion recovery
│   ├── test_kinematics.py          # Velocity, acceleration, and jerk calculation
│   ├── test_behaviours.py          # Synthetic trajectory tests for DROP, THROW, etc.
│   └── test_risk_scorer.py         # Severity mapping and score bounds
│
├── outputs/
│   ├── detections.json             # Frame-level perception output
│   ├── tracks.json                 # Persistent trajectories
│   ├── events.json                 # Exported incident stream
│   └── evidence/                   # Incident keyframe snapshots and MP4 clips
│
├── requirements.txt
├── README.md
└── INSTRUCTIONS.md
```

---

## 8. Quick Start & CLI Pipeline

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/rishbCLN/prismforge.git
cd prismforge

# Install core dependencies
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
Run the complete video-to-incident pipeline on a warehouse video:
```bash
python scripts/run_pipeline.py --video data/raw/wh_clip_01_carton_drop.mp4 --config configs/warehouse.yaml
```

**Expected Terminal Output:**
```text
[INFO] Video loaded: wh_clip_01_carton_drop.mp4 (640x360 @ 25 FPS, 100 frames)
[INFO] YOLO perception completed: 184 detections across 100 frames
[INFO] Tracking initialized: 2 persistent tracks identified (Operator #1, Carton #7)
[INFO] Kinematic features computed across 100 frames
[INFO] Behaviour engine running: DROP, THROW, DRAG, ROUGH_HANDLING, ZONE_BREACH
[EVENT] DROP | track=7 | t=1.84s (frame 46) | risk=0.87 | severity=HIGH | priority=P1_IMMEDIATE
[INFO] Evidence saved: outputs/evidence/EVT_wh_clip_01_frame_46.jpg
[INFO] Pipeline complete. Structured events written to outputs/events.json
```

### 3. Step-by-Step Execution (Phased Workflow)
```bash
# Run Perception only (Phase 1-3)
python scripts/run_perception.py --video data/raw/sample.mp4 --out outputs/detections.json

# Run Tracking (Phase 4-5)
python scripts/run_tracking.py --detections outputs/detections.json --out outputs/tracks.json

# Compute Kinematics & Detect Behaviours (Phase 6-11)
python scripts/detect_behaviours.py --tracks outputs/tracks.json --out outputs/events.json
```

---

## 9. Configuration (`configs/warehouse.yaml`)

All detection thresholds, zone polygons, and risk weights are externalized to avoid hardcoded magic numbers:

```yaml
perception:
  model_weights: "yolov8n.pt"
  confidence_threshold: 0.25
  classes:
    - person
    - box
    - pallet
    - trolley

tracking:
  tracker_type: "bytetrack"
  max_lost_frames: 15
  history_window_size: 16

behaviours:
  drop:
    min_downward_velocity: 0.25
    min_worker_separation: 0.20
    floor_proximity_threshold: 0.80
    stationary_frames: 5
  drag:
    floor_proximity_threshold: 0.82
    min_lateral_velocity: 0.18
    persistence_frames: 8
    max_vertical_displacement: 0.05
  throw:
    min_horizontal_velocity: 0.35
    min_separation_rate: 0.20
  rough_handling:
    max_kinetic_jerk: 0.40
    velocity_variance_threshold: 0.30

zones:
  staging_zone:
    name: "DESIGNATED_STAGING_AREA"
    polygon:
      - [0.0625, 0.5000]
      - [0.4062, 0.5000]
      - [0.4062, 0.9167]
      - [0.0625, 0.9167]

risk:
  thresholds:
    low: 0.30
    medium: 0.65
    high: 0.85
```

---

## 10. Development Roadmap & Phasing

To guarantee stability, implementation follows a strict phased order:

- [x] **Phase 0: Environment & Codebase Inspection** (Audit dependencies, model weights, hardware).
- [ ] **Phase 1: Video Input & Ingestion** (Validated reader with frame metadata and error handling).
- [ ] **Phase 2: YOLOv8 Perception** (Perception layer returning structured `Detection` dataclasses).
- [ ] **Phase 3: Structured Detection Export** (Frame-by-frame JSON export contract).
- [ ] **Phase 4: Object Tracking** (ByteTrack persistent ID tracking across consecutive frames).
- [ ] **Phase 5: Track History** (Sliding-window trajectory management).
- [ ] **Phase 6: Kinematic Feature Extraction** (Velocity, acceleration, jerk, floor proxy).
- [ ] **Phase 7: DROP Behaviour Engine** (Controlled validation on drop benchmark clips).
- [ ] **Phase 8: DRAG Behaviour Engine** (Near-floor lateral motion persistence).
- [ ] **Phase 9: THROW Behaviour Engine** (Ballistic separation reasoning).
- [ ] **Phase 10: ROUGH_HANDLING Behaviour Engine** (Kinetic jerk and impact shock).
- [ ] **Phase 11: ZONE_BREACH Behaviour Engine** (Point-in-polygon staging violation).
- [ ] **Phase 12: Risk Scoring & Explainability Engine** (Continuous score, severity, and contributors).
- [ ] **Phase 13: Event Schema & Serialization** (Pydantic / Dataclass contract serialization).
- [ ] **Phase 14: Evidence Extraction** (Keyframe snapshot buffer and incident video clip generation).
- [ ] **Phase 15: Evaluation Utilities** (Clip-level precision, recall, F1, confusion matrix).
- [ ] **Phase 16: Downstream API & Pipeline Integration** (Ready for dashboard consumption).

---

## 11. Engineering Principles & Disclaimers

1. **Deterministic & Explainable**: Physics-based kinematic rules and state machines over opaque black-box deep learning.
2. **Camera-Space Proxy Disclaimer**: 2D normalized bounding-box coordinates represent visual proximity and image-space displacement proxies, not calibrated 3D physical centimeter measurements.
3. **Decision Support**: VigiAI provides operational risk indicators for supervisor intervention and does not replace certified warehouse quality assurance or safety audits.
