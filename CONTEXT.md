# PRISMForge — Physical Operations Video Intelligence Platform
## Unified Context & Master Architecture: Job-Site Safety & Warehouse Damage Prevention

---

## 1. Executive Summary & Vision

Modern industrial enterprises—from **construction job-sites** to **high-throughput logistics warehouses**—suffer immense financial losses, operational delays, and human injury due to a single systemic failure: **reliance on passive CCTV surveillance**.

Traditional camera systems only answer: *"What broke, who got hurt, and who was to blame after the incident occurred?"*

```
[Traditional CCTV]
Camera ──▶ Recording ──▶ Video Storage ──▶ Post-Incident Review ──▶ Damage / Loss Incurred
```

**PRISMForge** transforms commodity camera feeds into a **Proactive Field Intelligence Assistant**. Instead of recording damage or accidents after they occur, PRISMForge tracks human-object interactions in real time, computes spatio-temporal kinematic features, evaluates multi-tier risk, and triggers immediate interventions:

```
[PRISMForge Field Intelligence]
Camera ──▶ Perception ──▶ Tracking ──▶ Kinematic Reasoning ──▶ Risk Scoring ──▶ Alert / Intervention ──▶ Operational Learning
```

### The Unified Dual-Domain Platform
PRISMForge operates as a unified platform serving two specialized physical domains:
1. **DamageMesh (Warehouse Material Handling Intelligence)**: Detects damage-causing handling behaviors (dropping, throwing, dragging, rough impact, improper staging) during vehicle loading and unloading.
2. **HazardMesh (Job-Site Safety Intelligence)**: Detects PPE non-compliance, heavy machinery proximity zones, and prolonged hazardous worker exposures.

Both domains share the exact same underlying **PRISM Core Engine**:
$$\text{Perception (YOLOv8)} \longrightarrow \text{Spatio-Temporal Tracking} \longrightarrow \text{Feature Engineering} \longrightarrow \text{Risk Models (V0}\to\text{V3)} \longrightarrow \text{PRISM Audit Ledger}$$

---

## 2. Core Operational Philosophy: Prevention Over Detection

PRISMForge reframes physical operations around **proactive damage and accident prevention**:
- Instead of reporting: *"15 cartons were damaged during morning unloading."*
- PRISMForge reports: *"Detected 28 high-risk handling events (18 rough impacts, 7 dragging, 3 drops) and alerted supervisors in time to prevent an estimated \$4,200 in product replacement."*

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│     OBSERVED BEHAVIOR     │ ───▶ │       RISK RATING         │ ───▶ │    PREVENTIVE ACTION      │
│ Operator drops heavy box  │      │ High Risk (Impact Score)  │      │ Alert: Inspect & Restage  │
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

---

## 3. PRISMForge End-to-End Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │            RAW VIDEO INPUT STREAMS           │
                    │ (CCTV, RTSP, Smartphone Videos, Google Drive)│
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │          1. INGESTION & NORMALIZATION        │
                    │  - Standardize 640x360 @ 25 FPS              │
                    │  - Automated Chunker (4-second segments)     │
                    │  - Kaggle & Google Drive Connector           │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │          2. MULTI-CLASS PERCEPTION           │
                    │  - YOLOv8 Object & Operator Detection        │
                    │  - Entities: Workers, Boxes, Pallets, Trolley│
                    │  - Safety Gear: Helmets, High-Vis Vests      │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      3. SPATIO-TEMPORAL ENTITY TRACKER       │
                    │  - Persistent Object & Human Identifiers     │
                    │  - Centroid Trajectories & Velocities        │
                    │  - Multi-Entity Spatial Association          │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      4. KINEMATIC & BEHAVIOR ENGINE          │
                    │  - Vertical Drop Velocity & Impact Decel     │
                    │  - Ballistic Horizontal Throwing Vectors     │
                    │  - Floor Friction / Dragging Indicator       │
                    │  - Acceleration Variance (Jerk / Roughness)  │
                    │  - Zone Containment & Geofencing Breach      │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │         5. MULTI-TIER RISK REASONING         │
                    │  - V0: Static Heuristic Baseline             │
                    │  - V1: Spatial Context & Proximity Model     │
                    │  - V2: Temporal Persistence Filter           │
                    │  - V3: Dual-Head Learned Neural Risk MLP     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │          6. INTERVENTION & AUDIT LOG         │
                    │  - PRISM Verifiable Run Ledger (JSON)        │
                    │  - Real-time Visual Alerting (P1, P2, P3)    │
                    │  - Interactive Replay & Frame Inspection     │
                    │  - AI Supervisor Incident Q&A Assistant      │
                    └──────────────────────────────────────────────┘
```

---

## 4. Warehouse Material Handling Domain (DamageMesh)

### 4.1 Target Process: Loading & Unloading Bays
- **Operational Setting**: Loading dock, truck bed / container interior, staging area.
- **Key Equipment**: Manual pallet jacks, trolleys, conveyor rollers, pallets.
- **Key Entities**: Warehouse operators, cartons/packages, plastic totes, pallets, vehicle tailgate.

### 4.2 The 5 Core 36-Hour Feasible Behaviors
Based on hackathon feasibility, model reliability, and high business impact, the system focuses on **5 core behaviors**:

```
┌─────────────────────────┬──────────────┬─────────────┬───────────────────────────┐
│ Behavior                │ Feasibility  │ Difficulty  │ Recommendation & Scope    │
├─────────────────────────┼──────────────┼─────────────┼───────────────────────────┤
│ 1. Dropping             │ ⭐⭐⭐⭐⭐      │ Low-Medium  │ INCLUDED (High Impact)    │
│ 2. Throwing             │ ⭐⭐⭐⭐⭐      │ Low-Medium  │ INCLUDED (Critical Risk)  │
│ 3. Dragging             │ ⭐⭐⭐⭐       │ Medium      │ INCLUDED (High Wear)      │
│ 4. Rough / Fast Motion  │ ⭐⭐⭐⭐       │ Medium      │ INCLUDED (Shock / Jerk)   │
│ 5. Outside Staging Zone │ ⭐⭐⭐⭐⭐      │ Low         │ INCLUDED (Geofence Breach)│
├─────────────────────────┼──────────────┼─────────────┼───────────────────────────┤
│ 6. Improper Stacking    │ ⭐⭐⭐        │ Medium-High │ Optional / Phase 2        │
│ 7. Unstable Stacking    │ ⭐⭐         │ High        │ Deferred                  │
│ 8. Pallet Stability     │ ⭐⭐         │ High        │ Deferred                  │
│ 9. Product Orientation  │ ⭐⭐         │ High        │ Deferred                  │
│ 10. Forklift Crossings  │ ⭐⭐         │ High        │ Borrowed from HazardMesh  │
└─────────────────────────┴──────────────┴─────────────┴───────────────────────────┘
```

---

## 5. Kinematic & Mathematical Behavior Formulations

PRISMForge computes behavior detections deterministically using physical trajectories over sliding time windows $\Delta t = [t - k, t]$:

### 1. Dropping (Free-Fall & Sudden Impact Deceleration)
A carton changes state from being held to downward gravitational acceleration, followed by immediate zero velocity upon hitting the floor:
$$v_y(t) = \frac{y_{\text{box}}(t) - y_{\text{box}}(t-\Delta t)}{\Delta t}$$
$$\text{Drop Event} \iff \left( v_y(t) > \theta_{\text{drop\_vel}} \right) \;\land\; \left( |v(t+\Delta t)| \approx 0 \right) \;\land\; \left( \|\mathbf{p}_{\text{worker}} - \mathbf{p}_{\text{box}}\| > \theta_{\text{separation}} \right)$$

### 2. Throwing (Ballistic Trajectory & Rapid Separation)
A package moves with high horizontal velocity without physical support or carrier contact:
$$v_x(t) = \frac{|x_{\text{box}}(t) - x_{\text{box}}(t-\Delta t)|}{\Delta t}$$
$$\text{Throw Event} \iff \left( v_x(t) > \theta_{\text{throw\_vel}} \right) \;\land\; \left( \frac{d}{dt} \|\mathbf{p}_{\text{worker}} - \mathbf{p}_{\text{box}}\| > \theta_{\text{sep\_rate}} \right)$$

### 3. Dragging (Ground Friction Without Mechanical Carrier)
A carton translates laterally along the floor plane while staying in contact with the ground, in the absence of a trolley/pallet bounding box overlap:
$$\text{Drag Event} \iff \left( y_{\text{box}}(t) \ge y_{\text{floor\_min}} \right) \;\land\; \left( |v_x(t)| > \theta_{\text{drag\_vel}} \right) \;\land\; \left( \text{IoU}(\text{Box}, \text{Trolley}) = 0 \right)$$

### 4. Rough Handling / Kinetic Jerk
Excessive acceleration variance (jerk) applied to fragile merchandise during manual transfer:
$$j(t) = \frac{a(t) - a(t-\Delta t)}{\Delta t} = \frac{d^3 \mathbf{p}}{dt^3}$$
$$\text{Rough Handling Event} \iff |j(t)| > \theta_{\text{jerk\_threshold}}$$

### 5. Staging Zone Breach (Outside Designated Boundary)
Cartons or pallets placed outside designated yellow staging lines or vehicle alignment bounds:
$$\text{Zone Breach Event} \iff \mathbf{p}_{\text{box}}(t) \notin \mathcal{P}_{\text{staging\_polygon}}$$

---

## 6. Multi-Tier Risk Modeling & Progressive Evaluation

PRISMForge implements 4 distinct model iterations to ensure benchmark-grade progression:

| Model Tier | Name | Core Architecture | Failure Mode Addressed |
| :--- | :--- | :--- | :--- |
| **V0** | Baseline Heuristic | Threshold on raw violation flags | Ignores speed, distance, and context; high false alarm rate. |
| **V1** | Spatial Context Model | Proximity-weighted kinematic scoring | Accounts for distance to floor, worker, and dock boundary. |
| **V2** | Temporal Persistence Filter | Sliding-window trajectory smoothing | Filters transient camera jitter, occlusions, and brief re-grips. |
| **V3** | Learned Dual-Head Neural MLP | PyTorch MLP with risk score and intervention priority heads | Solves non-linear feature interactions with gradient attribution. |

### Severity Categories & Intervention Priorities
- **Risk Score**: Normalized $[0.0, 1.0]$.
- **Severity**: `LOW` ($<0.30$), `MEDIUM` ($0.30 - 0.65$), `HIGH` ($0.65 - 0.85$), `CRITICAL` ($>0.85$).
- **Intervention Priority**:
  - `P1_IMMEDIATE`: Halt movement, inspect carton immediately for structural/leak damage.
  - `P2_CORRECTIVE`: Restage package with mechanical handling equipment (trolley/pallet).
  - `P3_INFORMATIONAL`: Log warning to shift summary for operator feedback.

---

## 7. AI Warehouse Supervisor Assistant

The platform integrates a conversational operational assistant grounded strictly in verified PRISM event logs (no hallucinated telemetry):

### Example Supervisor Queries & Grounded Answers:
- **Query**: *"Show me all high-risk handling events from today's unloading."*
  - **Assistant**: *"Bay 3 recorded 4 high-risk events between 10:15 AM and 11:30 AM: 2 carton drops (drop velocity > 1.8 m/s, stationary impact) and 2 ground dragging events across 4.2 meters without trolley support."*
- **Query**: *"Why was event #104 classified as High Risk?"*
  - **Assistant**: *"Event #104 at 02:41 involved Carton_ID_14 dropped from approximately 1.1 meters with an estimated impact velocity of 2.1 m/s, followed by zero displacement. The package was classified as Fragile Goods."*
- **Query**: *"Which operator or bay needs coaching on material handling?"*
  - **Assistant**: *"Dock Bay 2 recorded 73% of dragging incidents this week. Shift observations indicate workers drag cartons over 50 lbs due to missing trolley availability at that bay."*

---

## 8. Responsible AI, Privacy & Non-Punitive Deployment

```
   ┌────────────────────────────────────────────────────────┐
   │         RESPONSIBLE AI FOR PHYSICAL OPERATIONS         │
   ├────────────────────────────────────────────────────────┤
   │ 1. Process Improvement over Indiscriminate Surveillance │
   │ 2. Edge Processing & Local Video Processing            │
   │ 3. Optional Face & Identity Blurring                   │
   │ 4. Verifiable Evidence Logs (Human-in-the-Loop Review) │
   │ 5. Asymmetric Loss Tuning (Penalize False Alarms)      │
   └────────────────────────────────────────────────────────┘
```

1. **Focus on Behavior, Not People**: The AI assesses kinematic vectors (velocity, distance, floor contact) of boxes and objects, not employee personal attributes.
2. **Privacy by Design**: Works on low-resolution feeds (640x360); faces can be blurred locally on ingestion.
3. **Coaching & Root-Cause Resolution**: Alerts are framed around equipment bottlenecks (e.g. *"Provide more trolleys at Bay 2"*) rather than punitive operator monitoring.

---

## 9. Pilot Environment & Submission Roadmap

### 9.1 Data Sources
- **Official Pilot Videos**: [Google Drive Folder](https://drive.google.com/drive/folders/1MG90LJowfSZ2qz5woDyarHdzskbCRLzP?usp=sharing)
- **Public & Synthetic Benchmarks**: Kaggle Warehouse Safety & Logistics Video Datasets
- **Controlled Mock Setup**: Controlled test video clips of box drops, dragging, and throwing.

### 9.2 Presentation Deck Structure (5 Slides)
- **Slide 1: Solution & Value Proposition** — *PRISMForge: AI Field Intelligence for Damage-Free Warehouse Logistics*.
- **Slide 2: Problem & User Journey** — *Traditional CCTV vs. Proactive Damage Prevention Journey*.
- **Slide 3: Technical Architecture** — *Perception $\to$ Tracking $\to$ Kinematics $\to$ Learned Dual-Head Risk MLP*.
- **Slide 4: Live Prototype & Replay Demo** — *Screenshots, event replay, risk telemetry HUD, and AI supervisor Q&A*.
- **Slide 5: Business Impact & Operational Metrics** — *ROI of prevented damages, reduced claims, supervisor coaching adoption*.

---

## 10. Repository Implementation Mapping

| Component | Code Location in Repository | Role in Unified System |
| :--- | :--- | :--- |
| **Ingestion Engine** | [`src/ingestion/pipeline.py`](file:///d:/prismforge/src/ingestion/pipeline.py) | Ingests, normalizes, and chunks warehouse videos and jobsite feeds. |
| **Perception Module** | [`src/perception/detector.py`](file:///d:/prismforge/src/perception/detector.py) | YOLOv8 detector for workers, cartons, pallets, and equipment. |
| **Object Tracker** | [`src/tracking/tracker.py`](file:///d:/prismforge/src/tracking/tracker.py) | Tracks worker-box spatial relationships, velocities, trajectories. |
| **Kinematic Features** | [`src/features/feature_extractor.py`](file:///d:/prismforge/src/features/feature_extractor.py) | Computes drop velocity, ballistic throw, drag duration, jerk. |
| **Risk Reasoning** | [`src/risk_model/`](file:///d:/prismforge/src/risk_model/) | V0 Heuristic $\to$ V1 Context $\to$ V2 Temporal $\to$ V3 Learned MLP. |
| **PRISM Audit Ledger**| [`src/prism/tracker.py`](file:///d:/prismforge/src/prism/tracker.py) | Immutable JSON log of timestamped incidents and risk metrics. |
| **Field Assistant UI** | [`web/index.html`](file:///d:/prismforge/web/index.html), [`web/server.py`](file:///d:/prismforge/web/server.py) | Live video reasoning stream, risk HUD, supervisor Q&A panel. |

---
*PRISMForge — Transforming passive surveillance into proactive operational intelligence.*
