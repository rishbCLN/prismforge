# HazardMesh — Context-Aware Construction Safety Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Perception-00FFFF.svg)](https://ultralytics.com)
[![PRISM Experiment Lineage](https://img.shields.io/badge/PRISM-Trace%20Verified-green.svg)]()

> **HazardMesh** is a context-aware construction-site safety intelligence platform. Instead of stopping at simplistic `"detect PPE violation → alert"` triggers, HazardMesh converts raw visual perceptions into structured spatio-temporal features and feeds them into a custom hazard-risk model to estimate **continuous risk score**, **hazard severity**, and **intervention priority**.

---

## 1. Core Architecture

HazardMesh follows a multi-tier pipeline designed for structured risk reasoning:

```text
                    ┌────────────────────┐
                    │     Video Input    │ (MP4/AVI/Webcam)
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │      YOLOv8        │ (Person, Machinery, PPE)
                    │    Perception      │
                    └─────────┬──────────┘
                              ↓
                ┌─────────────────────────────┐
                │ Detection / Tracking Layer  │ (Persistent Track IDs)
                └─────────────┬───────────────┘
                              ↓
                ┌─────────────────────────────┐
                │  Structured Scene Features  │
                │                             │
                │ • PPE status (helmet, vest) │
                │ • Spatial proximity proxy   │
                │ • Temporal persistence      │
                │ • Scene density & counts    │
                │ • Detector confidence       │
                └─────────────┬───────────────┘
                              ↓
                ┌─────────────────────────────┐
                │ Hazard Feature Processor    │
                └─────────────┬───────────────┘
                              ↓
                ┌─────────────────────────────┐
                │ Custom Risk Model (PyTorch) │
                └─────────────┬───────────────┘
                              ↓
                ┌─────────────────────────────┐
                │ Risk Score + Severity       │
                │ + Intervention Priority     │
                └─────────────┬───────────────┘
                              ↓
                ┌─────────────────────────────┐
                │  Evaluation / PRISM Layer   │
                └─────────────┬───────────────┘
                              ↓
                ┌─────────────────────────────┐
                │ Failure-Driven Improvement  │ (Clusters A, B, C, D)
                └─────────────────────────────┘
```

---

## 2. Iterative Model Progression (V0 → V3)

Every model version was evaluated against the **exact same held-out test split** (366 frames, strictly partitioned by clip to guarantee zero data leakage).

| Version | Macro F1 | High-Risk Recall | False Positive Rate | Weighted Hazard Error | Key Improvement |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **V0 Baseline** | 0.3053 | 0.4286 | 0.0000 | 1.1639 | Naive violation presence |
| **V1 Context** | 0.4367 | 0.4286 | 0.0000 | 0.8962 | Machinery proximity & PPE weights |
| **V2 Temporal** | 0.3541 | 0.3482 | 0.0000 | 1.3347 | Persistence gating & duration filtering |
| **V3 Learned (PyTorch)** | **0.7440** | **1.0000** | **0.0000** | **0.0164** | Non-linear compound risk reasoning |

* **Macro F1 Score**: Jumped from `0.3053` (V0) to `0.7440` (V3).
* **High-Risk Recall**: Reached `100.0%` on critical danger zones.
* **Weighted Hazard Error**: Asymmetric safety penalty dropped from `1.1639` to `0.0164` (98.6% reduction).

---

## 3. Failure-Driven Improvement Loop

HazardMesh does not introduce arbitrary versions. Each version directly addresses failure modes discovered in earlier models:

1. **V0 → V1**:
   * *Diagnosis*: V0 treats every missing PPE as an immediate emergency, even if the worker is safely walking in a designated green corridor far from machinery.
   * *Intervention*: Added normalized 2D camera-plane machinery proximity and danger radius calculations.
2. **V1 → V2**:
   * *Diagnosis*: V1 triggers false alarms on transient passersby (e.g. cutting across the perimeter for 0.3s) and detector flicker.
   * *Intervention*: Added temporal persistence gating (minimum consecutive violating frames and duration scaling).
3. **V2 → V3**:
   * *Diagnosis*: Linear heuristic weights fail to capture non-linear hazard compounding when multiple workers and machines interact simultaneously.
   * *Intervention*: Trained a PyTorch Multi-Layer Perceptron (`RiskMLP`) with dual-head loss and input gradient attribution.

---

## 4. Quick Start & CLI Pipeline

### Installation
```bash
git clone https://github.com/your-org/hazardmesh.git
cd hazardmesh
pip install -r requirements.txt
```

### Reproduce Full Pipeline
```bash
# 1. Generate synthetic construction scenarios with strict holdout splits
python -m data.generator

# 2. Run YOLOv8 perception
python scripts/run_perception.py

# 3. Track workers across frames
python scripts/run_tracking.py

# 4. Extract structured spatio-temporal features
python scripts/build_features.py

# 5. Train V3 PyTorch risk model
python scripts/train_model.py --version v3

# 6. Evaluate all versions on held-out test split
python scripts/evaluate_model.py --version all

# 7. Cluster failures and generate comparative diagnostic report
python scripts/analyze_failures.py

# 8. Generate automated improvement chart
python scripts/generate_improvement_chart.py

# 9. Launch interactive web demo
python scripts/run_demo.py --port 8000
```
Open **http://127.0.0.1:8000** in your browser to interact with the demo.

---

## 5. PRISM Experiment Tracking

All evaluation runs are recorded into `experiments/prism_runs.json` with:
* `run_id`: Unique identifier
* `model_version`: V0, V1, V2, or V3
* `git_commit`: Active git commit hash
* `hyperparameters`: Architecture, learning rates, loss functions
* `metrics`: Complete standard & safety-specific metrics
* `failure_summary`: Cluster distribution and error counts

---

## 6. Safety & Engineering Claims Disclaimer

* **Visual Proximity Proxy**: 2D normalized bounding-box distance is explicitly labeled as a camera-plane spatial proxy, not calibrated 3D physical distance.
* **Decision Support**: HazardMesh provides hazard-risk estimation and intervention prioritization for decision support; it does not guarantee accident prevention or replace certified human safety officers.
