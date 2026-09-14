# HazardMesh Execution Instructions

This document provides step-by-step instructions to reproduce the entire HazardMesh pipeline, run tests, and launch the interactive demo.

## Prerequisites
* Python 3.10+
* Windows / Linux / macOS
* GPU recommended but CPU fully supported

## 1. Setup Environment
```bash
pip install -r requirements.txt
```

## 2. Generate Dataset & Ground Truth
Generates 12 realistic construction video clips and strictly split ground truth annotations (70% train, 15% val, 15% holdout):
```bash
python -m data.generator
```

## 3. Perception Stage (YOLOv8)
Runs YOLOv8 detector and PPE inspection on all video clips:
```bash
python scripts/run_perception.py
```
*Output*: `data/detections/*_detections.json`

## 4. Multi-Object Tracking
Maintains persistent worker and machinery identities across frames:
```bash
python scripts/run_tracking.py
```
*Output*: `data/tracks/*_tracks.json`

## 5. Spatio-Temporal Feature Engineering
Computes 19 numeric tabular features (PPE, proximity, duration, density, confidence):
```bash
python scripts/build_features.py
```
*Output*: `data/features/train_features.json`, `val_features.json`, `holdout_features.json`

## 6. Train V3 Learned PyTorch Model
Trains the dual-head Multi-Layer Perceptron (`RiskMLP`) with class-weighted Cross-Entropy and continuous MSE risk loss:
```bash
python scripts/train_model.py --version v3 --epochs 70 --lr 0.005
```
*Output*: `models/v3/hazard_risk_mlp.pt`

## 7. Run Evaluation on Held-Out Test Split
Evaluates V0, V1, V2, and V3 on the exact same holdout split:
```bash
python scripts/evaluate_model.py --version all
```
*Output*: `reports/v0_baseline_eval.json`, `v1_context_eval.json`, `v2_temporal_eval.json`, `v3_learned_eval.json`

## 8. Failure Analysis & Clustering
Diagnoses misclassifications and calculates error reduction across iterations:
```bash
python scripts/analyze_failures.py
```
*Output*: `reports/failure_analysis_report.json`

## 9. Generate Benchmark Improvement Chart
Generates benchmark comparison table and visualization:
```bash
python scripts/generate_improvement_chart.py
```
*Output*: `reports/improvement_chart.png`, `reports/improvement_chart.json`

## 10. Run Automated Unit Tests
```bash
python -m unittest discover tests
```

## 11. Launch Web Demo
```bash
python scripts/run_demo.py --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.
