"""HazardMesh Interactive Web Demo Server.
FastAPI backend providing video streaming, frame-by-frame perception inspection,
dynamic risk model inference (V0-V3), explainability breakdown, and PRISM experiment traces.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import Dict, Any, Optional, List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.features.feature_extractor import HazardFeatures, FeatureExtractor
from src.risk_model.v0_baseline import V0RuleBaselineModel
from src.risk_model.v1_context import V1ContextRiskModel
from src.risk_model.v2_temporal import V2TemporalRiskModel
from src.risk_model.v3_learned import V3LearnedRiskModel
from src.prism.tracker import PrismTracker

app = FastAPI(title="HazardMesh Safety Intelligence API", version="1.0.0")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount raw videos
videos_dir = os.path.abspath("data/raw")
os.makedirs(videos_dir, exist_ok=True)
app.mount("/videos", StaticFiles(directory=videos_dir), name="videos")

# Mount reports for charts
reports_dir = os.path.abspath("reports")
os.makedirs(reports_dir, exist_ok=True)
app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

# Mount datasets for live test sample previews
datasets_dir = os.path.abspath("datasets")
if os.path.exists(datasets_dir):
    app.mount("/datasets", StaticFiles(directory=datasets_dir), name="datasets")

from src.risk_model.warehouse_models import (
    WarehouseV0Baseline,
    WarehouseV1Context,
    WarehouseV2Temporal,
    WarehouseV3Learned
)
from src.assistant.supervisor_ai import WarehouseSupervisorAssistant

# Initialize warehouse models and supervisor assistant
WAREHOUSE_MODELS = {
    "v0_baseline": WarehouseV0Baseline(),
    "v1_context": WarehouseV1Context(),
    "v2_temporal": WarehouseV2Temporal(),
    "v3_learned": WarehouseV3Learned("models/v3/warehouse_risk_mlp.pt")
}

# Initialize models for construction domain
MODELS = {
    "v0_baseline": V0RuleBaselineModel(),
    "v1_context": V1ContextRiskModel(),
    "v2_temporal": V2TemporalRiskModel(),
    "v3_learned": V3LearnedRiskModel()
}
prism_tracker = PrismTracker()
supervisor_assistant = WarehouseSupervisorAssistant()


@app.get("/")
def get_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/api/scenarios")
def list_scenarios():
    manifest_path = "data/labels/dataset_manifest.json"
    if not os.path.exists(manifest_path):
        return {"scenarios": []}
    with open(manifest_path, "r") as f:
        data = json.load(f)
    return {"scenarios": data.get("clips", [])}


@app.get("/api/scenario/{clip_id}/frames")
def get_scenario_frames(clip_id: str):
    features_file = os.path.join("data/features", f"{clip_id}_features.json")
    tracks_file = os.path.join("data/tracks", f"{clip_id}_tracks.json")

    if not os.path.exists(features_file) or not os.path.exists(tracks_file):
        raise HTTPException(status_code=404, detail=f"Data for {clip_id} not found")

    with open(features_file, "r") as f:
        feat_data = json.load(f)
    with open(tracks_file, "r") as f:
        track_data = json.load(f)

    return {
        "clip_id": clip_id,
        "total_frames": track_data.get("total_frames", 0),
        "fps": track_data.get("fps", 25.0),
        "tracks": track_data.get("frames", []),
        "features": feat_data.get("samples", [])
    }


class PredictRequest(BaseModel):
    model_version: str = "v3_learned"
    features: Dict[str, Any]


@app.post("/api/predict")
def predict_hazard_risk(req: PredictRequest):
    model = MODELS.get(req.model_version)
    if model is None:
        model = MODELS["v3_learned"]

    # Reconstruct HazardFeatures
    f_dict = req.features
    try:
        feat = HazardFeatures(
            track_id=int(f_dict.get("track_id", 1)),
            frame_id=int(f_dict.get("frame_id", 0)),
            timestamp=float(f_dict.get("timestamp", 0.0)),
            helmet_missing=float(f_dict.get("helmet_missing", 0.0)),
            vest_missing=float(f_dict.get("vest_missing", 0.0)),
            ppe_violation_count=float(f_dict.get("ppe_violation_count", 0.0)),
            ppe_violation_confidence=float(f_dict.get("ppe_violation_confidence", 0.9)),
            normalized_worker_machine_dist=float(f_dict.get("normalized_worker_machine_dist", 1.0)),
            worker_machine_overlap=float(f_dict.get("worker_machine_overlap", 0.0)),
            worker_count_machine_zone=float(f_dict.get("worker_count_machine_zone", 0.0)),
            machine_proximity_severity=float(f_dict.get("machine_proximity_severity", 0.0)),
            violation_duration=float(f_dict.get("violation_duration", 0.0)),
            consecutive_violating_frames=float(f_dict.get("consecutive_violating_frames", 0.0)),
            frames_since_first_detection=float(f_dict.get("frames_since_first_detection", 1.0)),
            persistence_ratio=float(f_dict.get("persistence_ratio", 0.0)),
            num_workers=float(f_dict.get("num_workers", 1.0)),
            num_machines=float(f_dict.get("num_machines", 1.0)),
            simultaneous_violations=float(f_dict.get("simultaneous_violations", 0.0)),
            scene_hazard_density=float(f_dict.get("scene_hazard_density", 0.0)),
            mean_detection_confidence=float(f_dict.get("mean_detection_confidence", 0.9)),
            min_relevant_confidence=float(f_dict.get("min_relevant_confidence", 0.85)),
            confidence_variance=float(f_dict.get("confidence_variance", 0.0))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature parsing error: {e}")

    pred = model.predict(feat)
    return {
        "model_version": req.model_version,
        "prediction": pred.to_dict()
    }


# ==================== WAREHOUSE MATERIAL HANDLING API ====================
class WarehousePredictRequest(BaseModel):
    model_version: str = "v3_learned"
    features: List[float]


WarehousePredictRequest.model_rebuild()


class AssistantChatRequest(BaseModel):
    message: str


@app.get("/api/warehouse/scenarios")
def list_warehouse_scenarios():
    manifest_path = "data/labels/warehouse_manifest.json"
    if not os.path.exists(manifest_path):
        return {"scenarios": []}
    with open(manifest_path, "r") as f:
        data = json.load(f)
    return {"scenarios": data.get("scenarios", [])}


@app.get("/api/warehouse/scenario/{clip_id}/frames")
def get_warehouse_scenario_frames(clip_id: str):
    features_file = os.path.join("data/features", f"{clip_id}_features.json")
    tracks_file = os.path.join("data/tracks", f"{clip_id}_tracks.json")
    labels_file = os.path.join("data/labels", f"{clip_id}_labels.json")

    if not os.path.exists(features_file) or not os.path.exists(tracks_file):
        raise HTTPException(status_code=404, detail=f"Data for {clip_id} not found")

    with open(features_file, "r") as f:
        feat_data = json.load(f)
    with open(tracks_file, "r") as f:
        track_data = json.load(f)
    labels_data = {}
    if os.path.exists(labels_file):
        with open(labels_file, "r") as lf:
            labels_data = json.load(lf)

    return {
        "clip_id": clip_id,
        "total_frames": track_data.get("total_frames", 100),
        "fps": 25.0,
        "tracks": track_data.get("tracks", []),
        "features": feat_data.get("samples", []),
        "ground_truth": labels_data.get("frames", [])
    }


@app.post("/api/warehouse/predict")
def predict_warehouse_risk(req: WarehousePredictRequest):
    model = WAREHOUSE_MODELS.get(req.model_version)
    if model is None:
        model = WAREHOUSE_MODELS["v3_learned"]
    if len(req.features) != 16:
        raise HTTPException(status_code=400, detail="Expected 16 kinematic features")
    pred = model.predict(req.features)
    return {
        "model_version": req.model_version,
        "prediction": pred.to_dict()
    }


@app.get("/api/warehouse/benchmark")
def get_warehouse_benchmark():
    chart_path = "reports/warehouse_improvement_chart.json"
    if not os.path.exists(chart_path):
        return {}
    with open(chart_path, "r") as f:
        return json.load(f)


@app.post("/api/assistant/chat")
def chat_with_supervisor_assistant(req: AssistantChatRequest):
    res = supervisor_assistant.query(req.message)
    return res


@app.get("/api/benchmark")
def get_benchmark_table():
    chart_path = "reports/improvement_chart.json"
    if not os.path.exists(chart_path):
        return {"models": []}
    with open(chart_path, "r") as f:
        return json.load(f)


@app.get("/api/failure_analysis")
def get_failure_analysis():
    report_path = "reports/failure_analysis_report.json"
    if not os.path.exists(report_path):
        return {"summary": {}, "iterations": {}}
    with open(report_path, "r") as f:
        return json.load(f)


@app.get("/api/prism_runs")
def get_prism_runs():
    return {"runs": prism_tracker.get_all_runs()}


# ==================== DATA INGESTION API ====================
from src.ingestion.kaggle_ingestor import KaggleIngestor
from src.ingestion.pipeline import IngestionPipeline
from fastapi import UploadFile, File, Form

ingestion_pipeline = IngestionPipeline()
kaggle_ingestor = KaggleIngestor()


@app.get("/api/ingestion/kaggle/status")
def get_kaggle_status():
    return kaggle_ingestor.check_connection()


@app.get("/api/ingestion/curated")
def get_curated_datasets():
    return {"datasets": KaggleIngestor.CURATED_DATASETS}


@app.get("/api/ingestion/kaggle/search")
def search_kaggle(q: str = "construction safety", max_results: int = 10):
    try:
        results = kaggle_ingestor.search_datasets(query=q, max_results=max_results)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class KaggleIngestRequest(BaseModel):
    dataset_slug: str
    target_split: str = "train"
    auto_train: bool = False


@app.post("/api/ingestion/kaggle/download")
def download_kaggle_dataset(req: KaggleIngestRequest):
    try:
        res = ingestion_pipeline.ingest_from_kaggle(
            dataset_slug=req.dataset_slug,
            target_split=req.target_split,
            auto_train=req.auto_train
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/api/ingestion/local/upload")
async def upload_local_file(file: UploadFile = File(...), target_split: str = Form("train"), auto_train: bool = Form(False)):
    upload_dir = "data/ingested/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, file.filename)
    with open(saved_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        res = ingestion_pipeline.ingest_from_local(
            file_or_dir=saved_path,
            target_split=target_split,
            auto_train=auto_train
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {e}")


@app.get("/api/ingestion/manifest")
def get_manifest():
    manifest_path = "data/labels/dataset_manifest.json"
    if not os.path.exists(manifest_path):
        return {"clips": [], "splits": {}}
    with open(manifest_path, "r") as f:
        return json.load(f)


# ==================== PPE ANALYSER API ====================
from src.features.ppe_inference import PPEInferenceEngine

ppe_engine = PPEInferenceEngine(
    model_path="models/ppe_reasoner.pt",
    yolo_path="models/yolov8_ppe_best.pt" if os.path.exists("models/yolov8_ppe_best.pt") else "yolov8n.pt"
)


@app.post("/api/ppe/analyze")
async def analyze_ppe_image(file: UploadFile = File(...)):
    """Accepts uploaded test image, executes YOLOv8 + PPEReasonerNet, and returns metrics & visual annotations."""
    try:
        content = await file.read()
        res = ppe_engine.analyze_image_bytes(content, filename=file.filename)
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPE analysis failed: {str(e)}")


@app.get("/api/ppe/info")
def get_ppe_analyser_info():
    """Returns architecture info and files needed by the PPE Analyser."""
    metrics_path = "models/ppe_reasoner_metrics.json"
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    return {
        "status": "ready",
        "files_needed": PPEInferenceEngine.FILES_USED,
        "weights_path": "models/ppe_reasoner.pt",
        "training_metrics": metrics
    }


@app.get("/api/ppe/samples")
def get_ppe_samples():
    """Returns sample images from datasets/ppe_master_folder for 1-click test inspection."""
    import glob
    samples = []
    base_folder = "datasets/ppe_master_folder"
    if os.path.exists(base_folder):
        all_imgs = glob.glob(f"{base_folder}/**/*.jpg", recursive=True)
        # Select 6 varied samples across ppe1, ppe2, ppe3
        for img_p in all_imgs[:6]:
            rel_p = os.path.relpath(img_p, ".").replace("\\", "/")
            samples.append({
                "path": rel_p,
                "url": f"/{rel_p}",
                "name": os.path.basename(img_p)
            })
    return {"samples": samples}


class AnalyzeSampleRequest(BaseModel):
    sample_path: str


@app.post("/api/ppe/analyze_sample")
def analyze_sample_image(req: AnalyzeSampleRequest):
    """Analyzes a preset sample image from datasets/ppe_master_folder."""
    if not os.path.exists(req.sample_path):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    try:
        with open(req.sample_path, "rb") as f:
            content = f.read()
        res = ppe_engine.analyze_image_bytes(content, filename=os.path.basename(req.sample_path))
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=False)
