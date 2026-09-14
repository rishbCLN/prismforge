"""Master Image Ingestion & Risk Model Training CLI.
Executes the complete pipeline:
  datasets/ subfolders -> Parse YOLO/COCO/VOC/Images -> Spatial Feature Extraction -> PyTorch Training -> PRISM Logging.

Usage:
    python scripts/ingest_and_train_images.py [--datasets-dir datasets] [--epochs 50] [--batch-size 32]
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from src.ingestion.image_dataset_ingestor import ImageDatasetIngestor
from src.features.image_hazard_features import ImageHazardFeatureExtractor
from src.training.train_image_hazard_model import train_image_risk_model
from src.prism.client import PrismClient


def main():
    parser = argparse.ArgumentParser(description="Ingest construction image datasets and train hazard risk model.")
    parser.add_argument("--datasets-dir", default="datasets/ppe_master_folder", help="Root directory containing dataset subfolders (defaults to datasets/ppe_master_folder)")
    parser.add_argument("--output-dir", default="models/image_models", help="Output directory for trained weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    args = parser.parse_args()

    print("=" * 70)
    print("      HAZARDMESH — REAL IMAGE INGESTION & TRAINING PIPELINE       ")
    print("=" * 70)
    print(f"Target datasets directory: {os.path.abspath(args.datasets_dir)}")

    ingestor = ImageDatasetIngestor(datasets_root=args.datasets_dir)
    subdirs = ingestor.scan_datasets_directory()

    if not subdirs:
        print("\n[READY & WAITING FOR DATA]")
        print(f"No dataset subfolders found in '{args.datasets_dir}/'.")
        print("\nTo ingest and train your real datasets:")
        print("  1. Place your dataset folders into 'datasets/' (e.g. 'datasets/my_ppe_data/')")
        print("  2. Supported formats: YOLO (images+labels), COCO (.json), Pascal VOC (.xml), or flat images.")
        print("  3. Run: python scripts/ingest_and_train_images.py")
        print("=" * 70)
        return

    print(f"\nFound {len(subdirs)} dataset subfolder(s): {[os.path.basename(s) for s in subdirs]}")

    # 1. Ingest datasets
    manifest = ingestor.ingest_all()
    if manifest["total_images"] == 0:
        print("[Warning] Datasets scanned but 0 valid images found.")
        return

    # 2. Extract spatial & PPE hazard features
    print("\n--- Extracting Spatial Hazard & PPE Proximity Features ---")
    extractor = ImageHazardFeatureExtractor()
    features = extractor.process_manifest(manifest)
    print(f"Extracted {len(features)} worker hazard feature vectors from {manifest['total_images']} images.")

    if not features:
        print("[Notice] No worker/person instances found in the ingested images to train hazard model.")
        print("Ensure bounding boxes include 'person' / 'worker' class.")
        return

    # 3. Train PyTorch Dual-Head Model
    res = train_image_risk_model(
        features=features,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    # 4. Save summary report
    report_file = "reports/image_training_report.json"
    os.makedirs("reports", exist_ok=True)
    with open(report_file, "w") as f:
        json.dump({
            "total_images": manifest["total_images"],
            "worker_instances": len(features),
            "datasets": manifest["datasets"],
            "training_results": res
        }, f, indent=2)

    # 5. Dispatch Live Telemetry to PRISM Cloud (Free, agent_name='rishabh')
    try:
        client = PrismClient()
        client.emit_trace(
            input_text=f"Trained Image Hazard Model on {len(features)} instances from {manifest['total_images']} images.",
            output_text=f"Validation F1: {res['metrics']['macro_f1']} | Accuracy: {res['metrics']['val_acc']} | MAE: {res['metrics']['risk_mae']}",
            latency_ms=12,
            agent_name="rishabh",
            model="HazardMesh-ImageMLP-V1",
            session_id="image-training-run",
            metadata={"train_samples": res["train_samples"], "val_samples": res["val_samples"]}
        )
        print("Logged training telemetry to PRISM Cloud under agent 'rishabh' ($0 cost).")
    except Exception as e:
        print(f"PRISM Cloud logging skipped: {e}")

    print("\n" + "=" * 70)
    print(f"Pipeline finished successfully! Model weights saved at: {res['model_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
