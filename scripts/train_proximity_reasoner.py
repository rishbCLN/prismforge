"""Master Training Script for LargeVehicleProximityNet V1.

Usage:
    python scripts/train_proximity_reasoner.py [options]

Options:
    --vehicle-dataset DIR   Path to large_vehicle dataset (default: datasets/large_vehicle)
    --large-dataset DIR     Path to large_dataset (default: datasets/large_dataset)
    --output-path PATH      Output model path (default: models/proximity_reasoner.pt)
    --epochs N              Max training epochs (default: 60)
    --batch-size N          Batch size (default: 64)
    --lr FLOAT              Learning rate (default: 0.002)
    --early-stopping N      Patience for early stopping, 0=disable (default: 10)
    --no-ema                Disable Exponential Moving Average weights
    --no-focal-loss         Disable focal loss
    --no-augmentation       Disable feature augmentation
    --label-smoothing FLOAT Label smoothing for CE loss (default: 0.05)
    --gradient-clip FLOAT   Max gradient norm (default: 1.0)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch

from src.ingestion.large_vehicle_ingestor import LargeVehicleIngestor
from src.features.proximity_features import extract_proximity_samples
from src.training.train_proximity_net import train_proximity_model, get_optimal_device


def main():
    parser = argparse.ArgumentParser(
        description="Train LargeVehicleProximityNet V1 on large vehicle datasets."
    )
    parser.add_argument("--vehicle-dataset", default="datasets/large_vehicle",
                        help="Path to large_vehicle Pascal VOC dataset")
    parser.add_argument("--large-dataset", default="datasets/large_dataset",
                        help="Path to YOLO-format large_dataset (for person context)")
    parser.add_argument("--output-path", default="models/proximity_reasoner.pt",
                        help="Output path for trained model weights")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--device", default="auto",
                        help="Compute device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--force-gpu", action="store_true", default=True)
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    # Boolean flags with negation
    parser.add_argument("--ema",          action="store_true", default=True)
    parser.add_argument("--no-ema",       action="store_true", default=False)
    parser.add_argument("--focal-loss",   action="store_true", default=True)
    parser.add_argument("--no-focal-loss",action="store_true", default=False)
    parser.add_argument("--augmentation", action="store_true", default=True)
    parser.add_argument("--no-augmentation", action="store_true", default=False)

    args = parser.parse_args()

    use_ema   = args.ema and not args.no_ema
    use_focal = args.focal_loss and not args.no_focal_loss
    use_aug   = args.augmentation and not args.no_augmentation

    print("=" * 70)
    print("   HAZARDMESH — LargeVehicleProximityNet V1 TRAINING PIPELINE")
    print("=" * 70)
    print(f"Vehicle Dataset:   {os.path.abspath(args.vehicle_dataset)}")
    print(f"Large Dataset:     {os.path.abspath(args.large_dataset)}")
    print(f"Output:            {args.output_path}")
    print(f"Epochs:            {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"EMA: {use_ema}  |  Focal: {use_focal}  |  Augment: {use_aug}")

    # Device
    target_device, desc = get_optimal_device(
        force_gpu=args.force_gpu,
        requested_device=args.device if args.device != "auto" else None,
    )
    print(f"Compute Device:    {desc}")

    # 1. Ingest datasets
    print("\n--- Ingesting Datasets ---")
    ingestor = LargeVehicleIngestor(
        large_vehicle_dir=args.vehicle_dataset,
        large_dataset_dir=args.large_dataset,
        use_large_dataset=True,
    )
    manifest = ingestor.ingest_all()

    if manifest["vehicle_images"] == 0:
        print(f"[ERROR] No images with vehicle annotations found.")
        print(f"  Check that '{args.vehicle_dataset}' exists and contains VOC XML annotations.")
        sys.exit(1)

    # 2. Extract proximity features
    print("\n--- Extracting Proximity Feature Vectors ---")
    samples = extract_proximity_samples(
        manifest,
        augment=use_aug,
        hard_negative_ratio=0.20,
        synthetic_danger_ratio=0.15,
        max_pairs_per_image=8,
    )
    print(f"Total training samples: {len(samples)}")

    from collections import Counter
    cls_dist = Counter(s["proximity_class"] for s in samples)
    cls_names = ["SAFE", "SUPERVISED", "DANGER", "TOO_FAR"]
    for cid, cname in enumerate(cls_names):
        print(f"  {cname:12s}: {cls_dist.get(cid, 0):5d} samples")

    if len(samples) < 20:
        print("[ERROR] Too few samples to train. Check dataset paths and structure.")
        sys.exit(1)

    # 3. Train
    print("\n--- Training LargeVehicleProximityNet V1 ---")
    results = train_proximity_model(
        samples=samples,
        output_path=args.output_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=0.20,
        device_name=str(target_device),
        show_progress=True,
        use_focal_loss=use_focal,
        label_smoothing=args.label_smoothing,
        use_augmentation=use_aug,
        early_stopping_patience=args.early_stopping,
        use_ema=use_ema,
        gradient_clip_norm=args.gradient_clip,
        force_gpu=args.force_gpu,
    )

    # 4. Summary
    m = results["final_metrics"]
    print()
    print("=" * 70)
    print("              TRAINING COMPLETE — PROXIMITY MODEL V1")
    print("=" * 70)
    print(f"Model Weights Saved:     {results['output_path']}")
    print(f"Architecture:            {results.get('architecture', 'LargeVehicleProximityNet V1')}")
    print(f"Parameters:              {results.get('trainable_parameters', 'N/A'):,}")
    print(f"Training Time:           {results['training_time_seconds']}s")
    print(f"Epochs (best):           {results.get('best_epoch', 'N/A')} / {results['epochs']}")
    print(f"Total Samples:           {results['total_samples']} "
          f"(Train: {results['train_samples']}, Val: {results['val_samples']})")
    print(f"Proximity Class Acc:     {m.get('val_proximity_class_acc', 'N/A')}%")
    print(f"Risk Score MAE:          {m.get('val_risk_score_mae', 'N/A')}")
    print(f"DANGER Zone Recall:      {m.get('val_danger_recall', 'N/A')}%")
    print("=" * 70)
    print()
    print("Training command to retrain:")
    print(f"  python scripts/train_proximity_reasoner.py --vehicle-dataset {args.vehicle_dataset}")


if __name__ == "__main__":
    main()
