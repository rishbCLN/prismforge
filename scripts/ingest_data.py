"""HazardMesh Data Ingestion CLI.
Ingests datasets from Kaggle (using KGAT token) or local video files/archives,
runs perception, tracking, feature extraction, and registers clips into the training corpus.
Usage:
    python scripts/ingest_data.py --search "construction safety"
    python scripts/ingest_data.py --source kaggle --dataset <slug> [--auto-train]
    python scripts/ingest_data.py --source local --path <file_or_dir> [--auto-train]
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.ingestion.kaggle_ingestor import KaggleIngestor
from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="HazardMesh Dataset Ingestion Tool")
    parser.add_argument("--search", type=str, help="Search Kaggle for safety datasets")
    parser.add_argument("--source", choices=["kaggle", "local"], help="Ingestion source type")
    parser.add_argument("--dataset", type=str, help="Kaggle dataset slug (e.g. owner/dataset-name)")
    parser.add_argument("--path", type=str, help="Local file or directory path")
    parser.add_argument("--split", default="train", choices=["train", "val", "holdout"], help="Target dataset split")
    parser.add_argument("--auto-train", action="store_true", help="Automatically retrain V3 PyTorch model after ingestion")
    args = parser.parse_args()

    # Search Mode
    if args.search:
        ingestor = KaggleIngestor()
        print(f"\nSearching Kaggle for: '{args.search}'...")
        results = ingestor.search_datasets(query=args.search, max_results=10)
        print("\n" + "=" * 95)
        print(f"{'Dataset Slug':<55} | {'Size':<12} | {'Downloads':<10} | {'Votes':<6}")
        print("-" * 95)
        for r in results:
            print(f"{r['slug']:<55} | {r['size_human']:<12} | {r['downloads']:<10} | {r['votes']:<6}")
        print("=" * 95 + "\n")
        return

    # Ingestion Mode
    if not args.source:
        parser.print_help()
        return

    pipeline = IngestionPipeline()

    if args.source == "kaggle":
        if not args.dataset:
            print("Error: --dataset <slug> is required when --source is kaggle.")
            return
        print(f"Starting ingestion from Kaggle: {args.dataset} (Target split: {args.split})...")
        res = pipeline.ingest_from_kaggle(dataset_slug=args.dataset, target_split=args.split, auto_train=args.auto_train)
        print("\nIngestion Result:", res)

    elif args.source == "local":
        if not args.path:
            print("Error: --path <file_or_dir> is required when --source is local.")
            return
        print(f"Starting ingestion from local path: {args.path} (Target split: {args.split})...")
        res = pipeline.ingest_from_local(file_or_dir=args.path, target_split=args.split, auto_train=args.auto_train)
        print("\nIngestion Result:", res)


if __name__ == "__main__":
    main()
