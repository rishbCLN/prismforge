"""Improvement Chart Generator Script.
Loads evaluation results from saved files and generates the central project improvement artifact:
an automated comparative chart (PNG) and structured metrics table (JSON).
Usage:
    python scripts/generate_improvement_chart.py [--reports-dir reports]
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import matplotlib.pyplot as plt


def generate_chart(reports_dir: str):
    version_keys = ["v0_baseline", "v1_context", "v2_temporal", "v3_learned"]
    version_labels = ["V0: Baseline", "V1: Context", "V2: Temporal", "V3: Learned"]

    eval_data = []
    for vk in version_keys:
        fpath = os.path.join(reports_dir, f"{vk}_eval.json")
        if not os.path.exists(fpath):
            print(f"Error: {fpath} not found. Run evaluate_model.py --all first.")
            return
        with open(fpath, "r") as f:
            eval_data.append(json.load(f))

    # Extract metrics
    macro_f1 = [d["metrics"]["macro_f1"] for d in eval_data]
    high_risk_recall = [d["metrics"]["high_risk_recall"] for d in eval_data]
    fpr = [d["metrics"]["false_positive_rate"] for d in eval_data]
    weighted_err = [d["metrics"]["weighted_hazard_error"] for d in eval_data]

    # Save structured JSON
    table_rows = []
    for i in range(len(version_keys)):
        table_rows.append({
            "version": version_keys[i],
            "display_name": version_labels[i],
            "macro_f1": macro_f1[i],
            "high_risk_recall": high_risk_recall[i],
            "false_positive_rate": fpr[i],
            "weighted_hazard_error": weighted_err[i]
        })

    out_json = os.path.join(reports_dir, "improvement_chart.json")
    with open(out_json, "w") as f:
        json.dump({
            "generated_from": "saved_eval_files",
            "benchmark_dataset": "holdout_features.json",
            "models": table_rows
        }, f, indent=2)

    # Print ASCII Table
    print("\n" + "="*80)
    print("                HAZARDMESH MODEL PROGRESSION BENCHMARK TABLE")
    print("="*80)
    print(f"{'Version':<16} | {'Macro F1':<10} | {'High-Risk Recall':<18} | {'False Positive Rate':<20} | {'Weighted Error':<14}")
    print("-"*80)
    for r in table_rows:
        print(f"{r['display_name']:<16} | {r['macro_f1']:<10.4f} | {r['high_risk_recall']:<18.4f} | {r['false_positive_rate']:<20.4f} | {r['weighted_hazard_error']:<14.4f}")
    print("="*80 + "\n")

    # Generate Matplotlib Graphic
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("HazardMesh: Measurable Progress from V0 to V3 (Held-Out Evaluation)", fontsize=14, fontweight="bold")

    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

    # 1. Macro F1
    axes[0, 0].bar(version_labels, macro_f1, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 0].set_title("Macro F1 Score (Higher is Better)", fontweight="semibold")
    axes[0, 0].set_ylim(0, 1.05)
    for i, v in enumerate(macro_f1):
        axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # 2. High-Risk Recall
    axes[0, 1].plot(version_labels, high_risk_recall, marker="o", linewidth=2.5, markersize=8, color="#27ae60")
    axes[0, 1].set_title("High-Risk Recall (Critical Safety Metric)", fontweight="semibold")
    axes[0, 1].set_ylim(0.4, 1.05)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    for i, v in enumerate(high_risk_recall):
        axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # 3. False Positive Rate
    axes[1, 0].bar(version_labels, fpr, color="#9b59b6", alpha=0.8, edgecolor="black")
    axes[1, 0].set_title("False Positive Rate (Nuisance Alarms - Lower is Better)", fontweight="semibold")
    axes[1, 0].set_ylim(0, max(0.5, max(fpr) + 0.1))
    for i, v in enumerate(fpr):
        axes[1, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    # 4. Asymmetric Weighted Hazard Error
    axes[1, 1].plot(version_labels, weighted_err, marker="s", linewidth=2.5, markersize=8, color="#c0392b")
    axes[1, 1].set_title("Asymmetric Weighted Hazard Error (Lower is Better)", fontweight="semibold")
    axes[1, 1].set_ylim(0, max(weighted_err) + 0.5)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    for i, v in enumerate(weighted_err):
        axes[1, 1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    out_png = os.path.join(reports_dir, "improvement_chart.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Generated visual improvement chart: {out_png}")
    print(f"Generated structured improvement table: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Generate HazardMesh improvement chart")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory")
    args = parser.parse_args()
    generate_chart(args.reports_dir)


if __name__ == "__main__":
    main()
