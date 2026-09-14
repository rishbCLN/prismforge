"""Failure Analysis Script.
Loads evaluation results across V0, V1, V2, and V3, groups misclassifications into failure clusters,
and evaluates how each version resolves earlier failure modes.
Usage:
    python scripts/analyze_failures.py [--reports-dir reports]
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from src.failure_analysis.analyzer import FailureAnalyzer


def analyze_cross_version_failures(reports_dir: str):
    v0_file = os.path.join(reports_dir, "v0_baseline_eval.json")
    v1_file = os.path.join(reports_dir, "v1_context_eval.json")
    v2_file = os.path.join(reports_dir, "v2_temporal_eval.json")
    v3_file = os.path.join(reports_dir, "v3_learned_eval.json")

    required = [v0_file, v1_file, v2_file, v3_file]
    for r in required:
        if not os.path.exists(r):
            print(f"Error: {r} not found. Run evaluate_model.py --all first.")
            return

    with open(v0_file) as f:
        v0_data = json.load(f)["failure_analysis"]
    with open(v1_file) as f:
        v1_data = json.load(f)["failure_analysis"]
    with open(v2_file) as f:
        v2_data = json.load(f)["failure_analysis"]
    with open(v3_file) as f:
        v3_data = json.load(f)["failure_analysis"]

    # Compare progression: V0 -> V1, V1 -> V2, V2 -> V3
    comp_v0_v1 = FailureAnalyzer.compare_runs(v0_data, v1_data, "v0_baseline", "v1_context")
    comp_v1_v2 = FailureAnalyzer.compare_runs(v1_data, v2_data, "v1_context", "v2_temporal")
    comp_v2_v3 = FailureAnalyzer.compare_runs(v2_data, v3_data, "v2_temporal", "v3_learned")

    full_report = {
        "summary": {
            "v0_total_errors": v0_data["total_failures"],
            "v1_total_errors": v1_data["total_failures"],
            "v2_total_errors": v2_data["total_failures"],
            "v3_total_errors": v3_data["total_failures"],
            "v0_to_v3_reduction": v0_data["total_failures"] - v3_data["total_failures"]
        },
        "iterations": {
            "v0_to_v1": {
                "diagnosis": "V0 flags all PPE violations as emergency regardless of proximity.",
                "intervention": "Added 2D normalized machinery proximity and contextual multipliers.",
                "results": comp_v0_v1
            },
            "v1_to_v2": {
                "diagnosis": "V1 produces false alarms on transient passersby and detector flicker.",
                "intervention": "Added temporal persistence duration gating and consecutive frame tracking.",
                "results": comp_v1_v2
            },
            "v2_to_v3": {
                "diagnosis": "V2 linear weights miscalibrate multi-worker compound danger zones.",
                "intervention": "Trained PyTorch MLP to learn non-linear feature interactions.",
                "results": comp_v2_v3
            }
        },
        "cluster_profiles": FailureAnalyzer.CLUSTER_DEFINITIONS
    }

    out_file = os.path.join(reports_dir, "failure_analysis_report.json")
    with open(out_file, "w") as f:
        json.dump(full_report, f, indent=2)

    # Print summary table
    print("\n========================================================")
    print("       HAZARDMESH FAILURE-DRIVEN IMPROVEMENT REPORT     ")
    print("========================================================")
    print(f"V0 Baseline Total Errors: {v0_data['total_failures']}")
    print(f"V1 Context  Total Errors: {v1_data['total_failures']} (Reduction: {comp_v0_v1['error_reduction']})")
    print(f"V2 Temporal Total Errors: {v2_data['total_failures']} (Reduction: {comp_v1_v2['error_reduction']})")
    print(f"V3 Learned  Total Errors: {v3_data['total_failures']} (Reduction: {comp_v2_v3['error_reduction']})")
    print("--------------------------------------------------------")
    print(f"Overall Error Reduction from V0 to V3: {full_report['summary']['v0_to_v3_reduction']} errors eliminated.")
    print(f"Report saved to: {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Run failure analysis across model versions")
    parser.add_argument("--reports-dir", default="reports", help="Path to reports directory")
    args = parser.parse_args()
    analyze_cross_version_failures(args.reports_dir)


if __name__ == "__main__":
    main()
