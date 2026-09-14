"""Evaluates Warehouse Risk Models V0, V1, V2, V3 on Held-Out Test Data.
Computes Macro F1, Weighted F1, High-Risk Recall, False Alarm Rate,
and Asymmetric Damage Cost Matrix.
Emits PRISM verifiable runs and generates benchmark progression chart.
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.risk_model.warehouse_models import (
    WarehouseV0Baseline,
    WarehouseV1Context,
    WarehouseV2Temporal,
    WarehouseV3Learned,
    SEVERITY_CLASSES
)
from src.prism.tracker import PrismTracker


def evaluate_warehouse_models(
    test_json: str = "data/features/warehouse_holdout_features.json",
    output_dir: str = "reports",
    prism_runs_file: str = "experiments/prism_runs.json"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prism_runs_file), exist_ok=True)

    with open(test_json, "r") as f:
        samples = json.load(f)

    print(f"Evaluating models on {len(samples)} held-out warehouse test frames...")

    models = [
        WarehouseV0Baseline(),
        WarehouseV1Context(),
        WarehouseV2Temporal(),
        WarehouseV3Learned("models/v3/warehouse_risk_mlp.pt")
    ]

    prism_tracker = PrismTracker(prism_runs_file)
    results = {}

    for model in models:
        version_name = model.version
        print(f"\nEvaluating {version_name}...")

        if hasattr(model, "reset"):
            model.reset()

        true_sevs = []
        pred_sevs = []
        true_scores = []
        pred_scores = []
        latencies = []

        for s in samples:
            feats = s["features"]
            t_start = time.perf_counter()
            out = model.predict(feats)
            t_elapsed = (time.perf_counter() - t_start) * 1000.0 # ms

            true_sevs.append(s["target_severity"])
            pred_sevs.append(out.severity)
            true_scores.append(s["target_risk_score"])
            pred_scores.append(out.risk_score)
            latencies.append(t_elapsed)

        # Metrics calculation
        sev_map = {name: i for i, name in enumerate(SEVERITY_CLASSES)}
        y_true = np.array([sev_map[s] for s in true_sevs])
        y_pred = np.array([sev_map[s] for s in pred_sevs])

        # High risk is classes 2 (HIGH) and 3 (CRITICAL)
        high_risk_true = (y_true >= 2)
        high_risk_pred = (y_pred >= 2)

        # Safe is class 0 (LOW)
        safe_true = (y_true == 0)
        safe_pred_as_high = (y_pred >= 2) & safe_true

        tp_high = np.sum(high_risk_true & high_risk_pred)
        fn_high = np.sum(high_risk_true & (~high_risk_pred))
        fp_safe = np.sum(safe_pred_as_high)
        tn_safe = np.sum(safe_true & (~safe_pred_as_high))

        high_risk_recall = float(tp_high / (tp_high + fn_high)) if (tp_high + fn_high) > 0 else 0.0
        false_alarm_rate = float(fp_safe / np.sum(safe_true)) if np.sum(safe_true) > 0 else 0.0

        # Multi-class Accuracy & F1
        acc = float(np.mean(y_true == y_pred))

        f1_per_class = []
        for c in range(4):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_per_class.append(f1)

        macro_f1 = float(np.mean(f1_per_class))

        # Asymmetric damage risk cost: 5x penalty for missed high-risk, 1x for false alarms
        damage_cost = float(5.0 * fn_high + 1.0 * fp_safe)
        avg_latency = float(np.mean(latencies))

        res = {
            "version": version_name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "high_risk_recall": round(high_risk_recall, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "asymmetric_damage_cost": round(damage_cost, 2),
            "avg_latency_ms": round(avg_latency, 3),
            "test_samples": len(samples)
        }
        results[version_name] = res

        print(f"  Macro F1:           {macro_f1:.4f}")
        print(f"  High-Risk Recall:   {high_risk_recall:.4f}")
        print(f"  False Alarm Rate:   {false_alarm_rate:.4f}")
        print(f"  Damage Risk Cost:   {damage_cost:.1f}")
        print(f"  Avg Latency:        {avg_latency:.2f} ms")

        # Record to PRISM
        prism_tracker.record_run(
            model_version=f"Warehouse_{version_name}",
            dataset_version="warehouse_holdout_test",
            feature_version="kinematic_16_feat",
            hyperparameters={"samples": len(samples), "classes": 4},
            metrics=res,
            diagnosis_note=f"DamageMesh benchmark evaluation of {version_name}"
        )

    # Save summary report JSON
    report_json = os.path.join(output_dir, "warehouse_improvement_chart.json")
    with open(report_json, "w") as f:
        json.dump(results, f, indent=2)

    # Generate visual benchmark chart
    chart_png = os.path.join(output_dir, "warehouse_improvement_chart.png")
    _plot_warehouse_chart(results, chart_png)

    print(f"\nEvaluation reports generated at:")
    print(f"  JSON: {report_json}")
    print(f"  PNG:  {chart_png}")


def _plot_warehouse_chart(results: Dict[str, Any], output_png: str):
    """Plots benchmark progression across V0 to V3."""
    versions = list(results.keys())
    labels = ["V0 (Baseline)", "V1 (Context)", "V2 (Temporal)", "V3 (Learned MLP)"]

    f1_scores = [results[v]["macro_f1"] for v in versions]
    recalls = [results[v]["high_risk_recall"] for v in versions]
    false_alarms = [results[v]["false_alarm_rate"] for v in versions]
    costs = [results[v]["asymmetric_damage_cost"] for v in versions]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#111625")

    colors = ["#f87171", "#fbbf24", "#60a5fa", "#34d399"]

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#182035")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # 1. Macro F1
    bars1 = ax1.bar(labels, f1_scores, color=colors, width=0.55, edgecolor="#ffffff", linewidth=0.6)
    ax1.set_title("Warehouse Macro F1 Progression (V0 -> V3)", color="#f1f5f9", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', color="#f8fafc", fontweight="bold", fontsize=10)

    # 2. High Risk Recall
    bars2 = ax2.bar(labels, recalls, color=colors, width=0.55, edgecolor="#ffffff", linewidth=0.6)
    ax2.set_title("High-Risk Drop & Throw Recall (Damage Prevention)", color="#f1f5f9", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', color="#f8fafc", fontweight="bold", fontsize=10)

    # 3. False Alarm Rate
    bars3 = ax3.bar(labels, false_alarms, color=["#ef4444", "#f59e0b", "#3b82f6", "#10b981"], width=0.55)
    ax3.set_title("False Alarm Rate on Safe Handling (Lower = Better)", color="#f1f5f9", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, max(false_alarms) * 1.3 if max(false_alarms) > 0 else 0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', color="#f8fafc", fontweight="bold", fontsize=10)

    # 4. Asymmetric Damage Cost Matrix
    bars4 = ax4.bar(labels, costs, color=["#e11d48", "#ea580c", "#2563eb", "#059669"], width=0.55)
    ax4.set_title("Asymmetric Damage Risk Cost: 5x FN + 1x FP (Lower = Better)", color="#f1f5f9", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, max(costs) * 1.25)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + max(costs)*0.02, f"{yval:.0f}", ha='center', va='bottom', color="#f8fafc", fontweight="bold", fontsize=10)

    plt.suptitle("PRISM DamageMesh: Material Handling Model Improvement Benchmark", color="#38bdf8", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


if __name__ == "__main__":
    evaluate_warehouse_models()
