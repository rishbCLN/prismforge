"""PPE-Specific Evaluation Module.

Comprehensive evaluation metrics for the PPEReasonerNet 4-class violation detection:
  - Per-class Precision, Recall, F1-Score
  - 4-class confusion matrix
  - Binary vest/hardhat accuracy, precision, recall, F1
  - Asymmetric safety cost (missing CRITICAL penalized 5×)
  - Boundary decision analysis (samples near 0.5 threshold)
  - Risk score calibration metrics
  - Per-dataset split analysis
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


class PPEEvaluator:
    """Comprehensive PPE-specific evaluation for PPEReasonerNet."""

    VIOLATION_NAMES = {
        0: "COMPLIANT",
        1: "MISSING_VEST",
        2: "MISSING_HARDHAT",
        3: "CRITICAL_NO_PPE"
    }

    # Asymmetric safety cost penalty matrix
    # COST_MATRIX[true_class][pred_class]
    # Missing CRITICAL is penalized 5×, missing MISSING_HARDHAT 4×, etc.
    COST_MATRIX = np.array([
        # Pred:  COMP   M_VEST  M_HHAT  CRIT
        [0.0,    1.0,    1.5,    2.0],   # True: COMPLIANT (false alarm costs)
        [3.0,    0.0,    1.5,    1.0],   # True: MISSING_VEST (missed vest violation)
        [4.0,    1.5,    0.0,    1.0],   # True: MISSING_HARDHAT (missed hardhat violation)
        [5.0,    3.0,    3.0,    0.0],   # True: CRITICAL_NO_PPE (missed critical is catastrophic)
    ], dtype=np.float32)

    @classmethod
    def evaluate(
        cls,
        y_true_violation: List[int],
        y_pred_violation: List[int],
        y_true_vest: List[float],
        y_pred_vest: List[float],
        y_true_hardhat: List[float],
        y_pred_hardhat: List[float],
        y_true_risk: List[float],
        y_pred_risk: List[float],
        vest_threshold: float = 0.5,
        hardhat_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Computes comprehensive PPE-specific evaluation metrics."""
        n = len(y_true_violation)
        if n == 0:
            return {"error": "No samples to evaluate"}

        results = {}

        # =============================================================
        # 1. 4-Class Violation Confusion Matrix & Per-Class Metrics
        # =============================================================
        confusion = np.zeros((4, 4), dtype=np.int64)
        for t, p in zip(y_true_violation, y_pred_violation):
            if 0 <= t < 4 and 0 <= p < 4:
                confusion[t, p] += 1

        overall_acc = float(np.trace(confusion)) / max(1, n)

        per_class = {}
        for c in range(4):
            tp = confusion[c, c]
            fp = int(confusion[:, c].sum() - tp)
            fn = int(confusion[c, :].sum() - tp)
            tn = int(confusion.sum() - tp - fp - fn)

            precision = float(tp) / max(1, tp + fp)
            recall = float(tp) / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            support = int(confusion[c, :].sum())

            per_class[cls.VIOLATION_NAMES[c]] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
                "true_positives": int(tp),
                "false_positives": fp,
                "false_negatives": fn,
            }

        # Macro and weighted averages
        macro_p = np.mean([per_class[cls.VIOLATION_NAMES[c]]["precision"] for c in range(4)])
        macro_r = np.mean([per_class[cls.VIOLATION_NAMES[c]]["recall"] for c in range(4)])
        macro_f1 = np.mean([per_class[cls.VIOLATION_NAMES[c]]["f1"] for c in range(4)])

        supports = [per_class[cls.VIOLATION_NAMES[c]]["support"] for c in range(4)]
        total_support = sum(supports)
        weighted_f1 = sum(
            per_class[cls.VIOLATION_NAMES[c]]["f1"] * supports[c]
            for c in range(4)
        ) / max(1, total_support)

        results["violation_classification"] = {
            "overall_accuracy": round(overall_acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
            "confusion_labels": [cls.VIOLATION_NAMES[c] for c in range(4)],
        }

        # =============================================================
        # 2. Binary Vest Compliance Metrics
        # =============================================================
        vest_metrics = cls._binary_metrics(
            y_true_vest, y_pred_vest, threshold=vest_threshold, name="vest"
        )
        results["vest_compliance"] = vest_metrics

        # =============================================================
        # 3. Binary Hardhat Compliance Metrics
        # =============================================================
        hardhat_metrics = cls._binary_metrics(
            y_true_hardhat, y_pred_hardhat, threshold=hardhat_threshold, name="hardhat"
        )
        results["hardhat_compliance"] = hardhat_metrics

        # =============================================================
        # 4. Asymmetric Safety Cost
        # =============================================================
        total_cost = 0.0
        for t, p in zip(y_true_violation, y_pred_violation):
            if 0 <= t < 4 and 0 <= p < 4:
                total_cost += cls.COST_MATRIX[t, p]
        avg_cost = total_cost / max(1, n)

        # Critical miss rate: True=CRITICAL predicted as anything else
        critical_true = sum(1 for t in y_true_violation if t == 3)
        critical_missed = sum(1 for t, p in zip(y_true_violation, y_pred_violation) if t == 3 and p != 3)
        critical_miss_rate = float(critical_missed) / max(1, critical_true) if critical_true > 0 else 0.0

        results["safety_cost"] = {
            "total_asymmetric_cost": round(float(total_cost), 2),
            "avg_asymmetric_cost_per_sample": round(float(avg_cost), 4),
            "critical_violation_miss_rate": round(critical_miss_rate, 4),
            "critical_true_count": critical_true,
            "critical_missed_count": critical_missed,
        }

        # =============================================================
        # 5. Risk Score Calibration
        # =============================================================
        risk_true = np.array(y_true_risk, dtype=np.float64)
        risk_pred = np.array(y_pred_risk, dtype=np.float64)
        risk_mae = float(np.mean(np.abs(risk_true - risk_pred)))
        risk_mse = float(np.mean((risk_true - risk_pred) ** 2))
        risk_rmse = float(np.sqrt(risk_mse))

        # Correlation
        if np.std(risk_true) > 1e-8 and np.std(risk_pred) > 1e-8:
            risk_corr = float(np.corrcoef(risk_true, risk_pred)[0, 1])
        else:
            risk_corr = 1.0

        results["risk_score"] = {
            "mae": round(risk_mae, 4),
            "mse": round(risk_mse, 6),
            "rmse": round(risk_rmse, 4),
            "correlation": round(risk_corr, 4),
            "max_absolute_error": round(float(np.max(np.abs(risk_true - risk_pred))), 4),
        }

        # =============================================================
        # 6. Boundary Decision Analysis
        # =============================================================
        # Find samples where vest or hardhat prediction is near 0.5 threshold
        vest_margins = [abs(p - vest_threshold) for p in y_pred_vest]
        hardhat_margins = [abs(p - hardhat_threshold) for p in y_pred_hardhat]

        narrow_vest = sum(1 for m in vest_margins if m < 0.1)
        narrow_hardhat = sum(1 for m in hardhat_margins if m < 0.1)

        avg_vest_margin = float(np.mean(vest_margins)) if vest_margins else 0.0
        avg_hardhat_margin = float(np.mean(hardhat_margins)) if hardhat_margins else 0.0

        results["boundary_analysis"] = {
            "vest_samples_near_boundary": narrow_vest,
            "hardhat_samples_near_boundary": narrow_hardhat,
            "vest_avg_decision_margin": round(avg_vest_margin, 4),
            "hardhat_avg_decision_margin": round(avg_hardhat_margin, 4),
            "boundary_threshold": 0.1,
            "pct_vest_near_boundary": round(narrow_vest / max(1, n) * 100, 2),
            "pct_hardhat_near_boundary": round(narrow_hardhat / max(1, n) * 100, 2),
        }

        # =============================================================
        # 7. Summary
        # =============================================================
        results["summary"] = {
            "total_samples": n,
            "overall_violation_accuracy": round(overall_acc * 100, 2),
            "macro_f1": round(float(macro_f1) * 100, 2),
            "vest_f1": round(vest_metrics["f1_positive"] * 100, 2),
            "hardhat_f1": round(hardhat_metrics["f1_positive"] * 100, 2),
            "risk_mae": round(risk_mae, 4),
            "critical_miss_rate_pct": round(critical_miss_rate * 100, 2),
            "avg_safety_cost": round(float(avg_cost), 4),
        }

        return results

    @classmethod
    def _binary_metrics(
        cls,
        y_true: List[float],
        y_pred: List[float],
        threshold: float = 0.5,
        name: str = ""
    ) -> Dict[str, Any]:
        """Computes binary classification metrics."""
        tp, fp, fn, tn = 0, 0, 0, 0
        for t, p in zip(y_true, y_pred):
            t_bin = 1 if t >= threshold else 0
            p_bin = 1 if p >= threshold else 0
            if t_bin == 1 and p_bin == 1:
                tp += 1
            elif t_bin == 0 and p_bin == 1:
                fp += 1
            elif t_bin == 1 and p_bin == 0:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
        precision_pos = tp / max(1, tp + fp)
        recall_pos = tp / max(1, tp + fn)
        f1_pos = 2 * precision_pos * recall_pos / max(1e-8, precision_pos + recall_pos)

        precision_neg = tn / max(1, tn + fn)
        recall_neg = tn / max(1, tn + fp)
        f1_neg = 2 * precision_neg * recall_neg / max(1e-8, precision_neg + recall_neg)

        return {
            "accuracy": round(accuracy, 4),
            "precision_positive": round(precision_pos, 4),
            "recall_positive": round(recall_pos, 4),
            "f1_positive": round(f1_pos, 4),
            "precision_negative": round(precision_neg, 4),
            "recall_negative": round(recall_neg, 4),
            "f1_negative": round(f1_neg, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }

    @classmethod
    def format_report(cls, results: Dict[str, Any]) -> str:
        """Formats evaluation results into a human-readable report string."""
        lines = []
        lines.append("=" * 70)
        lines.append("       PPE DETECTION EVALUATION REPORT (PPEReasonerNet V3)")
        lines.append("=" * 70)

        s = results.get("summary", {})
        lines.append(f"\nTotal Samples:              {s.get('total_samples', 0):,}")
        lines.append(f"Overall Violation Accuracy:  {s.get('overall_violation_accuracy', 0):.2f}%")
        lines.append(f"Macro F1 Score:              {s.get('macro_f1', 0):.2f}%")
        lines.append(f"Vest F1:                     {s.get('vest_f1', 0):.2f}%")
        lines.append(f"Hardhat F1:                  {s.get('hardhat_f1', 0):.2f}%")
        lines.append(f"Risk MAE:                    {s.get('risk_mae', 0):.4f}")
        lines.append(f"Critical Miss Rate:          {s.get('critical_miss_rate_pct', 0):.2f}%")
        lines.append(f"Avg Safety Cost:             {s.get('avg_safety_cost', 0):.4f}")

        # Per-class metrics
        vc = results.get("violation_classification", {})
        pc = vc.get("per_class", {})
        if pc:
            lines.append(f"\n{'─' * 70}")
            lines.append(f"{'Class':>20} | {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
            lines.append(f"{'─' * 70}")
            for name in ["COMPLIANT", "MISSING_VEST", "MISSING_HARDHAT", "CRITICAL_NO_PPE"]:
                m = pc.get(name, {})
                lines.append(
                    f"{name:>20} | {m.get('precision', 0):>8.4f} {m.get('recall', 0):>8.4f} "
                    f"{m.get('f1', 0):>8.4f} {m.get('support', 0):>8}"
                )

        # Confusion matrix
        cm = vc.get("confusion_matrix", [])
        if cm:
            lines.append(f"\n{'─' * 70}")
            lines.append("Confusion Matrix:")
            lines.append(f"{'':>20} | {'COMP':>8} {'M_VEST':>8} {'M_HHAT':>8} {'CRIT':>8}")
            lines.append(f"{'─' * 70}")
            labels = ["COMPLIANT", "MISS_VEST", "MISS_HHAT", "CRITICAL"]
            for i, row in enumerate(cm):
                row_str = " ".join(f"{v:>8}" for v in row)
                lines.append(f"{labels[i]:>20} | {row_str}")

        # Safety cost
        sc = results.get("safety_cost", {})
        lines.append(f"\n{'─' * 70}")
        lines.append(f"Asymmetric Safety Cost:     {sc.get('avg_asymmetric_cost_per_sample', 0):.4f}")
        lines.append(f"Critical Missed:            {sc.get('critical_missed_count', 0)} / {sc.get('critical_true_count', 0)}")

        # Boundary analysis
        ba = results.get("boundary_analysis", {})
        lines.append(f"\n{'─' * 70}")
        lines.append(f"Vest samples near boundary:    {ba.get('vest_samples_near_boundary', 0)} ({ba.get('pct_vest_near_boundary', 0):.1f}%)")
        lines.append(f"Hardhat samples near boundary: {ba.get('hardhat_samples_near_boundary', 0)} ({ba.get('pct_hardhat_near_boundary', 0):.1f}%)")
        lines.append(f"Avg Vest decision margin:      {ba.get('vest_avg_decision_margin', 0):.4f}")
        lines.append(f"Avg Hardhat decision margin:   {ba.get('hardhat_avg_decision_margin', 0):.4f}")

        lines.append(f"\n{'=' * 70}")
        return "\n".join(lines)
