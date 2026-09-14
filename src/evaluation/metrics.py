"""HazardMesh Evaluation Framework.
Computes standard multi-class metrics (Macro F1, Precision, Recall, Confusion Matrix)
and construction safety-specific metrics (High-Risk Recall, High-Risk FN rate, FPR, Weighted Hazard Error).
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


class SafetyEvaluator:
    """Evaluates risk predictions against ground truth labels."""

    CLASSES = ["NONE", "LOW", "MEDIUM", "HIGH"]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    # Asymmetric safety cost penalty matrix
    # Format: COST_MATRIX[true_idx, pred_idx]
    # Missing HIGH is penalized 5x; missing MEDIUM 3x; missing LOW 2x; false alarm on NONE 1x
    COST_MATRIX = np.array([
        [0.0, 1.0, 1.5, 2.0],  # True NONE: false alarms cost 1.0 - 2.0
        [2.0, 0.0, 1.0, 1.5],  # True LOW: cost 2.0 if missed as NONE
        [3.0, 1.5, 0.0, 1.0],  # True MEDIUM: cost 3.0 if missed as NONE
        [5.0, 4.0, 2.5, 0.0],  # True HIGH: missed as NONE is 5.0, as LOW is 4.0
    ], dtype=np.float32)

    @classmethod
    def evaluate(cls, y_true: List[str], y_pred: List[str], risk_scores: List[float] = None) -> Dict[str, Any]:
        """Calculates comprehensive safety evaluation metrics."""
        y_true_clean = [y if y in cls.CLASS_TO_IDX else "NONE" for y in y_true]
        y_pred_clean = [y if y in cls.CLASS_TO_IDX else "NONE" for y in y_pred]

        y_true_indices = [cls.CLASS_TO_IDX[y] for y in y_true_clean]
        y_pred_indices = [cls.CLASS_TO_IDX[y] for y in y_pred_clean]

        # 1. Standard Metrics
        acc = float(accuracy_score(y_true_clean, y_pred_clean))
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_clean, y_pred_clean, labels=cls.CLASSES, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_clean, y_pred_clean, labels=cls.CLASSES, average="weighted", zero_division=0
        )

        # Per-class metrics
        p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
            y_true_clean, y_pred_clean, labels=cls.CLASSES, average=None, zero_division=0
        )
        per_class = {}
        for i, cname in enumerate(cls.CLASSES):
            per_class[cname] = {
                "precision": round(float(p_per[i]), 4),
                "recall": round(float(r_per[i]), 4),
                "f1": round(float(f1_per[i]), 4),
                "support": int(sup_per[i])
            }

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=cls.CLASSES)

        # 3. Safety-Specific Metrics
        # High-Risk metrics (index 3)
        high_idx = cls.CLASS_TO_IDX["HIGH"]
        high_tp = int(cm[high_idx, high_idx])
        high_actual = int(np.sum(cm[high_idx, :]))
        high_risk_recall = float(high_tp / high_actual) if high_actual > 0 else 1.0
        high_risk_fn_rate = float(1.0 - high_risk_recall)

        # False Positive Rate (predicting LOW/MED/HIGH when ground truth is NONE)
        none_idx = cls.CLASS_TO_IDX["NONE"]
        none_tn = int(cm[none_idx, none_idx])
        none_actual = int(np.sum(cm[none_idx, :]))
        none_fp = int(none_actual - none_tn)
        false_positive_rate = float(none_fp / none_actual) if none_actual > 0 else 0.0

        # 4. Asymmetric Weighted Hazard Error
        total_penalty = 0.0
        n_samples = max(1, len(y_true_indices))
        for t_idx, p_idx in zip(y_true_indices, y_pred_indices):
            total_penalty += cls.COST_MATRIX[t_idx, p_idx]
        weighted_hazard_error = float(total_penalty / n_samples)

        return {
            "sample_count": len(y_true),
            "accuracy": round(acc, 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "high_risk_recall": round(high_risk_recall, 4),
            "high_risk_false_negative_rate": round(high_risk_fn_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "weighted_hazard_error": round(weighted_hazard_error, 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "cost_matrix_description": {
                "NONE_error": 1.0,
                "LOW_error": 2.0,
                "MEDIUM_error": 3.0,
                "HIGH_missed": 5.0
            }
        }
