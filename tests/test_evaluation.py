"""Unit tests for HazardMesh Evaluation Metrics and Cost Matrix."""
import unittest
from src.evaluation.metrics import SafetyEvaluator


class TestEvaluation(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = ["NONE", "LOW", "MEDIUM", "HIGH"]
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"]

        res = SafetyEvaluator.evaluate(y_true, y_pred)
        self.assertEqual(res["macro_f1"], 1.0)
        self.assertEqual(res["high_risk_recall"], 1.0)
        self.assertEqual(res["false_positive_rate"], 0.0)
        self.assertEqual(res["weighted_hazard_error"], 0.0)

    def test_missed_high_risk_penalty(self):
        # Case where a critical HIGH hazard is missed as NONE
        y_true = ["HIGH"]
        y_pred = ["NONE"]

        res = SafetyEvaluator.evaluate(y_true, y_pred)
        self.assertEqual(res["high_risk_recall"], 0.0)
        self.assertEqual(res["high_risk_false_negative_rate"], 1.0)
        # Missing HIGH as NONE carries penalty 5.0
        self.assertAlmostEqual(res["weighted_hazard_error"], 5.0, places=3)

    def test_false_positive_rate(self):
        # Predicting hazards when true state is NONE
        y_true = ["NONE", "NONE", "NONE", "NONE"]
        y_pred = ["LOW", "MEDIUM", "HIGH", "NONE"]

        res = SafetyEvaluator.evaluate(y_true, y_pred)
        # 3 out of 4 false alarms
        self.assertAlmostEqual(res["false_positive_rate"], 0.75, places=2)


if __name__ == "__main__":
    unittest.main()
