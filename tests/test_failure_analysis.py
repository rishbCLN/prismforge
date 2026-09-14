"""Unit tests for Failure Analysis and Failure-Driven Improvement."""
import unittest
from src.failure_analysis.analyzer import FailureAnalyzer
from src.features.feature_extractor import HazardFeatures


class TestFailureAnalysis(unittest.TestCase):
    def test_transient_cluster_assignment(self):
        feat = {
            "violation_duration": 0.2,
            "machine_proximity_severity": 0.7,
            "mean_detection_confidence": 0.9,
            "worker_count_machine_zone": 1.0,
            "normalized_worker_machine_dist": 0.15
        }
        cluster = FailureAnalyzer.assign_cluster(feat, y_true="LOW", y_pred="HIGH")
        self.assertEqual(cluster, "cluster_a_transient_proximity")

    def test_safe_zone_cluster_assignment(self):
        feat = {
            "violation_duration": 2.0,
            "machine_proximity_severity": 0.05,
            "mean_detection_confidence": 0.95,
            "worker_count_machine_zone": 0.0,
            "normalized_worker_machine_dist": 0.8,
            "helmet_missing": 1.0
        }
        cluster = FailureAnalyzer.assign_cluster(feat, y_true="LOW", y_pred="HIGH")
        self.assertEqual(cluster, "cluster_c_safe_zone_ppe")

    def test_compare_runs(self):
        old_analysis = {
            "total_failures": 25,
            "clusters": [
                {"cluster_id": "cluster_a_transient_proximity", "sample_count": 15},
                {"cluster_id": "cluster_b_detector_noise", "sample_count": 5},
                {"cluster_id": "cluster_c_safe_zone_ppe", "sample_count": 3},
                {"cluster_id": "cluster_d_compound_multi_worker", "sample_count": 2}
            ]
        }
        new_analysis = {
            "total_failures": 10,
            "clusters": [
                {"cluster_id": "cluster_a_transient_proximity", "sample_count": 2}, # -13
                {"cluster_id": "cluster_b_detector_noise", "sample_count": 3},
                {"cluster_id": "cluster_c_safe_zone_ppe", "sample_count": 3},
                {"cluster_id": "cluster_d_compound_multi_worker", "sample_count": 2}
            ]
        }

        comp = FailureAnalyzer.compare_runs(old_analysis, new_analysis, "V1", "V2")
        self.assertEqual(comp["error_reduction"], 15)
        self.assertEqual(comp["new_total_errors"], 10)
        # Check cluster_a improved
        transient_comp = [c for c in comp["comparison"] if c["cluster_id"] == "cluster_a_transient_proximity"][0]
        self.assertEqual(transient_comp["status"], "IMPROVED")
        self.assertEqual(transient_comp["reduction_count"], 13)


if __name__ == "__main__":
    unittest.main()
