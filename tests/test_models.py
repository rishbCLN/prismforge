"""Unit tests for HazardMesh Risk Models (V0, V1, V2, V3)."""
import unittest
import torch
from src.features.feature_extractor import HazardFeatures
from src.risk_model.v0_baseline import V0RuleBaselineModel
from src.risk_model.v1_context import V1ContextRiskModel
from src.risk_model.v2_temporal import V2TemporalRiskModel
from src.risk_model.v3_learned import V3LearnedRiskModel, RiskMLP


class TestRiskModels(unittest.TestCase):
    def setUp(self):
        # Sample high risk feature
        self.high_risk_features = HazardFeatures(
            track_id=1,
            frame_id=10,
            timestamp=2.5,
            helmet_missing=1.0,
            vest_missing=1.0,
            ppe_violation_count=2.0,
            ppe_violation_confidence=0.95,
            normalized_worker_machine_dist=0.08,
            worker_machine_overlap=0.4,
            worker_count_machine_zone=2.0,
            machine_proximity_severity=0.88,
            violation_duration=2.5,
            consecutive_violating_frames=60.0,
            frames_since_first_detection=60.0,
            persistence_ratio=1.0,
            num_workers=2.0,
            num_machines=1.0,
            simultaneous_violations=2.0,
            scene_hazard_density=1.0,
            mean_detection_confidence=0.92,
            min_relevant_confidence=0.88,
            confidence_variance=0.01
        )

        # Sample safe feature
        self.safe_features = HazardFeatures(
            track_id=2,
            frame_id=10,
            timestamp=2.5,
            helmet_missing=0.0,
            vest_missing=0.0,
            ppe_violation_count=0.0,
            ppe_violation_confidence=0.0,
            normalized_worker_machine_dist=0.75,
            worker_machine_overlap=0.0,
            worker_count_machine_zone=0.0,
            machine_proximity_severity=0.0,
            violation_duration=0.0,
            consecutive_violating_frames=0.0,
            frames_since_first_detection=60.0,
            persistence_ratio=0.0,
            num_workers=2.0,
            num_machines=1.0,
            simultaneous_violations=0.0,
            scene_hazard_density=0.0,
            mean_detection_confidence=0.95,
            min_relevant_confidence=0.92,
            confidence_variance=0.005
        )

    def test_v0_baseline(self):
        model = V0RuleBaselineModel()
        out_high = model.predict(self.high_risk_features)
        self.assertIn(out_high.severity, ["HIGH", "MEDIUM"])
        self.assertGreaterEqual(out_high.risk_score, 0.7)

        out_safe = model.predict(self.safe_features)
        self.assertEqual(out_safe.severity, "NONE")
        self.assertLess(out_safe.risk_score, 0.2)

    def test_v1_context(self):
        model = V1ContextRiskModel()
        out_high = model.predict(self.high_risk_features)
        self.assertEqual(out_high.severity, "HIGH")
        self.assertEqual(out_high.priority, "IMMEDIATE")

        out_safe = model.predict(self.safe_features)
        self.assertEqual(out_safe.severity, "NONE")

    def test_v2_temporal(self):
        model = V2TemporalRiskModel()
        out_high = model.predict(self.high_risk_features)
        self.assertEqual(out_high.severity, "HIGH")

        # Transient hazard: short duration & few consecutive frames
        transient_features = HazardFeatures(
            track_id=3,
            frame_id=3,
            timestamp=0.12,
            helmet_missing=1.0,
            vest_missing=0.0,
            ppe_violation_count=1.0,
            ppe_violation_confidence=0.90,
            normalized_worker_machine_dist=0.15,
            worker_machine_overlap=0.1,
            worker_count_machine_zone=1.0,
            machine_proximity_severity=0.6,
            violation_duration=0.12,
            consecutive_violating_frames=3.0,
            frames_since_first_detection=3.0,
            persistence_ratio=1.0,
            num_workers=1.0,
            num_machines=1.0,
            simultaneous_violations=1.0,
            scene_hazard_density=1.0,
            mean_detection_confidence=0.9,
            min_relevant_confidence=0.85,
            confidence_variance=0.01
        )
        out_transient = model.predict(transient_features)
        # Transient should be discounted compared to sustained
        self.assertLess(out_transient.risk_score, out_high.risk_score)

    def test_v3_pytorch_mlp_forward(self):
        mlp = RiskMLP(input_dim=len(HazardFeatures.FEATURE_NAMES))
        dummy_input = torch.randn(4, len(HazardFeatures.FEATURE_NAMES))
        sev_logits, risk_score = mlp(dummy_input)

        self.assertEqual(sev_logits.shape, (4, 4))
        self.assertEqual(risk_score.shape, (4, 1))
        # Sigmoid bounded in [0, 1]
        self.assertTrue(torch.all(risk_score >= 0.0) and torch.all(risk_score <= 1.0))


if __name__ == "__main__":
    unittest.main()
