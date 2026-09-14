"""HazardMesh V0 — Rule-Based Baseline Model.
Treats any detected PPE violation as an immediate high/medium risk alert,
completely ignoring spatial proximity, temporal persistence, or scene context.
"""
from src.features.feature_extractor import HazardFeatures
from src.risk_model.base import BaseRiskModel, RiskOutput


class V0RuleBaselineModel(BaseRiskModel):
    """V0: Naive Rule-based baseline (any violation = risk)."""

    def __init__(self):
        super().__init__(version_name="v0_baseline")

    def predict(self, features: HazardFeatures) -> RiskOutput:
        # Simplistic rule: if helmet or vest missing, trigger immediate alarm
        has_violation = (features.helmet_missing > 0.5) or (features.vest_missing > 0.5)

        primary_factors = []
        breakdown = {
            "ppe_violation": 0.0,
            "machinery_proximity": 0.0,
            "temporal_persistence": 0.0
        }

        if has_violation:
            # Naive baseline assumes any missing PPE is high risk
            if features.helmet_missing > 0.5 and features.vest_missing > 0.5:
                risk_score = 0.90
                primary_factors.append("Multiple PPE violations detected (Naive)")
            elif features.helmet_missing > 0.5:
                risk_score = 0.80
                primary_factors.append("Missing helmet detected (Naive)")
            else:
                risk_score = 0.70
                primary_factors.append("Missing vest detected (Naive)")

            breakdown["ppe_violation"] = risk_score
            severity = "HIGH" if risk_score >= 0.75 else "MEDIUM"
            priority = "IMMEDIATE" if severity == "HIGH" else "REVIEW"
        else:
            risk_score = 0.05
            severity = "NONE"
            priority = "MONITOR"
            primary_factors.append("Full PPE present (Compliant)")

        return RiskOutput(
            risk_score=risk_score,
            severity=severity,
            priority=priority,
            primary_factors=primary_factors,
            factor_breakdown=breakdown
        )
