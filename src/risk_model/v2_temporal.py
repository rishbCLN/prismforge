"""HazardMesh V2 — Temporal Risk Model.
Incorporates temporal persistence, consecutive frame counts, and duration gating
to suppress single-frame detector noise and transient non-hazardous crossings.
"""
from src.features.feature_extractor import HazardFeatures
from src.risk_model.base import BaseRiskModel, RiskOutput
from src.risk_model.v1_context import V1ContextRiskModel


class V2TemporalRiskModel(BaseRiskModel):
    """V2: Temporal Persistence-gated risk scoring."""

    def __init__(
        self,
        min_consecutive_frames: int = 8,
        full_persistence_duration: float = 2.0,
        transient_discount: float = 0.45
    ):
        super().__init__(version_name="v2_temporal")
        self.v1_model = V1ContextRiskModel()
        self.min_consecutive_frames = min_consecutive_frames
        self.full_persistence_duration = full_persistence_duration
        self.transient_discount = transient_discount

    def predict(self, features: HazardFeatures) -> RiskOutput:
        # Compute base contextual risk from spatial & PPE factors
        v1_out = self.v1_model.predict(features)
        base_score = v1_out.risk_score

        # Calculate temporal persistence factor
        # If consecutive frames are below minimum threshold, condition is transient / detector flicker
        consec = features.consecutive_violating_frames
        duration = features.violation_duration
        ratio = features.persistence_ratio

        temporal_weight = 1.0
        primary_factors = list(v1_out.primary_factors)

        has_any_hazard = (features.helmet_missing > 0.5 or features.vest_missing > 0.5 or features.machine_proximity_severity > 0.3)

        if has_any_hazard:
            if consec < self.min_consecutive_frames and duration < 0.5:
                # Transient exposure or brief detector fluctuation
                temporal_weight = self.transient_discount
                primary_factors.append("~ Transient exposure (< 0.5s filtered)")
            else:
                # Persistent exposure
                duration_factor = min(1.0, duration / self.full_persistence_duration)
                temporal_weight = float(0.7 + 0.3 * duration_factor)
                if duration >= 1.5:
                    primary_factors.append(f"+ Persistent exposure ({duration:.1f}s sustained)")

            # If persistence ratio is low, further discount noise
            if ratio < 0.25 and consec < 15:
                temporal_weight *= 0.8
        else:
            temporal_weight = 1.0

        # Confidence gating: if detector confidence is low, penalize uncertainty
        if features.mean_detection_confidence < 0.50:
            temporal_weight *= 0.85
            primary_factors.append("~ Low detector confidence penalty")

        # Modulated risk score
        risk_score = float(min(1.0, max(0.02, base_score * temporal_weight)))

        breakdown = dict(v1_out.factor_breakdown)
        breakdown["temporal_persistence"] = round(features.violation_duration, 2)
        breakdown["temporal_modulation"] = round(temporal_weight, 2)

        severity = RiskOutput.score_to_severity(risk_score)
        priority = RiskOutput.severity_to_priority(severity)

        return RiskOutput(
            risk_score=risk_score,
            severity=severity,
            priority=priority,
            primary_factors=primary_factors,
            factor_breakdown=breakdown
        )
