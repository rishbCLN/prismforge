"""HazardMesh V1 — Context-Aware Risk Scoring Model.
Incorporates spatial proximity to dangerous machinery and multi-worker scene interactions.
Evaluates hazards contextually rather than relying solely on raw detection presence.
"""
from src.features.feature_extractor import HazardFeatures
from src.risk_model.base import BaseRiskModel, RiskOutput


class V1ContextRiskModel(BaseRiskModel):
    """V1: Context-aware weighted risk scoring."""

    def __init__(
        self,
        weight_helmet: float = 0.25,
        weight_vest: float = 0.15,
        weight_proximity: float = 0.45,
        weight_interaction: float = 0.15
    ):
        super().__init__(version_name="v1_context")
        self.w_helmet = weight_helmet
        self.w_vest = weight_vest
        self.w_prox = weight_proximity
        self.w_inter = weight_interaction

    def predict(self, features: HazardFeatures) -> RiskOutput:
        # 1. Base PPE score
        ppe_score = (
            features.helmet_missing * self.w_helmet +
            features.vest_missing * self.w_vest
        )

        # 2. Machinery proximity factor
        prox_score = features.machine_proximity_severity * self.w_prox

        # 3. Scene interaction factor (density & multiple workers in zone)
        interaction = 0.0
        if features.worker_count_machine_zone > 1.0:
            interaction += 0.08
        if features.scene_hazard_density > 0.4:
            interaction += 0.07
        interaction_score = min(self.w_inter, interaction)

        # 4. Contextual multiplication:
        # A missing PPE in close proximity to active machinery is far more dangerous
        # than missing PPE in an isolated safe corridor.
        context_multiplier = 1.0
        if (features.helmet_missing > 0.5 or features.vest_missing > 0.5) and features.machine_proximity_severity > 0.4:
            context_multiplier = 1.6 # Compounding multiplier

        raw_score = (ppe_score + prox_score + interaction_score) * context_multiplier
        risk_score = float(min(1.0, max(0.02, raw_score)))

        # Determine factors for explanation
        primary_factors = []
        breakdown = {
            "ppe_severity": round(ppe_score * context_multiplier, 3),
            "machinery_proximity": round(prox_score * context_multiplier, 3),
            "scene_interaction": round(interaction_score, 3)
        }

        if features.machine_proximity_severity > 0.5:
            primary_factors.append("+ High machinery proximity zone")
        elif features.machine_proximity_severity > 0.2:
            primary_factors.append("+ Moderate machinery proximity")

        if features.helmet_missing > 0.5:
            primary_factors.append("+ Missing safety helmet")
        if features.vest_missing > 0.5:
            primary_factors.append("+ Missing high-visibility vest")

        if features.worker_count_machine_zone > 1.0:
            primary_factors.append("+ Multiple workers in equipment perimeter")

        if not primary_factors:
            primary_factors.append("No critical hazard factors detected")

        severity = RiskOutput.score_to_severity(risk_score)
        priority = RiskOutput.severity_to_priority(severity)

        return RiskOutput(
            risk_score=risk_score,
            severity=severity,
            priority=priority,
            primary_factors=primary_factors,
            factor_breakdown=breakdown
        )
