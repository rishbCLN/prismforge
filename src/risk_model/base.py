"""HazardMesh Base Risk Model and Output Schemas.
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from src.features.feature_extractor import HazardFeatures


@dataclass
class RiskOutput:
    risk_score: float                     # Continuous in [0.0, 1.0]
    severity: str                         # 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH'
    priority: str                         # 'MONITOR' | 'REVIEW' | 'IMMEDIATE'
    primary_factors: List[str]            # Top contributing factors (e.g. "+ High machinery proximity")
    factor_breakdown: Dict[str, float] = field(default_factory=dict)

    SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]
    PRIORITY_LEVELS = ["MONITOR", "REVIEW", "IMMEDIATE"]

    @classmethod
    def score_to_severity(cls, score: float) -> str:
        if score < 0.20:
            return "NONE"
        elif score < 0.50:
            return "LOW"
        elif score < 0.75:
            return "MEDIUM"
        else:
            return "HIGH"

    @classmethod
    def severity_to_priority(cls, severity: str) -> str:
        if severity == "HIGH":
            return "IMMEDIATE"
        elif severity in ("MEDIUM", "LOW"):
            return "REVIEW"
        else:
            return "MONITOR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 3),
            "severity": self.severity,
            "priority": self.priority,
            "primary_factors": self.primary_factors,
            "factor_breakdown": {k: round(v, 3) for k, v in self.factor_breakdown.items()}
        }


class BaseRiskModel(ABC):
    """Abstract interface for all HazardMesh risk model versions."""

    def __init__(self, version_name: str):
        self.version_name = version_name

    @abstractmethod
    def predict(self, features: HazardFeatures) -> RiskOutput:
        """Predicts risk score, severity, and intervention priority from structured features."""
        pass
