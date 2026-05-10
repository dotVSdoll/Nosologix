from enum import StrEnum

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"


class SafetyAssessment(BaseModel):
    risk_level: RiskLevel = RiskLevel.LOW
    should_seek_doctor: bool = False
    safety_warnings: list[str] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)
