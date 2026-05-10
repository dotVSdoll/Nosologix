from app.schemas.safety import RiskLevel
from app.services.safety_service import assess_medical_safety


def test_emergency_symptoms_trigger_emergency_level() -> None:
    result = assess_medical_safety("I have chest pain and difficulty breathing")

    assert result.risk_level == RiskLevel.EMERGENCY
    assert result.should_seek_doctor is True
    assert "chest_pain" in result.matched_rules


def test_medication_dosage_triggers_high_risk() -> None:
    result = assess_medical_safety("How much aspirin should I take as dosage?")

    assert result.risk_level == RiskLevel.HIGH
    assert result.should_seek_doctor is True
    assert "medication_dosage" in result.matched_rules


def test_chronic_disease_triggers_medium_risk() -> None:
    result = assess_medical_safety("What is hypertension?")

    assert result.risk_level == RiskLevel.MEDIUM
    assert result.should_seek_doctor is True
    assert "chronic_disease" in result.matched_rules


def test_general_health_question_stays_low_risk() -> None:
    result = assess_medical_safety("What is hydration?")

    assert result.risk_level == RiskLevel.LOW
    assert result.should_seek_doctor is False
