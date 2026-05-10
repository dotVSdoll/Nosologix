from __future__ import annotations

from app.schemas.safety import RiskLevel, SafetyAssessment

_EMERGENCY_RULES = {
    "chest_pain": ["chest pain", "chest tightness", "heart pain"],
    "breathing_difficulty": [
        "difficulty breathing",
        "shortness of breath",
        "cannot breathe",
    ],
    "stroke_symptoms": [
        "face drooping",
        "slurred speech",
        "one side weakness",
        "sudden weakness",
    ],
    "severe_allergy": ["severe allergic", "anaphylaxis", "throat swelling"],
    "suicidal_ideation": ["suicide", "kill myself", "self harm"],
}

_HIGH_RISK_RULES = {
    "medication_dosage": ["dosage", "dose", "how much", "how many pills"],
    "diagnosis_request": ["diagnose", "do i have", "am i having"],
    "treatment_plan": ["treatment plan", "prescribe", "prescription"],
    "pregnancy_child": ["pregnant", "infant", "baby", "child", "children"],
}

_MEDIUM_RISK_RULES = {
    "abnormal_report": ["abnormal", "elevated", "low level", "lab report"],
    "chronic_disease": ["hypertension", "diabetes", "chronic disease"],
}


def assess_medical_safety(question: str, answer: str = "") -> SafetyAssessment:
    text = f"{question}\n{answer}".lower()

    emergency_matches = _match_rules(text, _EMERGENCY_RULES)
    if emergency_matches:
        return SafetyAssessment(
            risk_level=RiskLevel.EMERGENCY,
            should_seek_doctor=True,
            safety_warnings=[
                "This may describe urgent or severe symptoms. "
                "Seek emergency medical care immediately."
            ],
            matched_rules=emergency_matches,
        )

    high_matches = _match_rules(text, _HIGH_RISK_RULES)
    if high_matches:
        return SafetyAssessment(
            risk_level=RiskLevel.HIGH,
            should_seek_doctor=True,
            safety_warnings=[
                "This question may require professional medical judgment. "
                "Consult a licensed clinician."
            ],
            matched_rules=high_matches,
        )

    medium_matches = _match_rules(text, _MEDIUM_RISK_RULES)
    if medium_matches:
        return SafetyAssessment(
            risk_level=RiskLevel.MEDIUM,
            should_seek_doctor=True,
            safety_warnings=[
                "Use this information as general guidance and confirm important "
                "health decisions with a clinician."
            ],
            matched_rules=medium_matches,
        )

    return SafetyAssessment(
        risk_level=RiskLevel.LOW,
        should_seek_doctor=False,
        safety_warnings=["This answer is informational and does not replace medical advice."],
        matched_rules=[],
    )


def _match_rules(text: str, rules: dict[str, list[str]]) -> list[str]:
    matches: list[str] = []
    for rule_name, keywords in rules.items():
        if any(keyword.lower() in text for keyword in keywords):
            matches.append(rule_name)
    return matches
