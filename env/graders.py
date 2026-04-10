"""
Graders for all 3 tasks.
CRITICAL: All scores must be strictly between 0.001 and 0.999.
Never return exactly 0.0 or 1.0.
"""
from __future__ import annotations
from env.models import PatientAction, RewardBreakdown, TriageLevel

ADJACENT = {
    TriageLevel.EMERGENT:   {TriageLevel.URGENT},
    TriageLevel.URGENT:     {TriageLevel.EMERGENT, TriageLevel.NON_URGENT},
    TriageLevel.NON_URGENT: {TriageLevel.URGENT},
}

VALID_GUIDELINES = {
    "aha 2023 chest pain", "nice sepsis 2024", "bts asthma 2023",
    "acep headache", "ottawa ankle rules", "sofa score", "news2 score",
    "who triage", "aha stemi 2022", "nice pneumonia 2024",
    "sofa", "news2", "aha", "nice", "bts", "acep", "ottawa",
}


def _clamp(value: float) -> float:
    """Strictly between 0 and 1 — never exactly 0.0 or 1.0."""
    return round(max(0.001, min(0.999, value)), 4)


def _guideline_bonus(guidelines: list) -> float:
    if not guidelines:
        return 0.0
    count = sum(
        1 for g in guidelines
        if any(v in g.lower() for v in VALID_GUIDELINES)
    )
    return min(0.1, count * 0.05)


def grade_task1(action: PatientAction, gold_label: str) -> RewardBreakdown:
    r = RewardBreakdown()

    if action.task1 is None:
        r.total = _clamp(0.05)
        return r

    predicted = action.task1.triage_level.value
    gold = gold_label or "urgent"

    if predicted == gold:
        r.triage_accuracy = 0.75
    elif predicted in [t.value for t in ADJACENT.get(TriageLevel(gold), set())]:
        r.triage_accuracy = 0.45
    else:
        r.triage_accuracy = 0.08

    if gold == "emergent" and predicted == "non_urgent":
        r.emergent_penalty = -0.30

    cited = [action.task1.cited_guideline] if action.task1.cited_guideline else []
    r.guideline_bonus = _guideline_bonus(cited)

    rationale = action.task1.rationale or ""
    r.reasoning_quality = 0.08 if len(rationale) > 60 else 0.02

    raw = r.triage_accuracy + r.emergent_penalty + r.guideline_bonus + r.reasoning_quality
    r.total = _clamp(raw)
    return r


def grade_task2(action: PatientAction, gold_label: str) -> RewardBreakdown:
    r = RewardBreakdown()

    if action.task2 is None:
        r.total = _clamp(0.05)
        return r

    predicted = action.task2.triage_level.value
    gold = gold_label or "urgent"

    if predicted == gold:
        r.triage_accuracy = 0.70
    elif predicted in [t.value for t in ADJACENT.get(TriageLevel(gold), set())]:
        r.triage_accuracy = 0.40
    else:
        r.triage_accuracy = 0.08

    if gold == "emergent" and predicted == "non_urgent":
        r.emergent_penalty = -0.25

    r.guideline_bonus = _guideline_bonus(action.task2.cited_guidelines or [])

    reasoning = action.task2.reasoning or ""
    r.reasoning_quality = 0.12 if len(reasoning) > 100 else 0.03

    raw = r.triage_accuracy + r.emergent_penalty + r.guideline_bonus + r.reasoning_quality
    r.total = _clamp(raw)
    return r


def grade_task3(action: PatientAction, gold_icu_patient: str) -> RewardBreakdown:
    r = RewardBreakdown()

    if action.task3 is None:
        r.total = _clamp(0.05)
        return r

    decisions = action.task3.allocation_decisions or []
    allocated = [d.patient_id for d in decisions if d.allocated_icu]
    gold = gold_icu_patient or "PB"

    if len(allocated) == 1 and allocated[0] == gold:
        r.triage_accuracy = 0.72
    elif len(allocated) == 1:
        r.triage_accuracy = 0.18
    else:
        r.triage_accuracy = 0.06

    r.guideline_bonus = _guideline_bonus(action.task3.cited_guidelines or [])

    reasoning = action.task3.overall_reasoning or ""
    r.reasoning_quality = 0.12 if len(reasoning) > 150 else 0.03

    raw = r.triage_accuracy + r.guideline_bonus + r.reasoning_quality
    r.total = _clamp(raw)
    return r
