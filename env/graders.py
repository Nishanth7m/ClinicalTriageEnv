"""
Graders for all 3 tasks.
Each grader returns a RewardBreakdown with scores strictly in (0.0, 1.0).
IMPORTANT: Scores must NEVER be exactly 0.0 or 1.0 per hackathon rules.
"""
from __future__ import annotations
from env.models import PatientAction, RewardBreakdown, TriageLevel

ADJACENT = {
    TriageLevel.EMERGENT:   {TriageLevel.URGENT},
    TriageLevel.URGENT:     {TriageLevel.EMERGENT, TriageLevel.NON_URGENT},
    TriageLevel.NON_URGENT: {TriageLevel.URGENT},
}

VALID_GUIDELINES = {
    "AHA 2023 Chest Pain", "NICE Sepsis 2024", "BTS Asthma 2023",
    "ACEP Headache", "Ottawa Ankle Rules", "SOFA Score", "NEWS2 Score",
    "WHO Triage", "AHA STEMI 2022", "NICE Pneumonia 2024"
}


def _clamp(value: float) -> float:
    """Ensure score is strictly between 0 and 1 — never exactly 0.0 or 1.0."""
    return max(0.001, min(0.999, value))


def grade_task1(action: PatientAction, gold_label: str) -> RewardBreakdown:
    r = RewardBreakdown()
    if action.task1 is None:
        r.triage_accuracy = 0.001
        r.total = 0.001
        return r

    predicted = action.task1.triage_level.value
    gold = gold_label

    if predicted == gold:
        r.triage_accuracy = 0.85
    elif predicted in [t.value for t in ADJACENT.get(TriageLevel(gold), set())]:
        r.triage_accuracy = 0.5
    else:
        r.triage_accuracy = 0.05

    # Hard penalty: missed emergent
    if gold == "emergent" and predicted == "non_urgent":
        r.emergent_penalty = -0.4

    # Guideline citation bonus
    if action.task1.cited_guideline:
        if any(g.lower() in action.task1.cited_guideline.lower()
               for g in VALID_GUIDELINES):
            r.guideline_bonus = 0.08

    # Reasoning quality
    rationale = action.task1.rationale or ""
    if len(rationale) > 80:
        r.reasoning_quality = 0.07

    raw = (r.triage_accuracy + r.emergent_penalty
           + r.guideline_bonus + r.reasoning_quality)
    r.total = _clamp(raw)
    return r


def grade_task2(action: PatientAction, gold_label: str) -> RewardBreakdown:
    r = RewardBreakdown()
    if action.task2 is None:
        r.triage_accuracy = 0.001
        r.total = 0.001
        return r

    predicted = action.task2.triage_level.value
    gold = gold_label

    if predicted == gold:
        r.triage_accuracy = 0.75
    elif predicted in [t.value for t in ADJACENT.get(TriageLevel(gold), set())]:
        r.triage_accuracy = 0.45
    else:
        r.triage_accuracy = 0.05

    if gold == "emergent" and predicted == "non_urgent":
        r.emergent_penalty = -0.35

    # Guideline citations — up to +0.15
    cited = action.task2.cited_guidelines or []
    valid_count = sum(
        1 for c in cited
        if any(g.lower() in c.lower() for g in VALID_GUIDELINES)
    )
    r.guideline_bonus = min(0.15, valid_count * 0.05)

    # Reasoning quality
    if len(action.task2.reasoning or "") > 150:
        r.reasoning_quality = 0.1

    raw = (r.triage_accuracy + r.emergent_penalty
           + r.guideline_bonus + r.reasoning_quality)
    r.total = _clamp(raw)
    return r


def grade_task3(action: PatientAction, gold_icu_patient: str) -> RewardBreakdown:
    r = RewardBreakdown()
    if action.task3 is None:
        r.triage_accuracy = 0.001
        r.total = 0.001
        return r

    decisions = action.task3.allocation_decisions
    allocated = [d.patient_id for d in decisions if d.allocated_icu]

    if len(allocated) == 1 and allocated[0] == gold_icu_patient:
        r.triage_accuracy = 0.75
    elif len(allocated) == 1:
        r.triage_accuracy = 0.2
    else:
        r.triage_accuracy = 0.05

    # Reasoning quality
    if len(action.task3.overall_reasoning or "") > 200:
        r.reasoning_quality = 0.1

    cited = action.task3.cited_guidelines or []
    valid_count = sum(
        1 for c in cited
        if any(g.lower() in c.lower() for g in VALID_GUIDELINES)
    )
    r.guideline_bonus = min(0.15, valid_count * 0.05)

    raw = (r.triage_accuracy + r.guideline_bonus + r.reasoning_quality)
    r.total = _clamp(raw)
    return r
