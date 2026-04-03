"""
Graders for all 3 tasks.
Each grader returns a RewardBreakdown with scores filled in.
"""
from __future__ import annotations
from env.models import (
    PatientAction, RewardBreakdown, TriageLevel, TaskName
)

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

def grade_task1(action: PatientAction, gold_label: str) -> RewardBreakdown:
    r = RewardBreakdown()
    if action.task1 is None:
        r.compute_total()
        return r

    predicted = action.task1.triage_level.value
    gold      = gold_label

    if predicted == gold:
        r.triage_accuracy = 1.0
    elif predicted in [t.value for t in ADJACENT.get(TriageLevel(gold), set())]:
        r.triage_accuracy = 0.5
    else:
        r.triage_accuracy = 0.0

    # Hard penalty: if patient was emergent and agent said non_urgent
    if gold == "emergent" and predicted == "non_urgent":
        r.emergent_penalty = -1.0

    # Guideline citation bonus
    if action.task1.cited_guideline:
        if any(g.lower() in action.task1.cited_guideline.lower()
               for g in VALID_GUIDELINES):
            r.guideline_bonus = 0.1

    # Basic reasoning quality (length heuristic — LLM judge in prod)
    rationale = action.task1.rationale or ""
    if len(rationale) > 80:
        r.reasoning_quality = 0.1

    r.compute_total()
    return r

def grade_task2(action: PatientAction, gold_label: str) -> RewardBreakdown:
    r = RewardBreakdown()
    if action.task2 is None:
        r.compute_total()
        return r

    predicted = action.task2.triage_level.value
    gold      = gold_label

    if predicted == gold:
        r.triage_accuracy = 1.0
    elif predicted in [t.value for t in ADJACENT.get(TriageLevel(gold), set())]:
        r.triage_accuracy = 0.5

    if gold == "emergent" and predicted == "non_urgent":
        r.emergent_penalty = -1.0

    # Guideline citations — up to +0.3
    cited = action.task2.cited_guidelines or []
    valid_count = sum(
        1 for c in cited
        if any(g.lower() in c.lower() for g in VALID_GUIDELINES)
    )
    r.guideline_bonus = min(0.3, valid_count * 0.1)

    # Reasoning quality
    if len(action.task2.reasoning or "") > 150:
        r.reasoning_quality = 0.2

    r.compute_total()
    return r

def grade_task3(action: PatientAction, gold_icu_patient: str) -> RewardBreakdown:
    r = RewardBreakdown()
    if action.task3 is None:
        r.compute_total()
        return r

    decisions = action.task3.allocation_decisions
    allocated  = [d.patient_id for d in decisions if d.allocated_icu]

    if len(allocated) == 1 and allocated[0] == gold_icu_patient:
        r.triage_accuracy = 1.0
    elif len(allocated) == 1:
        # Gave the bed to someone, but wrong patient
        r.triage_accuracy = 0.2
    else:
        # Gave bed to nobody or multiple — invalid
        r.triage_accuracy = 0.0

    # Reasoning quality
    if len(action.task3.overall_reasoning or "") > 200:
        r.reasoning_quality = 0.2

    cited = action.task3.cited_guidelines or []
    valid_count = sum(
        1 for c in cited
        if any(g.lower() in c.lower() for g in VALID_GUIDELINES)
    )
    r.guideline_bonus = min(0.3, valid_count * 0.1)

    r.compute_total()
    return r
