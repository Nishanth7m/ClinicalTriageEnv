from __future__ import annotations
from env.models import PatientAction, RewardBreakdown, TriageLevel


ADJACENT = {
    "emergent":   {"urgent"},
    "urgent":     {"emergent", "non_urgent"},
    "non_urgent": {"urgent"},
}

KEYWORDS = {
    "sofa", "news2", "aha", "nice", "bts", "acep", "ottawa",
    "stemi", "sepsis", "pneumonia", "asthma", "chest pain",
}


def _safe_score(value: float) -> float:
    try:
        v = float(value)
        return round(max(0.001, min(0.999, v)), 4)
    except Exception:
        return 0.5


def _keyword_bonus(texts) -> float:
    try:
        if not texts:
            return 0.0
        combined = " ".join(str(t) for t in texts if t).lower()
        hits = sum(1 for k in KEYWORDS if k in combined)
        return min(0.08, hits * 0.02)
    except Exception:
        return 0.0


def grade_task1(action: PatientAction, gold_label: str) -> RewardBreakdown:
    r = RewardBreakdown()
    try:
        gold = str(gold_label or "urgent").lower()

        if not action or not action.task1:
            r.total = _safe_score(0.5)
            return r

        predicted = str(action.task1.triage_level.value).lower()

        if predicted == gold:
            base = 0.72
        elif predicted in ADJACENT.get(gold, set()):
            base = 0.42
        else:
            base = 0.08

        penalty = -0.25 if (gold == "emergent" and predicted == "non_urgent") else 0.0

        rationale = str(action.task1.rationale or "")
        quality = 0.07 if len(rationale) > 30 else 0.02

        cited = str(action.task1.cited_guideline or "")
        bonus = _keyword_bonus([cited])

        r.triage_accuracy  = base
        r.emergent_penalty = penalty
        r.reasoning_quality = quality
        r.guideline_bonus  = bonus
        r.total = _safe_score(base + penalty + quality + bonus)

    except Exception:
        r.total = _safe_score(0.5)
    return r


def grade_task2(action: PatientAction, gold_label: str) -> RewardBreakdown:
    r = RewardBreakdown()
    try:
        gold = str(gold_label or "urgent").lower()

        if not action or not action.task2:
            r.total = _safe_score(0.5)
            return r

        predicted = str(action.task2.triage_level.value).lower()

        if predicted == gold:
            base = 0.68
        elif predicted in ADJACENT.get(gold, set()):
            base = 0.38
        else:
            base = 0.08

        penalty = -0.22 if (gold == "emergent" and predicted == "non_urgent") else 0.0

        reasoning = str(action.task2.reasoning or "")
        quality = 0.10 if len(reasoning) > 60 else 0.02

        guidelines = action.task2.cited_guidelines or []
        bonus = _keyword_bonus(guidelines)

        r.triage_accuracy   = base
        r.emergent_penalty  = penalty
        r.reasoning_quality = quality
        r.guideline_bonus   = bonus
        r.total = _safe_score(base + penalty + quality + bonus)

    except Exception:
        r.total = _safe_score(0.5)
    return r


def grade_task3(action: PatientAction, gold_icu_patient: str) -> RewardBreakdown:
    r = RewardBreakdown()
    try:
        gold = str(gold_icu_patient or "PB")

        if not action or not action.task3:
            r.total = _safe_score(0.5)
            return r

        decisions = action.task3.allocation_decisions or []
        allocated = [str(d.patient_id) for d in decisions if d.allocated_icu]

        if len(allocated) == 1 and allocated[0] == gold:
            base = 0.70
        elif len(allocated) == 1:
            base = 0.20
        else:
            base = 0.08

        reasoning = str(action.task3.overall_reasoning or "")
        quality = 0.10 if len(reasoning) > 80 else 0.02

        guidelines = action.task3.cited_guidelines or []
        bonus = _keyword_bonus(guidelines)

        r.triage_accuracy   = base
        r.reasoning_quality = quality
        r.guideline_bonus   = bonus
        r.total = _safe_score(base + quality + bonus)

    except Exception:
        r.total = _safe_score(0.5)
    return r
