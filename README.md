# ClinicalTriageEnv

A real-world OpenEnv RL environment for medical triage decision-making.

## Overview

An AI agent learns to perform medical triage across 3 tasks of increasing difficulty:
1. **Easy** — Single-symptom triage (emergent / urgent / non-urgent)
2. **Medium** — Differential diagnosis with multi-symptom presentations
3. **Hard** — ICU resource allocation across multiple critical patients

## Action Space

`PatientAction` — JSON with one of `task1`, `task2`, or `task3` populated.

## Observation Space

`TriageObservation` — JSON containing patient record, feedback, and reward breakdown.

## Reward

- Correct triage: **+1.0**
- Adjacent severity: **+0.5**
- Missed emergent case: **−1.0**
- Per valid guideline cited: **+0.1** (max +0.3)
- Reasoning quality: **+0.0–0.2**

Total clamped to **[−1.0, 1.0]**.

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start new episode, returns first observation |
| POST | `/step` | Submit action, returns observation + reward |
| GET | `/state` | Returns full episode state |

## Baseline

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
export ENV_URL=http://localhost:7860
python inference.py
```
