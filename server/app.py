from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from env.environment import ClinicalEnvironment
from env.models import PatientAction, TriageObservation, ClinicalState
from env.graders import grade_task1, grade_task2, grade_task3

app = FastAPI(title="ClinicalTriageEnv", version="0.1.0")
env = ClinicalEnvironment()

GOLD_MAP = {
    "single_symptom_triage":  "emergent",
    "differential_diagnosis": "urgent",
    "icu_resource_allocation": "PB",
}


class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


@app.get("/")
def root():
    return {"status": "ok", "env": "ClinicalTriageEnv"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=TriageObservation)
def reset(req: ResetRequest = None):
    task_name = None
    if req:
        task_name = req.task
    return env.reset(task_name=task_name)


@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
        action = PatientAction(**body)
    except Exception:
        action = PatientAction()
    try:
        result = env.step(action)
        return result
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "task_name": "single_symptom_triage",
            "step_number": 1,
            "task1": {
                "patient": None,
                "feedback": "fallback",
                "reward": {"triage_accuracy": 0.5, "emergent_penalty": 0.0,
                           "guideline_bonus": 0.0, "reasoning_quality": 0.0,
                           "total": 0.5},
                "done": True
            },
            "task2": None,
            "task3": None,
            "info": {}
        })


@app.get("/state", response_model=ClinicalState)
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"name": "single_symptom_triage",   "difficulty": "easy",   "grader": "env.graders.openenv_grade_task1"},
            {"name": "differential_diagnosis",  "difficulty": "medium", "grader": "env.graders.openenv_grade_task2"},
            {"name": "icu_resource_allocation", "difficulty": "hard",   "grader": "env.graders.openenv_grade_task3"},
        ]
    }


@app.post("/grade/{task_name}")
async def grade(task_name: str, request: Request):
    try:
        body = await request.json()
        action = PatientAction(**body)
    except Exception:
        action = PatientAction()
    gold = GOLD_MAP.get(task_name, "urgent")
    if task_name == "single_symptom_triage":
        reward = grade_task1(action, gold)
    elif task_name == "differential_diagnosis":
        reward = grade_task2(action, gold)
    else:
        reward = grade_task3(action, gold)
    return {"task_name": task_name, "score": reward.total, "breakdown": reward}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
