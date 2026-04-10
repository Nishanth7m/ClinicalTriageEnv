from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from env.environment import ClinicalEnvironment
from env.models import PatientAction, TriageObservation, ClinicalState

app = FastAPI(title="ClinicalTriageEnv", version="0.1.0")
env = ClinicalEnvironment()


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


@app.post("/step", response_model=TriageObservation)
def step(action: PatientAction):
    return env.step(action)


@app.get("/state", response_model=ClinicalState)
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"name": "single_symptom_triage",   "difficulty": "easy"},
            {"name": "differential_diagnosis",  "difficulty": "medium"},
            {"name": "icu_resource_allocation", "difficulty": "hard"},
        ]
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
