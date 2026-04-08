import uvicorn
from fastapi import FastAPI
from env.environment import ClinicalEnvironment, TASK_SEQUENCE
from env.models import PatientAction, TriageObservation, ClinicalState

app = FastAPI(title="ClinicalTriageEnv", version="0.1.0")
env = ClinicalEnvironment()

@app.get("/")
def root():
    return {"status": "Clinical Triage Env Running", "env": "ClinicalTriageEnv"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    """Return all task IDs — required by the OpenEnv validator."""
    return {"tasks": [t.value for t in TASK_SEQUENCE]}

@app.post("/reset", response_model=TriageObservation)
def reset():
    return env.reset()

@app.post("/step", response_model=TriageObservation)
def step(action: PatientAction):
    return env.step(action)

@app.get("/state", response_model=ClinicalState)
def state():
    return env.state()

def main():
    # This makes the main() function callable by the validator
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
