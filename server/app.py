from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import ClinicalEnvironment
from env.models import PatientAction, TriageObservation, ClinicalState

app = FastAPI(title="ClinicalTriageEnv", version="0.1.0")
env = ClinicalEnvironment()

@app.get("/")
def root():
    return {"status": "ok", "env": "ClinicalTriageEnv"}

@app.post("/reset", response_model=TriageObservation)
def reset():
    return env.reset()

@app.post("/step", response_model=TriageObservation)
def step(action: PatientAction):
    return env.step(action)

@app.get("/state", response_model=ClinicalState)
def state():
    return env.state()
