"""
Baseline inference script for ClinicalTriageEnv.
Structured for Meta PyTorch Hackathon automated evaluation.
"""
import os
import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# 1. Load configuration
load_dotenv()
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "https://nishanthdev7-clinicaltriageenv.hf.space")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", "")

# 2. Initialize Client
client = OpenAI(api_key=HF_TOKEN or OPENAI_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert emergency medicine physician.
Respond ONLY with a valid JSON object matching the required action schema.
Do not add any prose outside the JSON."""

def call_env(endpoint: str, payload: dict = None):
    url = f"{ENV_URL.rstrip('/')}{endpoint}"
    if payload:
        r = requests.post(url, json=payload, timeout=60)
    else:
        r = requests.post(url, timeout=60)
    r.raise_for_status()
    return r.json()

def agent_act(obs: dict) -> dict:
    task_name = obs.get("task_name", "")
    content   = json.dumps(obs, indent=2)

    if task_name == "single_symptom_triage":
        schema = '{"task1": {"triage_level": "emergent|urgent|non_urgent", "rationale": "..."}}'
    elif task_name == "differential_diagnosis":
        schema = '{"task2": {"primary_diagnosis": "...", "triage_level": "emergent", "reasoning": "..."}}'
    else:
        schema = '{"task3": {"allocation_decisions": [{"patient_id": "P001", "allocated_icu": true, "priority_rank": 1, "justification": "..."}]}}'

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Observation: {content}\n\nSchema: {schema}"},
                ],
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except (RateLimitError, json.JSONDecodeError):
            time.sleep(1)
    return {}

def main():
    # REQUIRED BY JUDGES: Start Tag
    print("[START]")
    
    tasks = ["single_symptom_triage", "differential_diagnosis", "icu_resource_allocation"]
    
    for task_name in tasks:
        try:
            # Reset the environment
            obs = call_env("/reset")
            # Get the action from the LLM
            action = agent_act(obs)
            # Step the environment
            result = call_env("/step", action)

            # Extract score from the result
            score = 0.0
            for key in ["task1", "task2", "task3"]:
                if result.get(key) and "reward" in result[key]:
                    score = result[key]["reward"].get("total", 0.0)
                    break
            
            # REQUIRED BY JUDGES: Structured Step Log
            print(f"[STEP] task_name={task_name} score={score:.2f}")

        except Exception:
            print(f"[STEP] task_name={task_name} score=0.00")

    # REQUIRED BY JUDGES: End Tag
    print("[END]")

if __name__ == "__main__":
    main()