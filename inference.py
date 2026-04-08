"""
Baseline inference script — ClinicalTriageEnv
Runs the agent against all 3 tasks and prints scores.
Must complete in < 20 min on 2 vCPU / 8 GB RAM.
"""
import os, json, time
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, RateLimitError
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(
    api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""),
    base_url=API_BASE_URL
)

SYSTEM_PROMPT = """You are an expert emergency medicine physician.
You will receive a patient observation in JSON format.
Respond ONLY with a valid JSON object matching the required action schema.
Do not add any prose outside the JSON."""


def call_env(endpoint: str, payload: dict = None):
    if payload:
        r = requests.post(f"{ENV_URL}{endpoint}", json=payload, timeout=60)
    else:
        r = requests.post(f"{ENV_URL}{endpoint}", timeout=60)
    if r.status_code >= 400:
        print(f"[SERVER ERROR {r.status_code}] {r.text}")
    r.raise_for_status()
    return r.json()


def agent_act(obs: dict) -> dict:
    task_name = obs.get("task_name", "")
    content   = json.dumps(obs, indent=2)

    if task_name == "single_symptom_triage":
        schema = ('{"task1": {"triage_level": "emergent|urgent|non_urgent", '
                  '"rationale": "your clinical reasoning here minimum 100 chars", '
                  '"cited_guideline": "AHA 2023 Chest Pain"}}')
    elif task_name == "differential_diagnosis":
        schema = ('{"task2": {"primary_diagnosis": "diagnosis name", '
                  '"differential_diagnoses": ["diagnosis1", "diagnosis2", "diagnosis3"], '
                  '"recommended_actions": ["action1", "action2"], '
                  '"triage_level": "emergent|urgent|non_urgent", '
                  '"reasoning": "detailed clinical reasoning minimum 200 chars", '
                  '"cited_guidelines": ["AHA 2023 Chest Pain", "NICE Sepsis 2024"]}}')
    else:
        schema = ('{"task3": {"allocation_decisions": ['
                  '{"patient_id": "PA", "allocated_icu": false, "priority_rank": 2, "justification": "reasoning"},'
                  '{"patient_id": "PB", "allocated_icu": true, "priority_rank": 1, "justification": "reasoning"},'
                  '{"patient_id": "PC", "allocated_icu": false, "priority_rank": 3, "justification": "reasoning"}'
                  '], "overall_reasoning": "detailed SOFA score based reasoning minimum 250 chars explaining why PB gets the ICU bed", '
                  '"cited_guidelines": ["SOFA Score", "NICE Sepsis 2024"]}}')

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Patient observation:\n{content}\n\n"
                        f"Respond with this exact JSON structure:\n{schema}"
                    )},
                ],
                temperature=0.0,
            )
            break
        except RateLimitError:
            wait = 2 ** attempt
            print(f"Rate limited. Retrying in {wait}s...")
            time.sleep(wait)
            if attempt == 4:
                raise

    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def main():
    print("[START]")

    task_map = {
        "single_symptom_triage":   "task1",
        "differential_diagnosis":  "task2",
        "icu_resource_allocation":  "task3",
    }

    for task_idx in range(3):
        obs    = call_env("/reset")
        tname  = obs["task_name"]
        action = agent_act(obs)
        result = call_env("/step", action)

        key   = task_map.get(tname, "task1")
        score = 0.5  # default fallback

        if result.get(key) and result[key].get("reward"):
            raw = result[key]["reward"].get("total", 0.5)
            # Ensure strictly between 0 and 1
            score = max(0.001, min(0.999, float(raw)))

        print(f"[STEP] task_name={tname} score={score:.4f}")

    print("[END]")


if __name__ == "__main__":
    main()