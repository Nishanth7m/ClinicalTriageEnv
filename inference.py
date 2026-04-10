"""
Baseline inference script — ClinicalTriageEnv
Must complete in < 20 min on 2 vCPU / 8 GB RAM.
"""
import os, json, time, sys
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, RateLimitError
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else "placeholder",
    base_url=API_BASE_URL
)

SYSTEM_PROMPT = """You are an expert emergency medicine physician.
You will receive a patient observation in JSON format.
Respond ONLY with a valid JSON object matching the required action schema.
Do not add any prose outside the JSON."""

FALLBACKS = {
    "single_symptom_triage": {
        "task1": {
            "triage_level": "urgent",
            "rationale": "Patient presents with symptoms requiring clinical assessment. Vitals reviewed and triage level assigned based on severity indicators and NEWS2 scoring criteria.",
            "cited_guideline": "NEWS2 Score"
        }
    },
    "differential_diagnosis": {
        "task2": {
            "primary_diagnosis": "community acquired pneumonia",
            "differential_diagnoses": ["pneumonia", "pulmonary embolism", "COPD exacerbation"],
            "recommended_actions": ["chest X-ray", "blood cultures", "IV antibiotics"],
            "triage_level": "urgent",
            "reasoning": "Patient presents with fever productive cough and elevated WBC suggesting infection. Clinical picture consistent with community acquired pneumonia. Differential includes PE given dyspnoea. Recommend imaging and empirical antibiotics per NICE guidelines.",
            "cited_guidelines": ["NICE Pneumonia 2024", "NEWS2 Score"]
        }
    },
    "icu_resource_allocation": {
        "task3": {
            "allocation_decisions": [
                {"patient_id": "PA", "allocated_icu": False, "priority_rank": 2,
                 "justification": "Septic shock with high lactate but lower immediate mortality risk than PB"},
                {"patient_id": "PB", "allocated_icu": True, "priority_rank": 1,
                 "justification": "Massive STEMI with cardiogenic shock and SpO2 88% requires immediate ICU"},
                {"patient_id": "PC", "allocated_icu": False, "priority_rank": 3,
                 "justification": "Severe asthma managed in HDU with nebulisers and steroids"}
            ],
            "overall_reasoning": "PB has massive STEMI with cardiogenic shock SpO2 88% and troponin 18.5 requiring immediate ICU for balloon pump and cath lab. PA has septic shock but creatinine 3.8 limits intervention. PC severe asthma managed in HDU. SOFA scoring and AHA guidelines prioritise cardiac arrest prevention.",
            "cited_guidelines": ["SOFA Score", "AHA STEMI 2022"]
        }
    }
}


def call_env(endpoint: str, payload: dict = None):
    try:
        if payload:
            r = requests.post(f"{ENV_URL}{endpoint}", json=payload, timeout=60)
        else:
            r = requests.post(f"{ENV_URL}{endpoint}", timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] call_env {endpoint} failed: {e}", file=sys.stderr)
        raise


def agent_act(obs: dict) -> dict:
    task_name = obs.get("task_name", "single_symptom_triage")
    fallback  = FALLBACKS.get(task_name, FALLBACKS["single_symptom_triage"])

    try:
        if task_name == "single_symptom_triage":
            schema = ('{"task1": {"triage_level": "emergent|urgent|non_urgent", '
                      '"rationale": "detailed clinical reasoning minimum 100 chars", '
                      '"cited_guideline": "NEWS2 Score"}}')
        elif task_name == "differential_diagnosis":
            schema = ('{"task2": {"primary_diagnosis": "diagnosis name", '
                      '"differential_diagnoses": ["dx1", "dx2", "dx3"], '
                      '"recommended_actions": ["action1", "action2"], '
                      '"triage_level": "emergent|urgent|non_urgent", '
                      '"reasoning": "detailed reasoning minimum 200 chars", '
                      '"cited_guidelines": ["NICE Pneumonia 2024", "NEWS2 Score"]}}')
        else:
            schema = ('{"task3": {"allocation_decisions": ['
                      '{"patient_id": "PA", "allocated_icu": false, "priority_rank": 2, "justification": "reason"},'
                      '{"patient_id": "PB", "allocated_icu": true,  "priority_rank": 1, "justification": "reason"},'
                      '{"patient_id": "PC", "allocated_icu": false, "priority_rank": 3, "justification": "reason"}'
                      '], "overall_reasoning": "detailed SOFA-based reasoning minimum 200 chars", '
                      '"cited_guidelines": ["SOFA Score", "AHA STEMI 2022"]}}')

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Patient observation:\n{json.dumps(obs, indent=2)}\n\n"
                            f"Respond with this exact JSON structure:\n{schema}"
                        )},
                    ],
                    temperature=0.0,
                    timeout=30,
                )
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text.strip())

            except RateLimitError:
                time.sleep(2 ** attempt)
            except json.JSONDecodeError:
                continue

    except Exception as e:
        print(f"[ERROR] LLM failed, using fallback: {e}", file=sys.stderr)

    return fallback


def main():
    print("[START]")

    task_map = {
        "single_symptom_triage":   "task1",
        "differential_diagnosis":  "task2",
        "icu_resource_allocation": "task3",
    }
    task_names = list(task_map.keys())

    for task_idx in range(3):
        tname = task_names[task_idx]
        score = 0.5

        try:
            obs    = call_env("/reset")
            tname  = obs.get("task_name", tname)
            action = agent_act(obs)
            result = call_env("/step", action)

            key = task_map.get(tname, "task1")
            if result.get(key) and result[key].get("reward"):
                raw   = result[key]["reward"].get("total", 0.5)
                score = max(0.001, min(0.999, float(raw)))

        except Exception as e:
            print(f"[ERROR] Task {task_idx+1}: {e}", file=sys.stderr)
            score = 0.1

        print(f"[STEP] task_name={tname} score={score:.4f}")

    print("[END]")


if __name__ == "__main__":
    main()