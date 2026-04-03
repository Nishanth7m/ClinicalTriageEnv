"""
Baseline inference script.
Runs the agent against all 3 tasks and prints scores.
Must complete in < 20 min on 2 vCPU / 8 GB RAM.
"""
import os, json
from openai import OpenAI
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""),
                base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert emergency medicine physician.
You will receive a patient observation in JSON format.
Respond ONLY with a valid JSON object matching the required action schema.
Do not add any prose outside the JSON."""

def call_env(endpoint: str, payload: dict = None):
    if payload:
        r = requests.post(f"{ENV_URL}{endpoint}", json=payload, timeout=60)
    else:
        r = requests.post(f"{ENV_URL}{endpoint}", timeout=60)
    r.raise_for_status()
    return r.json()

def agent_act(obs: dict) -> dict:
    task_name = obs.get("task_name", "")
    content   = json.dumps(obs, indent=2)

    if task_name == "single_symptom_triage":
        schema = ('{"task1": {"triage_level": "emergent|urgent|non_urgent", '
                  '"rationale": "...", "cited_guideline": "optional"}}')
    elif task_name == "differential_diagnosis":
        schema = ('{"task2": {"primary_diagnosis": "...", '
                  '"differential_diagnoses": ["..."], '
                  '"recommended_actions": ["..."], '
                  '"triage_level": "emergent|urgent|non_urgent", '
                  '"reasoning": "...", "cited_guidelines": []}}')
    else:
        schema = ('{"task3": {"allocation_decisions": ['
                  '{"patient_id": "...", "allocated_icu": true/false, '
                  '"priority_rank": 1, "justification": "..."}], '
                  '"overall_reasoning": "...", "cited_guidelines": []}}')

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"Patient observation:\n{content}\n\n"
                f"Respond with this exact JSON structure:\n{schema}"
            )},
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)

def main():
    print("=== ClinicalTriageEnv Baseline Inference ===\n")
    scores = {}

    obs = call_env("/reset")
    for task_idx in range(3):
        print(f"--- Task {task_idx + 1} ---")
        print(f"Task: {obs['task_name']}")

        action = agent_act(obs)
        result = call_env("/step", action)

        # Extract reward from whichever task field is populated
        for key in ["task1", "task2", "task3"]:
            if result.get(key) and result[key].get("reward"):
                score = result[key]["reward"]["total"]
                scores[obs["task_name"]] = score
                feedback = result[key].get("feedback", "")
                print(f"Score: {score:.2f}  |  {feedback}")
                break
        
        obs = result  # The result from /step is the next observation

    print("\n=== Final Scores ===")
    for task, score in scores.items():
        print(f"  {task}: {score:.2f}")
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"  Average: {avg:.2f}")

if __name__ == "__main__":
    main()
