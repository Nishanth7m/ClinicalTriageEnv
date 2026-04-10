import time
import requests
import json

SPACE_URL = "https://nishanthdev7-clinicaltriageenv.hf.space"

def wait_for_ready():
    print("Waiting for Space to be ready...")
    for i in range(30):
        try:
            r = requests.get(f"{SPACE_URL}/health", timeout=5)
            if r.status_code == 200:
                print("Space is ready!")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("Space did not become ready in time.")
    return False

def test_grade(endpoint, payload):
    print(f"\nTest: POST {SPACE_URL}/grade/{endpoint}")
    r = requests.post(f"{SPACE_URL}/grade/{endpoint}", json=payload)
    print(f"Status Code: {r.status_code}")
    print("Response:")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Failed to parse JSON:", r.text)

if wait_for_ready():
    test_grade("single_symptom_triage", {"task1": {"triage_level": "urgent", "rationale": "test rationale here"}})
    test_grade("differential_diagnosis", {"task2": {"primary_diagnosis": "pneumonia", "differential_diagnoses": ["pneumonia"], "recommended_actions": ["xray"], "triage_level": "urgent", "reasoning": "test reasoning here"}})
    test_grade("icu_resource_allocation", {"task3": {"allocation_decisions": [{"patient_id": "PB", "allocated_icu": True, "priority_rank": 1, "justification": "critical"}], "overall_reasoning": "PB is most critical"}})
