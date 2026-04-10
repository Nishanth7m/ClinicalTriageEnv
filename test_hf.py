import time
import requests

SPACE_URL = "https://nishanthdev7-clinicaltriageenv.hf.space"

def wait_for_ready():
    print("Waiting for Space to be ready...")
    for i in range(30): # max 60 seconds
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

def test_task(task_name):
    print(f"\n--- Testing task: {task_name} ---")
    payload = {"task": task_name}
    r = requests.post(f"{SPACE_URL}/reset", json=payload)
    print(f"Status Code: {r.status_code}")
    try:
        print("Response JSON:")
        import json
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Failed to decode JSON:", e)

if wait_for_ready():
    test_task("single_symptom_triage")
    test_task("differential_diagnosis")
    test_task("icu_resource_allocation")
