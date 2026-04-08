from huggingface_hub import HfApi
import sys
import os

repo_id = "NishanthDev7/ClinicalTriageEnv"
local_path = "."

api = HfApi()

print(f"Uploading {local_path} to {repo_id}...")
try:
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_path,
        repo_type="space",
        ignore_patterns=[".git/*", ".venv/*", "__pycache__/*", "*.pyc", ".env"]
    )
    print("Upload successful!")
except Exception as e:
    print(f"Error during upload: {e}", file=sys.stderr)
    sys.exit(1)
