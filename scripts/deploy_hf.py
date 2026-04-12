"""
deploy_hf.py — uploads this repo to HuggingFace Spaces as a Docker space.

Reads HF_TOKEN from .env or environment. Creates the space if it doesn't exist.

Usage:
    HF_TOKEN=hf_... .venv/bin/python scripts/deploy_hf.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from huggingface_hub import HfApi, whoami

SPACE_NAME = "finance_env_india"
REPO_ROOT  = Path(__file__).parent.parent

# Patterns excluded from upload
IGNORE_PATTERNS = [
    ".venv/**", "**/__pycache__/**", "**/*.pyc",
    ".env", ".git/**", "baseline/results.json", "scripts/deploy_hf.py",
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    api = HfApi(token=token)
    username = whoami(token=token)["name"]
    repo_id = f"{username}/{SPACE_NAME}"

    print(f"Logged in as : {username}")
    print(f"Space        : {repo_id}")

    print("\nCreating Space (Docker)...")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    print(f"Ready: https://huggingface.co/spaces/{repo_id}")

    print("\nUploading...")
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Deploy FinanceEnv",
    )

    print(f"\nDeployed: https://huggingface.co/spaces/{repo_id}")
    print("Note: Space build takes ~2-3 mins. Check logs in HF UI.")


if __name__ == "__main__":
    main()
