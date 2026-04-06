"""
Deploy FinanceEnv to HuggingFace Spaces as a Docker space.

Usage:
    python scripts/deploy_hf.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from huggingface_hub import HfApi, whoami

SPACE_NAME = "finance_env_india"
REPO_ROOT  = Path(__file__).parent.parent

# Files/dirs to include (everything not in .gitignore)
IGNORE = {".venv", "__pycache__", ".env", ".git", "*.pyc", "baseline/results.json"}

def should_ignore(path: Path) -> bool:
    for pattern in IGNORE:
        if pattern.startswith("*"):
            if path.name.endswith(pattern[1:]):
                return True
        elif path.name == pattern or any(p == pattern for p in path.parts):
            return True
    return False


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in .env")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = whoami(token=token)
    username = user_info["name"]
    repo_id = f"{username}/{SPACE_NAME}"

    print(f"Logged in as : {username}")
    print(f"Space        : {repo_id}")

    # Create the space (Docker SDK)
    print("\nCreating Space (Docker)...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )
    print(f"Space ready  : https://huggingface.co/spaces/{repo_id}")

    # Upload the entire repo folder, excluding ignored items
    print("\nUploading files...")
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            ".env",
            ".git/**",
            "baseline/results.json",
            "scripts/deploy_hf.py",
        ],
        commit_message="Deploy FinanceEnv Task 1",
    )

    print("\n" + "=" * 60)
    print("  DEPLOYED SUCCESSFULLY")
    print(f"  URL : https://huggingface.co/spaces/{repo_id}")
    print("  Note: Space build takes ~2-3 mins. Check logs in HF UI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
