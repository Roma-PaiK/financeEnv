"""
End-to-end HTTP test for Task 1 — hits the live FastAPI server.

Usage:
    # In one terminal: uvicorn app:app --port 7860
    # In another:      python baseline/test_e2e_task1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import urllib.request
import urllib.error

BASE_URL = "http://localhost:7860"


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}") as resp:
        return json.loads(resp.read())


def check_server() -> bool:
    try:
        get("/health")
        return True
    except Exception:
        return False


def main() -> None:
    print("=" * 65)
    print("  FinanceEnv — Task 1 End-to-End HTTP Test")
    print("=" * 65)

    if not check_server():
        print("\n  ERROR: Server not reachable at http://localhost:7860")
        print("  Start it with:  .venv/bin/uvicorn app:app --port 7860")
        sys.exit(1)

    print("\n  [1] POST /reset ...")
    obs = post("/reset", {"task_id": "task1"})
    print(f"      task_id        : {obs['task_id']}")
    print(f"      current_month  : {obs['current_month']}")
    print(f"      transactions   : {len(obs['transactions'])}")
    print(f"      sources        : {obs['sources_present']}")
    print(f"      hidden fields  : correct_category={obs['transactions'][0]['correct_category']}")
    assert len(obs["transactions"]) == 20, "Expected 20 transactions"
    assert obs["transactions"][0]["correct_category"] is None, "Hidden field not stripped!"

    print("\n  [2] POST /step — correct categorize ...")
    step = post("/step", {
        "action_type": "categorize",
        "payload": {"transaction_id": "txn_sbi_001", "category": "Savings & Investment"},
        "confidence": 0.9,
    })
    print(f"      feedback       : {step['reward']['feedback']}")
    print(f"      score delta    : {step['reward']['score']}")
    print(f"      cumulative     : {step['reward']['cumulative_score']}")
    print(f"      done           : {step['done']}")
    assert step["reward"]["score"] == 0.04, f"Expected +0.04, got {step['reward']['score']}"

    print("\n  [3] POST /step — wrong category ...")
    step = post("/step", {
        "action_type": "categorize",
        "payload": {"transaction_id": "txn_sbi_002", "category": "Food & Dining"},
        "confidence": 0.5,
    })
    print(f"      feedback       : {step['reward']['feedback']}")
    print(f"      score delta    : {step['reward']['score']}")
    assert step["reward"]["score"] == 0.0, f"Expected 0.0, got {step['reward']['score']}"

    print("\n  [4] POST /step — near-miss ...")
    step = post("/step", {
        "action_type": "categorize",
        "payload": {"transaction_id": "txn_sbi_003", "category": "Utilities & Bills"},
        "confidence": 0.6,
    })
    print(f"      feedback       : {step['reward']['feedback']}")
    print(f"      score delta    : {step['reward']['score']}")
    assert step["reward"]["score"] == 0.015, f"Expected +0.015, got {step['reward']['score']}"

    print("\n  [5] POST /step — invalid category string ...")
    step = post("/step", {
        "action_type": "categorize",
        "payload": {"transaction_id": "txn_sbi_004", "category": "Groceries"},
        "confidence": 0.5,
    })
    print(f"      feedback       : {step['reward']['feedback']}")
    print(f"      score delta    : {step['reward']['score']}")
    assert step["reward"]["score"] == -0.01, f"Expected -0.01, got {step['reward']['score']}"

    print("\n  [6] POST /step — illegal action type for task1 ...")
    step = post("/step", {
        "action_type": "reconcile",
        "payload": {},
        "confidence": 0.5,
    })
    print(f"      feedback       : {step['reward']['feedback']}")
    print(f"      score delta    : {step['reward']['score']}")
    assert step["reward"]["score"] == -0.02, f"Expected -0.02, got {step['reward']['score']}"

    print("\n  [7] GET /state ...")
    state = get("/state")
    print(f"      step_count     : {state['step_count']}")
    print(f"      cumulative     : {state['cumulative_score']}")
    print(f"      addressed_ids  : {state['addressed_ids']}")

    print("\n  [8] POST /step — finalize ...")
    step = post("/step", {
        "action_type": "finalize",
        "payload": {},
        "confidence": 1.0,
    })
    print(f"      feedback       : {step['reward']['feedback']}")
    print(f"      score delta    : {step['reward']['score']}")
    print(f"      done           : {step['done']}")
    assert step["done"] is True, "Expected done=True after finalize"

    print("\n" + "=" * 65)
    print("  ALL CHECKS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
