"""
Baseline inference script — agent calls the LIVE HuggingFace Space endpoints.

Usage:
    python baseline/run_baseline_http.py

Requires OPENAI_API_KEY and HF_TOKEN in .env (or env vars).
Space URL: https://romapai-finance-env-india.hf.space
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI

SPACE_URL = "https://romapai-finance-env-india.hf.space"

TASKS = [
    ("task1", "Transaction Categorisation", "Easy"),
]

SYSTEM_PROMPT = """\
You are a financial intelligence agent operating in the FinanceEnv environment.
At each step you receive an observation and must respond with a single JSON action.

Action schema:
{
  "action_type": "categorize" | "reconcile" | "query" | "set_budget" | "finalize",
  "payload": { ... },
  "confidence": 0.0–1.0
}

For Task 1, use action_type "categorize" with payload:
  { "transaction_id": "<id>", "category": "<one of the 9 categories below>" }

The ONLY valid category strings are (use exactly as written):
  "Food & Dining"
  "Transport & Commute"
  "Utilities & Bills"
  "EMI & Loan Repayment"
  "Entertainment & Subscriptions"
  "Healthcare"
  "Shopping & Apparel"
  "Savings & Investment"
  "Other"

When all transactions are categorized, submit action_type "finalize" with an empty payload {}.

Respond ONLY with a valid JSON object. No prose, no markdown fences.
"""


# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def http_post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{SPACE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def http_get(path: str) -> dict:
    with urllib.request.urlopen(f"{SPACE_URL}{path}", timeout=30) as resp:
        return json.loads(resp.read())


def build_user_message(obs: dict, last_feedback: str | None) -> str:
    lines = [
        f"task_id: {obs['task_id']}",
        f"task_context: {obs['task_context']}",
        f"step_count: {obs['step_count']}",
        f"account_balance: ₹{obs['account_balance']}",
        f"monthly_income: ₹{obs['monthly_income']}",
        f"current_month: {obs['current_month']}",
        f"sources_present: {obs['sources_present']}",
        "\ntransactions:",
    ]
    for t in obs["transactions"]:
        lines.append(
            f"  [{t['id']}] {t['date']} | {t['source']} | {t['description']} | ₹{t['amount']}"
        )
    if last_feedback:
        lines.append(f"\nlast_feedback: {last_feedback}")
    return "\n".join(lines)


def run_task(client: OpenAI, task_id: str) -> float:
    print(f"\n{'=' * 70}")
    print(f"  Resetting {task_id.upper()} via POST {SPACE_URL}/reset ...")

    obs = http_post("/reset", {"task_id": task_id})
    print(f"  Transactions : {len(obs['transactions'])} | Month: {obs['current_month']} | Sources: {obs['sources_present']}")
    print(f"{'=' * 70}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_feedback: str | None = None
    final_score = 0.0
    total_in = total_out = 0

    while True:
        user_msg = build_user_message(obs, last_feedback)
        messages.append({"role": "user", "content": user_msg})

        step_num = obs["step_count"] + 1
        print(f"\n  [step {step_num:>2}] Calling OpenAI...", end="", flush=True)

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    seed=42,
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 10 * (attempt + 1)
                    print(f" rate limit, retrying in {wait}s...", end="", flush=True)
                    time.sleep(wait)
                else:
                    raise

        tokens_in  = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        total_in  += tokens_in
        total_out += tokens_out
        raw_json = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw_json})

        print(f" done (in={tokens_in} out={tokens_out})")
        print(f"          Model  : {raw_json}")

        try:
            action = json.loads(raw_json)
            # Validate required fields
            assert "action_type" in action and "payload" in action and "confidence" in action
        except Exception as e:
            print(f"          Parse ERROR: {e} — sending finalize fallback")
            action = {"action_type": "finalize", "payload": {}, "confidence": 0.0}

        print(f"          Sending to {SPACE_URL}/step ...", end="", flush=True)
        result = http_post("/step", action)
        reward = result["reward"]
        obs = result["observation"]
        done = result["done"]

        last_feedback = reward["feedback"]
        final_score   = reward["cumulative_score"]

        print(
            f"\n          Result  : delta={reward['score']:+.3f} | "
            f"cumulative={reward['cumulative_score']:.3f} | "
            f"\"{reward['feedback']}\""
        )
        if reward.get("partial_scores"):
            print(f"          Partial : {reward['partial_scores']}")

        if done:
            break

    print(f"\n{'=' * 70}")
    print(f"  EPISODE COMPLETE")
    print(f"  Final score  : {final_score:.3f} / 1.000")
    print(f"  Total tokens : {total_in} in / {total_out} out (total={total_in + total_out})")
    print(f"{'=' * 70}")
    return final_score


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    print(f"Space URL : {SPACE_URL}")

    # Verify space is reachable
    try:
        health = http_get("/health")
        print(f"Health    : {health}")
    except Exception as e:
        print(f"ERROR: Space not reachable — {e}")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    results = []

    for task_id, description, difficulty in TASKS:
        score = run_task(client, task_id)
        results.append((task_id, description, difficulty, score))

    # Save results
    results_path = Path(__file__).parent / "results.json"
    results_path.write_text(json.dumps(
        [{"task_id": t, "description": d, "difficulty": diff, "score": s}
         for t, d, diff, s in results],
        indent=2,
    ))

    print(f"\n{'=' * 70}")
    print(f"  FINAL SCORES")
    print(f"  {'Task':<8} {'Description':<30} {'Difficulty':<12} {'Score':>6}")
    print(f"  {'-' * 60}")
    for task_id, desc, diff, score in results:
        print(f"  {task_id:<8} {desc:<30} {diff:<12} {score:>6.3f}")
    print(f"  Results saved to baseline/results.json")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
