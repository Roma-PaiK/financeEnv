"""
Baseline inference script — runs an OpenAI agent through all tasks.

Usage:
    OPENAI_API_KEY=sk-... python baseline/run_baseline.py

Produces deterministic scores: temperature=0, seed=42, static data.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from finance_env import FinanceEnv
from finance_env.models import Action

TASKS = [
    ("task1", "Transaction Categorisation", "Easy"),
    # ("task2", "Reconciliation",             "Medium"),  # not yet implemented
    # ("task3", "Budget Planning",             "Hard"),   # not yet implemented
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


def build_user_message(obs_dict: dict, last_feedback: str | None) -> str:
    lines = [f"task_id: {obs_dict['task_id']}"]
    lines.append(f"task_context: {obs_dict['task_context']}")
    lines.append(f"step_count: {obs_dict['step_count']}")
    lines.append(f"account_balance: ₹{obs_dict['account_balance']}")
    lines.append(f"monthly_income: ₹{obs_dict['monthly_income']}")
    lines.append(f"current_month: {obs_dict['current_month']}")
    lines.append(f"sources_present: {obs_dict['sources_present']}")
    lines.append("\ntransactions:")
    for t in obs_dict["transactions"]:
        lines.append(
            f"  [{t['id']}] {t['date']} | {t['source']} | {t['description']} | ₹{t['amount']}"
        )
    if last_feedback:
        lines.append(f"\nlast_feedback: {last_feedback}")
    return "\n".join(lines)


def run_task(client: OpenAI, env: FinanceEnv, task_id: str) -> float:
    obs = env.reset(task_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_feedback: str | None = None
    final_score = 0.0
    total_tokens_in = 0
    total_tokens_out = 0

    print(f"\n{'=' * 70}")
    print(f"  Running {task_id.upper()}  |  {len(obs.transactions)} transactions  |  max_steps={25}")
    print(f"{'=' * 70}")

    while True:
        obs_dict = obs.model_dump()
        user_msg = build_user_message(obs_dict, last_feedback)
        messages.append({"role": "user", "content": user_msg})

        print(f"\n  [step {obs.step_count + 1}] Calling OpenAI API...", end="", flush=True)

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
                    print(f"\n  Rate limit — retrying in {wait}s...", end="", flush=True)
                    time.sleep(wait)
                else:
                    raise

        usage = response.usage
        tokens_in  = usage.prompt_tokens
        tokens_out = usage.completion_tokens
        total_tokens_in  += tokens_in
        total_tokens_out += tokens_out

        raw_json = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw_json})

        print(f" done  (in={tokens_in} out={tokens_out})")
        print(f"  Model response: {raw_json}")

        try:
            action = Action(**json.loads(raw_json))
            print(f"  Parsed action : type={action.action_type}  payload={action.payload}  confidence={action.confidence}")
        except Exception as e:
            print(f"  Parse ERROR   : {e} — submitting penalty action")
            action = Action(
                action_type="categorize",
                payload={"transaction_id": "__invalid__", "category": "Other"},
                confidence=0.0,
            )

        print(f"  Stepping env  ...", end="", flush=True)
        obs, reward, done, _ = env.step(action)
        last_feedback = reward.feedback
        final_score = reward.cumulative_score

        print(
            f"\n  Result        : delta={reward.score:+.3f} | "
            f"cumulative={reward.cumulative_score:.3f} | "
            f"feedback=\"{reward.feedback}\""
        )
        if reward.partial_scores:
            print(f"  Partial scores: {reward.partial_scores}")

        if done:
            break

    print(f"\n{'=' * 70}")
    print(f"  EPISODE COMPLETE")
    print(f"  Final score   : {final_score:.3f} / 1.000")
    print(f"  Total tokens  : {total_tokens_in} in / {total_tokens_out} out  "
          f"(total={total_tokens_in + total_tokens_out})")
    print(f"{'=' * 70}")

    return final_score


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...")

    client = OpenAI(api_key=api_key)
    env = FinanceEnv()
    results = []

    for task_id, description, difficulty in TASKS:
        score = run_task(client, env, task_id)
        results.append((task_id, description, difficulty, score))

    print(f"\n{'=' * 70}")
    print(f"  FINAL SCORES")
    print(f"  {'Task':<8} {'Description':<30} {'Difficulty':<12} {'Score':>6}")
    print(f"  {'-' * 60}")
    for task_id, desc, diff, score in results:
        print(f"  {task_id:<8} {desc:<30} {diff:<12} {score:>6.3f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
