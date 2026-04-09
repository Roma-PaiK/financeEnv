"""
Baseline inference script — runs an agent through all tasks.

Supports two LLM providers:
  1. OpenAI (default, for submission)
  2. Ollama (local development)

Usage (OpenAI — submission):
    OPENAI_API_KEY=sk-... python baseline/run_baseline.py

Usage (Ollama — local dev):
    LLM_PROVIDER=ollama OLLAMA_MODEL=mistral python baseline/run_baseline.py

Environment variables:
    LLM_PROVIDER        "openai" (default) or "ollama"
    OPENAI_API_KEY      Required if LLM_PROVIDER=openai
    OLLAMA_BASE_URL     Base URL for Ollama (default: http://localhost:11434)
    OLLAMA_MODEL        Model name for Ollama (default: mistral)

Produces deterministic scores with OpenAI: temperature=0, seed=42, static data.
Note: Ollama may produce different results due to lack of seed control.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from finance_env import FinanceEnv
from finance_env.models import Action

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

TASKS = [
    ("task1", "Transaction Categorisation", "Easy"),
    ("task2", "Reconciliation",             "Medium"),
    ("task3", "Budget Planning",            "Hard"),
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

=== TASK 1: Categorisation ===
Use action_type "categorize" with payload: { "transaction_id": "<id>", "category": "<category>" }
Valid categories (use exactly):
  "Food & Dining" | "Transport & Commute" | "Utilities & Bills" | "EMI & Loan Repayment"
  "Entertainment & Subscriptions" | "Healthcare" | "Shopping & Apparel" | "Savings & Investment" | "Other"
Then submit "finalize" with empty payload {}.

=== TASK 2: Reconciliation ===
Identify duplicate/settlement rows. Use:
  "reconcile": { "transaction_id": "<id>", "classification": "genuine_spend" | "cc_settlement" | "internal_transfer" | "refund" }
  "query": { "query_type": "merchant" | "date_range", "value": "<merchant_substring or YYYY-MM-DD:YYYY-MM-DD>" }
  "finalize": { "reconciled_totals": { "2024-01": { "Food & Dining": float, ... }, ... }, "excluded_ids": [<cc_settlement_ids>] }

=== TASK 3: Budget Planning ===
You are given 2 months of pre-reconciled spend history. Build a realistic monthly budget for all 9 categories.
Goals: Save ₹8,000. Reduce Food & Dining by 15% vs last month. Fixed EMI: ₹12,000. Income: ₹85,000.
Strategy: First use "query" to explore each category's historical spend, then "set_budget" for each category, then "finalize".
  "query": { "category": "<one of the 9 categories>", "months": ["2024-01", "2024-02"] }
  "set_budget": { "category": "<category>", "amount": <float> }
  "finalize": { "budget": { "Food & Dining": float, "Transport & Commute": float, "Utilities & Bills": float,
                             "EMI & Loan Repayment": float, "Entertainment & Subscriptions": float,
                             "Healthcare": float, "Shopping & Apparel": float,
                             "Savings & Investment": float, "Other": float } }
Rules: budget sum MUST be <= 85000. All 9 categories required. Set "Other" with a buffer for surprises.

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


def run_task(client: OpenAI, env: FinanceEnv, task_id: str, provider: str, model: str) -> float:
    obs = env.reset(task_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_feedback: str | None = None
    final_score = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    max_steps = {"task1": 2, "task2": 3, "task3": 4}.get(task_id, 25)

    print(f"\n{'=' * 70}")
    print(f"  Running {task_id.upper()}  |  {len(obs.transactions)} transactions  |  max_steps={max_steps}")
    print(f"  LLM: {provider.upper()} / {model}")
    print(f"{'=' * 70}")

    while True:
        obs_dict = obs.model_dump()
        user_msg = build_user_message(obs_dict, last_feedback)
        messages.append({"role": "user", "content": user_msg})

        provider_name = "Ollama" if provider == "ollama" else "OpenAI"
        print(f"\n  [step {obs.step_count + 1}] Calling {provider_name}...", end="", flush=True)

        for attempt in range(5):
            try:
                # Build request kwargs based on provider
                request_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                }

                # OpenAI-specific options
                if provider == "openai":
                    request_kwargs["response_format"] = {"type": "json_object"}
                    request_kwargs["seed"] = 42

                response = client.chat.completions.create(**request_kwargs)
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 10 * (attempt + 1)
                    print(f"\n  Rate limit — retrying in {wait}s...", end="", flush=True)
                    time.sleep(wait)
                else:
                    raise

        usage = response.usage
        tokens_in  = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        total_tokens_in  += tokens_in
        total_tokens_out += tokens_out

        raw_json = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw_json})

        token_info = f"in={tokens_in} out={tokens_out}" if tokens_in or tokens_out else "no token count"
        print(f" done  ({token_info})")
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
    if total_tokens_in or total_tokens_out:
        print(f"  Total tokens  : {total_tokens_in} in / {total_tokens_out} out  "
              f"(total={total_tokens_in + total_tokens_out})")
    print(f"{'=' * 70}")

    return final_score


def main() -> None:
    # Validate and initialize LLM client
    if LLM_PROVIDER == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set.\n"
                "Run: export OPENAI_API_KEY=sk-... && python baseline/run_baseline.py"
            )
        client = OpenAI(api_key=api_key)
        model = "gpt-4o"
        print(f"Using OpenAI API (model: {model})")

    elif LLM_PROVIDER == "ollama":
        # Ollama uses OpenAI-compatible API
        client = OpenAI(
            api_key="dummy",  # Ollama doesn't require a real key
            base_url=OLLAMA_BASE_URL.rstrip("/") + "/v1",
        )
        model = OLLAMA_MODEL
        print(f"Using Ollama (base_url: {OLLAMA_BASE_URL}, model: {model})")
        # Test connection
        try:
            client.models.list()
        except Exception as e:
            raise ValueError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                f"Make sure Ollama is running: {e}"
            )

    else:
        raise ValueError(f"LLM_PROVIDER must be 'openai' or 'ollama', got: {LLM_PROVIDER}")

    env = FinanceEnv()
    results = []

    for task_id, description, difficulty in TASKS:
        score = run_task(client, env, task_id, LLM_PROVIDER, model)
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
