"""
run_baseline.py — runs an LLM agent through all tasks by calling env.step() directly (no HTTP server needed).

Supports two LLM providers:
    openai (default) — uses OPENAI_API_KEY or API_KEY, temperature=0, seed=42
    ollama           — uses OLLAMA_BASE_URL + OLLAMA_MODEL, no seed control

Env vars:
    LLM_PROVIDER     "openai" (default) or "ollama"
    OPENAI_API_KEY   Required for openai provider
    API_KEY          Alternative key name (hackathon proxy)
    API_BASE_URL     Override base URL (hackathon proxy injects this)
    OLLAMA_BASE_URL  Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL     Ollama model name (default: qwen2.5:7b)
    TASK             Run a single task: task1 | task2 | task3 (default: all)
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

LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

ALL_TASKS = [
    ("task1", "Transaction Categorisation", "Easy"),
    ("task2", "Reconciliation",             "Medium"),
    ("task3", "Budget Planning",            "Hard"),
]

_task_filter = os.getenv("TASK")
TASKS = [t for t in ALL_TASKS if _task_filter is None or t[0] == _task_filter]
if not TASKS:
    raise ValueError(f"TASK={_task_filter!r} is not valid. Choose from: task1, task2, task3")

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
  "finalize": { "reconciled_totals": { "2024-01": { "Food & Dining": float, ... }, ... }, "excluded_ids": [<ids>] }

=== TASK 3: Budget Planning ===
Build a realistic monthly budget for all 9 categories. Income: ₹85,000. Savings goal: ₹8,000.
  "query": { "category": "<category>", "months": ["2024-01", "2024-02"] }
  "set_budget": { "category": "<category>", "amount": <float> }
  "finalize": { "budget": { "Food & Dining": float, "Transport & Commute": float, "Utilities & Bills": float,
                             "EMI & Loan Repayment": float, "Entertainment & Subscriptions": float,
                             "Healthcare": float, "Shopping & Apparel": float,
                             "Savings & Investment": float, "Other": float } }
Rules: budget sum MUST be <= 85000. All 9 categories required.

Respond ONLY with a valid JSON object. No prose, no markdown fences.
"""


def build_user_message(obs_dict: dict, last_feedback: str | None) -> str:
    lines = [
        f"task_id: {obs_dict['task_id']}",
        f"task_context: {obs_dict['task_context']}",
        f"step_count: {obs_dict['step_count']}",
        f"account_balance: ₹{obs_dict['account_balance']}",
        f"monthly_income: ₹{obs_dict['monthly_income']}",
        f"current_month: {obs_dict['current_month']}",
        f"sources_present: {obs_dict['sources_present']}",
        "\ntransactions:",
    ]
    for t in obs_dict["transactions"]:
        lines.append(f"  [{t['id']}] {t['date']} | {t['source']} | {t['description']} | ₹{t['amount']}")
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
    print(f"  {task_id.upper()}  |  {len(obs.transactions)} transactions  |  max_steps={max_steps}  |  {provider.upper()} / {model}")
    print(f"{'=' * 70}")

    while True:
        obs_dict = obs.model_dump()
        messages.append({"role": "user", "content": build_user_message(obs_dict, last_feedback)})

        print(f"\n  [step {obs.step_count + 1}] Calling LLM...", end="", flush=True)

        for attempt in range(5):
            try:
                kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
                if provider == "openai":
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["seed"] = 42
                response = client.chat.completions.create(**kwargs)
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
        print(f" done  (in={tokens_in} out={tokens_out})")
        print(f"  Response: {raw_json}")

        try:
            action = Action(**json.loads(raw_json))
            print(f"  Action  : type={action.action_type}  payload={action.payload}  confidence={action.confidence}")
        except Exception as e:
            print(f"  Parse ERROR: {e} — using penalty action")
            action = Action(action_type="categorize", payload={"transaction_id": "__invalid__", "category": "Other"}, confidence=0.0)

        obs, reward, done, _ = env.step(action)
        last_feedback = reward.feedback
        final_score = reward.cumulative_score

        print(f"  Result  : delta={reward.score:+.3f} | cumulative={reward.cumulative_score:.3f} | \"{reward.feedback}\"")
        if reward.partial_scores:
            print(f"  Partial : {reward.partial_scores}")

        if done:
            break

    print(f"\n{'=' * 70}")
    print(f"  DONE  score={final_score:.3f}  tokens={total_tokens_in}in/{total_tokens_out}out")
    print(f"{'=' * 70}")
    return final_score


def main() -> None:
    if LLM_PROVIDER == "openai":
        api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key. Set API_KEY (proxy) or OPENAI_API_KEY.")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        api_base_url = os.environ.get("API_BASE_URL")
        if api_base_url:
            client_kwargs["base_url"] = api_base_url
        client = OpenAI(**client_kwargs)
        model = "gpt-4o"
        print(f"Provider: OpenAI  model={model}  base_url={api_base_url or 'default'}")

    elif LLM_PROVIDER == "ollama":
        client = OpenAI(api_key="dummy", base_url=OLLAMA_BASE_URL.rstrip("/") + "/v1")
        model = OLLAMA_MODEL
        print(f"Provider: Ollama  base_url={OLLAMA_BASE_URL}  model={model}")
        try:
            client.models.list()
        except Exception as e:
            raise ValueError(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}")

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
