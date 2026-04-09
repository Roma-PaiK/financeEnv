"""
inference.py — FinanceEnv OpenEnv Submission
=============================================
MANDATORY environment variables:
    API_BASE_URL   LLM endpoint  (default: OpenAI)
    MODEL_NAME     Model to use  (default: gpt-4o)
    HF_TOKEN       API key for the LLM
    SPACE_URL      HF Space URL  (default: https://romapai-finance-env-india.hf.space)
    TASK_NAME      Which task to run: task1 | task2 | task3  (default: task1)

STDOUT FORMAT (mandatory, do not alter):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Load .env if present (local dev convenience — not needed on HF Spaces)
# ---------------------------------------------------------------------------
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Mandatory for OpenAI API calls
HF_TOKEN     = os.getenv("HF_TOKEN")     # Optional: for HF Space auth if needed
SPACE_URL    = os.getenv("SPACE_URL",    "https://romapai-finance-env-india.hf.space").rstrip("/")
TASK_NAME    = os.getenv("TASK_NAME",    "task1")
BENCHMARK    = "finance_env_india"

SUCCESS_SCORE_THRESHOLD = 0.5  # cumulative_score >= this → success

# Max steps per task (mirrors env.py MAX_STEPS)
MAX_STEPS = {"task1": 2, "task2": 3, "task3": 4}

# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

# Task 1 — Categorise 20 single-source SBI transactions
SYSTEM_PROMPT_TASK1 = textwrap.dedent("""
    You are a financial intelligence agent for FinanceEnv Task 1.
    You will receive 20 SBI Savings transactions from January 2024.
    Your job: assign each transaction to exactly one of the 9 canonical categories below.

    For each transaction use action_type "categorize":
        {"action_type": "categorize", "payload": {"transaction_id": "<id>", "category": "<category>"}, "confidence": 0.0-1.0}

    After all transactions are categorized, send:
        {"action_type": "finalize", "payload": {}, "confidence": 1.0}

    VALID CATEGORIES (use exactly as written — any other string is penalised):
        "Food & Dining"
        "Transport & Commute"
        "Utilities & Bills"
        "EMI & Loan Repayment"
        "Entertainment & Subscriptions"
        "Healthcare"
        "Shopping & Apparel"
        "Savings & Investment"
        "Other"

    Rules:
    - Do NOT re-submit a transaction you already categorized (duplicate penalty: -0.020).
    - Do NOT invent category names (invalid string penalty: -0.010).
    - Categorize one transaction per step.
    - Respond ONLY with a valid JSON object. No prose, no markdown.
""").strip()

# Task 2 — Cross-source deduplication & reconciliation
SYSTEM_PROMPT_TASK2 = textwrap.dedent("""
    You are a financial intelligence agent for FinanceEnv Task 2: Cross-Source Reconciliation.

    OBJECTIVE:
    You are given 3 months of transactions across 3 sources: HDFC Credit Card, SBI Savings, Paytm UPI.
    Identify duplicate rows (primarily: SBI rows that are CC bill payments, which settle HDFC CC charges).
    Classify each transaction correctly and submit reconciled monthly spend totals per category.

    THE CORE PROBLEM:
    When the SBI account pays the HDFC CC bill (e.g., "HDFC CREDIT CARD PAYMENT ₹18,450" in SBI),
    this row is NOT a spend event — it's the settlement of previous CC charges already in the HDFC feed.
    Counting both the CC charges AND the SBI payment inflates total spend by ₹18,450.

    ACTIONS:
    1. "query": Explore transactions by merchant name or date range to understand the data.
       Payload: {"query_type": "merchant"|"date_range", "value": "<substring>"|"YYYY-MM-DD:YYYY-MM-DD"}
       Score: +0.010 if the query reveals unresolved duplicates, -0.010 if all are already resolved.

    2. "reconcile": Classify a single transaction.
       Payload: {"transaction_id": "<id>", "classification": "genuine_spend"|"cc_settlement"|"internal_transfer"|"refund"}
       Score: +0.060 if correct, -0.040 if wrong.

    3. "finalize": Submit the complete reconciliation. Ends the episode.
       Payload: {
           "excluded_ids": ["t2s05", "t2s08", ...],  # IDs classified as non-spend (settlements, transfers)
           "reconciled_totals": {
               "2024-01": {"Food & Dining": 1330.0, "Transport & Commute": 2255.0, ...},
               "2024-02": {...},
               "2024-03": {...}
           }
       }
       Scoring:
       - Exclusion F1 (precision+recall on duplicate detection): F1 × 0.200
       - Totals accuracy: up to 0.180 (per-category accuracy across all months)

    STRATEGY:
    - Start with "query" to understand merchant patterns and identify obvious duplicates.
    - Use "reconcile" to systematically classify each unique transaction.
    - Build accurate reconciled_totals by summing genuine_spend transactions per category per month.
    - Use "finalize" only when confident in classifications and totals.

    Respond ONLY with a valid JSON object. No prose, no markdown fences.
""").strip()

# Task 3 — Forward budget planning with life event shock
SYSTEM_PROMPT_TASK3 = textwrap.dedent("""
    You are a financial intelligence agent for FinanceEnv Task 3: Forward Budget Planning.

    OBJECTIVE:
    You are given 2 months of pre-reconciled spend history (Jan-Feb 2024) for Ananya Sharma.
    Your job: Build a realistic monthly budget across all 9 categories to meet stated financial goals.

    FINANCIAL GOALS:
    - Save ₹8,000 this month (leftover after all spending)
    - Reduce Food & Dining by 15% compared to last month (Feb: ₹9,100)
    - Maintain zero deficit across all categories
    - Fixed EMI: ₹12,000 per month
    - Income: ₹85,000/month

    VALID CATEGORIES (all 9 must be included in finalize):
        "Food & Dining"
        "Transport & Commute"
        "Utilities & Bills"
        "EMI & Loan Repayment"
        "Entertainment & Subscriptions"
        "Healthcare"
        "Shopping & Apparel"
        "Savings & Investment"
        "Other"

    ACTIONS:
    1. "query": Explore historical spend for a category to understand baselines.
       Payload: {"category": "<one of the 9>", "months": ["2024-01", "2024-02"]}
       Score: +0.015 if category not yet budgeted, -0.010 if already set.
       Use this to understand historical patterns before setting budgets.

    2. "set_budget": Set or revise a category's budget allocation.
       Payload: {"category": "<category>", "amount": <float >= 0>}
       Score: +0.020 if within ±30% of historical average (realistic),
              -0.060 if zero for a category that had historical spend.
       You can revise multiple times with no penalty.

    3. "finalize": Lock the budget and trigger simulation. Must include all 9 categories.
       Payload: {"budget": {"Food & Dining": <float>, "Transport & Commute": <float>, ..., "Other": <float>}}
       Score breakdown:
         - Budget valid (sum <= ₹85,000): +0.100
         - Adherence (all 9 categories within actual spend): up to +0.150
         - Savings goal met (₹8,000+ remaining): up to +0.200
         - Life event absorption (₹7,500 buffer in Other or excess Savings): up to +0.100
         - Penalties: critical underfunding of essentials (EMI, Utilities, Healthcare): -0.080 each
       No queries before finalize: -0.050 penalty.

    STRATEGY:
    1. Start with "query" on a few key categories (Food, Transport, EMI, Savings) to understand baselines.
    2. Use "set_budget" to allocate realistic amounts — stay close to historical averages.
    3. Ensure total budget sum <= ₹85,000 (about ₹77,000 if you want ₹8,000 savings buffer).
    4. Set "Other" with a reasonable buffer (₹3,000–5,000 minimum) for unexpected expenses.
    5. Use "finalize" when confident all 9 categories are set and sum is within income.

    CRITICAL RULES:
    - Must include ALL 9 categories in finalize payload or -0.050 per missing category.
    - Budget sum MUST be <= ₹85,000 or -0.200 penalty and episode ends.
    - Do NOT submit finalize without at least one prior query action.
    - Respond ONLY with a valid JSON object. No prose, no markdown fences.
""").strip()

SYSTEM_PROMPTS = {
    "task1": SYSTEM_PROMPT_TASK1,
    "task2": SYSTEM_PROMPT_TASK2,
    "task3": SYSTEM_PROMPT_TASK3,
}

# ---------------------------------------------------------------------------
# Stdout loggers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# HTTP helpers (calls the live HF Space)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Observation → user message
# ---------------------------------------------------------------------------

def build_user_message(obs: dict, last_feedback: Optional[str]) -> str:
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
            f"  [{t['id']}] {t['date']} | {t['source']} | "
            f"{t['description']} | ₹{t['amount']}"
        )
    if last_feedback:
        lines.append(f"\nlast_feedback: {last_feedback}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# LLM call with rate-limit retry
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, messages: list) -> str:
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                seed=42,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(10 * (attempt + 1))
            else:
                raise
    raise RuntimeError("LLM call failed after 5 retries")

# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def main() -> None:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Required for OpenAI API calls.")

    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

    system_prompt = SYSTEM_PROMPTS.get(TASK_NAME, SYSTEM_PROMPT_TASK1)
    max_steps     = MAX_STEPS.get(TASK_NAME, 25)

    rewards:      List[float] = []
    steps_taken:  int         = 0
    final_score:  float       = 0.0
    success:      bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset the environment
        obs = http_post("/reset", {"task_id": TASK_NAME})

        messages:      list           = [{"role": "system", "content": system_prompt}]
        last_feedback: Optional[str]  = None

        for step in range(1, max_steps + 1):
            # Build and send user message
            user_msg = build_user_message(obs, last_feedback)
            messages.append({"role": "user", "content": user_msg})

            # Get action from LLM
            raw_json = call_llm(client, messages)
            messages.append({"role": "assistant", "content": raw_json})

            error: Optional[str] = None
            try:
                action = json.loads(raw_json)
                assert "action_type" in action
                assert "payload" in action
                assert "confidence" in action
            except Exception as e:
                error = str(e)
                # Fallback: finalize to avoid crash
                action = {"action_type": "finalize", "payload": {}, "confidence": 0.0}

            # Step the environment
            result = http_post("/step", action)
            reward_obj = result["reward"]
            obs        = result["observation"]
            done       = result["done"]

            reward        = reward_obj["score"]
            last_feedback = reward_obj["feedback"]
            final_score   = reward_obj["cumulative_score"]

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(action, separators=(",", ":")),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        # Always emit [END] even on failure
        print(f"[DEBUG] Exception: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


if __name__ == "__main__":
    main()
