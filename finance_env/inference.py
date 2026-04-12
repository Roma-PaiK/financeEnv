"""
inference.py — LLM agent that runs one episode against the FinanceEnv HTTP server.

Reads from environment variables, calls /reset and /step, logs structured output
consumed by the hackathon evaluator.

Env vars:
    API_BASE_URL   LLM proxy endpoint (default: HuggingFace router)
    API_KEY        LLM key (or HF_TOKEN)
    MODEL_NAME     Model to use (auto-resolved from proxy if unset)
    ENV_BASE_URL   Environment server URL (default: http://localhost:7860)
    SPACE_URL      Alias for ENV_BASE_URL
    TASK_NAME      task1 | task2 | task3 (default: task1)

Stdout format (machine-parsed by evaluator):
    [START] task=<name> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<float> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

# Load .env file if present (local dev only)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or ""
MODEL_NAME   = os.getenv("MODEL_NAME", "")
SPACE_URL    = (os.getenv("ENV_BASE_URL") or os.getenv("SPACE_URL") or "http://localhost:7860").rstrip("/")
TASK_NAME    = os.getenv("TASK_NAME", "task1")
BENCHMARK    = "finance_env_india"

MAX_STEPS               = int(os.getenv("MAX_STEPS", "25"))
TEMPERATURE             = float(os.getenv("TEMPERATURE", "0"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))

SYSTEM_PROMPT = textwrap.dedent("""
    You are a financial intelligence agent operating in the FinanceEnv environment.
    At each step you receive an observation and must respond with a single JSON action.

    Action schema:
    {
      "action_type": "categorize" | "reconcile" | "query" | "set_budget" | "finalize",
      "payload": { ... },
      "confidence": 0.0-1.0
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
    Build a realistic monthly budget for all 9 categories. Income: Rs85,000. Savings goal: Rs8,000.
      "query": { "category": "<category>", "months": ["2024-01", "2024-02"] }
      "set_budget": { "category": "<category>", "amount": <float> }
      "finalize": { "budget": { "Food & Dining": float, ... } }
    Rules: budget sum MUST be <= 85000. All 9 categories required.

    Respond ONLY with a valid JSON object. No prose, no markdown fences.
""").strip()


# ---------------------------------------------------------------------------
# Structured log helpers — format required by the evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{SPACE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Model resolution — probe proxy for a working model
# ---------------------------------------------------------------------------

def _resolve_model_candidates(client: Any) -> List[str]:
    candidates: List[str] = []
    if MODEL_NAME:
        candidates.append(MODEL_NAME)
    try:
        for item in getattr(client.models.list(), "data", []) or []:
            mid = getattr(item, "id", "")
            if mid and mid not in candidates:
                candidates.append(str(mid))
                if len(candidates) >= 8:
                    break
    except Exception:
        pass
    # Hardcoded fallbacks if proxy returns nothing
    for fallback in ["Qwen/Qwen2.5-72B-Instruct", "google/gemma-3-27b-it", "gpt-4o-mini", "gpt-4o"]:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _resolve_working_model(client: Any) -> str:
    """Try each candidate model with a test call; return the first that responds."""
    last_err: Exception | None = None
    for model_name in _resolve_model_candidates(client):
        try:
            client.chat.completions.create(
                model=model_name, temperature=0, max_tokens=1,
                messages=[{"role": "system", "content": "Reply with one word."}, {"role": "user", "content": "ping"}],
            )
            return model_name
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"No working model found on proxy: {last_err}")


# ---------------------------------------------------------------------------
# LLM call with rate-limit retry
# ---------------------------------------------------------------------------

def call_llm(client: Any, model_name: str, messages: list) -> str:
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages,
                response_format={"type": "json_object"},
                temperature=TEMPERATURE, seed=42,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(10 * (attempt + 1))
            else:
                raise
    raise RuntimeError("LLM call failed after 5 retries")


# ---------------------------------------------------------------------------
# Observation → text for LLM
# ---------------------------------------------------------------------------

def build_user_message(obs: dict, last_feedback: Optional[str]) -> str:
    lines = [
        f"task_id: {obs['task_id']}",
        f"task_context: {obs['task_context']}",
        f"step_count: {obs['step_count']}",
        f"account_balance: Rs{obs['account_balance']}",
        f"monthly_income: Rs{obs['monthly_income']}",
        f"current_month: {obs['current_month']}",
        f"sources_present: {obs['sources_present']}",
        "\ntransactions:",
    ]
    for t in obs["transactions"]:
        lines.append(f"  [{t['id']}] {t['date']} | {t['source']} | {t['description']} | Rs{t['amount']}")
    if last_feedback:
        lines.append(f"\nlast_feedback: {last_feedback}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def main() -> None:
    from openai import OpenAI

    rewards:      List[float] = []
    steps_taken:  int         = 0
    final_score:  float       = 0.0
    success:      bool        = False
    emitted_step: bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME or "auto")

    try:
        if not API_KEY:
            raise RuntimeError("No API key. Set API_KEY or HF_TOKEN.")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        model_name = _resolve_working_model(client)

        messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
        last_feedback: Optional[str] = None
        max_steps = {"task1": 2, "task2": 3, "task3": 4}.get(TASK_NAME, MAX_STEPS)

        obs = http_post("/reset", {"task_id": TASK_NAME})

        for step in range(1, max_steps + 1):
            messages.append({"role": "user", "content": build_user_message(obs, last_feedback)})

            error: Optional[str] = None
            action_str = "{}"
            reward = 0.0
            done = False

            try:
                raw_json = call_llm(client, model_name, messages)
                messages.append({"role": "assistant", "content": raw_json})
                action_str = raw_json
                action = json.loads(raw_json)
                assert "action_type" in action and "payload" in action
            except Exception as exc:
                error = str(exc)
                action = {"action_type": "finalize", "payload": {}, "confidence": 0.0}
                action_str = json.dumps(action)

            try:
                result       = http_post("/step", action)
                reward_obj   = result.get("reward") or {}
                obs          = result.get("observation") or result
                done         = bool(result.get("done", False))
                reward       = float(reward_obj.get("score", 0.0)) if isinstance(reward_obj, dict) else float(reward_obj or 0.0)
                last_feedback = reward_obj.get("feedback") if isinstance(reward_obj, dict) else None
                final_score  = float(reward_obj.get("cumulative_score", final_score)) if isinstance(reward_obj, dict) else final_score
            except Exception as exc:
                error = str(exc)
                done = True

            rewards.append(reward)
            steps_taken = step
            emitted_step = True
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)
        if not emitted_step:
            rewards.append(0.0)
            steps_taken = 1
            log_step(step=1, action="noop", reward=0.0, done=True, error=str(exc))

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


if __name__ == "__main__":
    main()
