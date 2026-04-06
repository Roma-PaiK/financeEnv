# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**FinanceEnv** is an OpenEnv-compatible reinforcement learning environment for the OpenEnv Hackathon. It trains AI agents to perform personal financial intelligence tasks: transaction categorization, cross-source deduplication/reconciliation, and forward budget planning. The environment follows a gym-like interface (`reset` / `step` / `state`).

---

## Setup & Running

```bash
pip install -r requirements.txt

# Run the OpenAI baseline agent across all three tasks
OPENAI_API_KEY=sk-... python baseline/run_baseline.py

# Run the FastAPI server (HuggingFace Spaces entry point)
uvicorn app:app --reload
```

Build the Docker image:
```bash
docker build -t financeenv .
docker run -p 7860:7860 financeenv
```

---

## Architecture

### Core Contracts (read these first)

All Pydantic models are in `finance_env/models.py`. **Never redefine them elsewhere — always import from there.**

Key models:
- `Observation` — what the agent sees (transactions stripped of hidden fields, account balance, task context)
- `Action` — what the agent submits (`action_type` + `payload` + `confidence`)
- `Reward` — step result (`score`, `partial_scores`, `feedback`, `done`, `cumulative_score`)
- `State` — internal environment state (not exposed to agent)

### Environment (`finance_env/env.py`)

`FinanceEnv` has three public methods:
- `reset(task_id)` — initializes episode, strips hidden fields from transactions, returns `Observation`
- `step(action)` — the **only** method that mutates state; dispatches to the correct grader
- `state()` — read-only; returns current `State`

All scores are `float` in `[0.0, 1.0]`. `cumulative_score` is normalized (never clipped raw deltas). `step()` enforces penalties before dispatching to graders — invalid `action_type`, failed Pydantic validation, or repeated actions on the same transaction.

### Data (`finance_env/data/`)

All data is **static, pre-generated JSON — never generated at runtime.**

| File | Used by |
|---|---|
| `task1_transactions.json` | Task 1 grader |
| `task2_multi_source.json`, `task2_ground_truth.json` | Task 2 grader |
| `task3_history.json`, `task3_goals.json`, `task3_simulation.json` | Task 3 grader |
| `categories.json` | All tasks — exactly 9 canonical categories |

**Critical:** `Transaction.correct_category` and `Transaction.is_cc_settlement` exist in raw data files but **must be stripped** before returning to the agent. `env.py` handles this; graders import raw files directly.

### Task Graders (`finance_env/tasks/`)

Three independent modules with no shared state:
- `task1_categorize.py` — scores `categorize` actions; max 25 steps
- `task2_reconcile.py` — scores `reconcile`, `query`, `finalize`; max 40 steps
- `task3_budget.py` — scores `set_budget`, `query`, `finalize`; max 60 steps

### Baseline Agent (`baseline/run_baseline.py`)

Runs OpenAI (GPT-4o) as the agent across all 3 tasks with `temperature=0, seed=42` for deterministic scores. Uses `response_format: {"type": "json_object"}` to enforce valid Action JSON.

---

## Key Rules (from tech spec Part 0)

- All scores are `float` strictly in `[0.0, 1.0]`; cumulative scores are normalized, never clipped
- All Pydantic models live in `finance_env/models.py` — import from there, never redefine
- Ground truth data is in `finance_env/data/` — static, committed, never runtime-generated
- The three graders are completely independent — no shared state
- `step()` is the only function that mutates environment state

## Canonical Category Taxonomy (exactly 9, no others valid)

```
Food & Dining
Transport & Commute
Utilities & Bills
EMI & Loan Repayment
Entertainment & Subscriptions
Healthcare
Shopping & Apparel
Savings & Investment
Other
```

## Universal Penalties (enforced in `env.py` before grader dispatch)

| Trigger | Score Delta |
|---|---|
| `action_type` not legal for active task | −0.020 |
| Failed Pydantic validation on payload | −0.010 |
| Same `transaction_id` actioned 3+ times | −0.030 per repeat |
| Max steps exceeded | Hard stop, score capped |