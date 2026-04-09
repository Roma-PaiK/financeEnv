# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**FinanceEnv** is an OpenEnv-compatible reinforcement learning environment for the OpenEnv Hackathon. It trains AI agents to perform personal financial intelligence tasks: transaction categorization, cross-source deduplication/reconciliation, and forward budget planning. The environment follows a gym-like interface (`reset` / `step` / `state`).

---

## Setup & Running

```bash
pip install -r requirements.txt

# Run the FastAPI server (HuggingFace Spaces entry point)
.venv/bin/uvicorn app:app --reload --port 7860

# Run the OpenAI baseline agent across all three tasks
OPENAI_API_KEY=sk-... .venv/bin/python baseline/run_baseline.py
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
- `Observation` — what the agent sees (transactions, account balance, task context). Also carries `done: bool` and `reward: Optional[float]` required by the openenv HTTP protocol — these are set by the adapter in `app.py`, not by `env.py`.
- `Action` — what the agent submits (`action_type` + `payload` + `confidence`)
- `Reward` — step result returned internally by `env.py` (`score`, `partial_scores`, `feedback`, `done`, `cumulative_score`). Not exposed over HTTP — the adapter extracts `done` and `cumulative_score` from it and embeds them into `Observation`.
- `State` — internal environment state (not exposed to the agent)

### Environment (`finance_env/env.py`)

`FinanceEnv` is a plain Python class (no openenv base class). It has three public methods:
- `reset(task_id)` — initializes episode, strips hidden fields from transactions, returns `Observation`
- `step(action)` — the **only** method that mutates state; returns `(Observation, Reward, bool, dict)`
- `state()` — read-only method; returns current `State`

All scores are `float` in `[0.0, 1.0]`. `cumulative_score` is normalized (never clipped raw deltas). `step()` enforces penalties before dispatching to graders — invalid `action_type`, failed Pydantic validation, or repeated actions on the same transaction.

**Do not change `env.py` or `baseline/run_baseline.py` to conform to openenv's interface — that is handled entirely by the adapter in `app.py`.**

### HTTP Server (`app.py`) — Adapter Pattern

The openenv library (`openenv.core.env_server`) is **stateless per HTTP request**: it creates a fresh env instance for every `/reset` and `/step` call. `FinanceEnv` is stateful (step requires prior reset on the same instance). This mismatch is resolved with an adapter:

```
HTTP request
     │
     ▼
FinanceEnvAdapter (inherits openenv's Environment)
     │  delegates all calls
     ▼
_singleton = FinanceEnv()   ← module-level, persists across requests
```

**Critical rules — do not break these:**

1. **Import path** — `from openenv.core.env_server import ...` (package is `openenv`, NOT `openenv_core`)
2. **Pass the class as factory** — `create_fastapi_app(FinanceEnvAdapter, ...)` not an instance. openenv calls it to create a new adapter per request; each adapter delegates to the singleton.
3. **Inherit from `Environment`** — required so openenv's `step_async.__func__` check doesn't crash. Gives `close()`, `step_async`, `reset_async` for free.
4. **`state` must be a `@property`** — the library accesses `_env.state` not `_env.state()`.
5. **`step()` must return just `Observation`** — the library calls `serialize_observation(obs)` which reads `obs.done` and `obs.reward`. The adapter unpacks the 4-tuple from `FinanceEnv.step()` and embeds `done`/`cumulative_score` into the observation before returning.

Auto-generated endpoints:
- `POST /reset` — starts episode, returns `Observation`
- `POST /step` — executes action, returns `StepResponse` (observation + reward + done)
- `GET /state` — read-only state
- `GET /health` — health check
- `GET /schema` — action/observation JSON schemas
- `GET /docs` — auto-generated API docs

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
- `task1_categorize.py` — scores `categorize` actions; max 2 steps
- `task2_reconcile.py` — scores `reconcile`, `query`, `finalize`; max 3 steps
- `task3_budget.py` — scores `set_budget`, `query`, `finalize`; max 4 steps

### Baseline Agent (`baseline/run_baseline.py`)

Runs OpenAI (GPT-4o) as the agent across all 3 tasks with `temperature=0, seed=42` for deterministic scores. Uses `response_format: {"type": "json_object"}` to enforce valid Action JSON. Calls `env.step()` directly (not via HTTP), so it receives the 4-tuple `(Observation, Reward, bool, dict)` and accesses `reward.feedback`, `reward.cumulative_score` etc. directly.

---

## Key Rules (from tech spec Part 0)

- All scores are `float` strictly in `[0.0, 1.0]`; cumulative scores are normalized, never clipped
- All Pydantic models live in `finance_env/models.py` — import from there, never redefine
- Ground truth data is in `finance_env/data/` — static, committed, never runtime-generated
- The three graders are completely independent — no shared state
- `step()` is the only function that mutates environment state
- `env.py` has no openenv dependency — it is pure Python; openenv conformance lives in `app.py` only

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
