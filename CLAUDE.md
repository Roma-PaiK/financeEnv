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
.venv/bin/uvicorn server.app:app --reload --port 7860

# Run the LLM inference agent against the HTTP server (requires server running)
ENV_BASE_URL=http://localhost:7860 API_KEY=... .venv/bin/python finance_env/inference.py

# Run the local baseline agent (calls env.step() directly, no HTTP)
OPENAI_API_KEY=sk-... .venv/bin/python baseline/run_baseline.py

# Validate submission before upload
.venv/bin/python finance_env/validate_submission.py
```

Build and run Docker image:
```bash
docker build -t financeenv .
docker run -p 7860:7860 financeenv
# Docker runs start.sh: starts server, waits for /health, then runs inference.py if API_BASE_URL + API_KEY set
```

### Entry Points (pyproject.toml scripts)

| Script | Module |
|---|---|
| `server` | `finance_env.server.app:main` |
| `inference` | `finance_env.inference:main` |
| `validate-submission` | `finance_env.validate_submission:main` |

---

## Architecture

### Core Contracts (read these first)

All Pydantic models are in `finance_env/models.py`. **Never redefine them elsewhere — always import from there.**

Key models:
- `Observation` — what the agent sees (transactions, account balance, task context). Also carries `done: bool` and `reward: Optional[float]` required by the openenv HTTP protocol — these are set by the adapter in `server/app.py`, not by `env.py`.
- `Action` — what the agent submits (`action_type` + `payload` + `confidence`)
- `Reward` — step result returned internally by `env.py` (`score`, `partial_scores`, `feedback`, `done`, `cumulative_score`). Not exposed over HTTP — the adapter extracts `done` and `cumulative_score` from it and embeds them into `Observation`.
- `State` — internal environment state (`task_id`, `step_count`, `cumulative_score`, `done`, `addressed_ids`, `budget_draft`)

### Environment (`finance_env/env.py`)

`FinanceEnv` is a plain Python class (no openenv base class). It has three public methods:
- `reset(task_id)` — initializes episode, strips hidden fields from transactions, returns `Observation`
- `step(action)` — the **only** method that mutates state; returns `(Observation, Reward, bool, dict)`
- `state()` — read-only method; returns current `State`

All scores are `float` clamped to `(EPSILON, 1-EPSILON)` — not raw `[0.0, 1.0]`. `cumulative_score` is normalized. `step()` enforces universal penalties before dispatching to graders.

Task configuration in `env.py`:
- `MAX_STEPS`: `{task1: 2, task2: 3, task3: 4}`
- `LEGAL_ACTIONS`: task-specific allowed action types
- `TASK_CONTEXT`: natural language task descriptions per task

**Do not change `env.py` or `baseline/run_baseline.py` to conform to openenv's interface — that is handled entirely by the adapter in `server/app.py`.**

### HTTP Server (`server/app.py`) — Adapter Pattern

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

### Inference Agent (`finance_env/inference.py`)

LLM-powered agent that runs episodes via HTTP against the running server. Used both locally and in Docker (via `start.sh`).

Configuration (env vars):
- `ENV_BASE_URL` / `SPACE_URL` — server base URL (default: HuggingFace router)
- `API_KEY` / `HF_TOKEN` — auth token for LLM proxy
- `MODEL_NAME` — LLM model to use (probed automatically if unset)
- `API_BASE_URL` — LLM proxy base URL
- `TASK_NAME` — which task to run (default: task1)

Behavior:
- Calls `POST /reset` → loops `POST /step` up to `MAX_STEPS`
- Sends structured system prompt with task instructions and category taxonomy
- Parses LLM JSON response → posts as `Action`
- Rate limit retry: 5 attempts with exponential backoff
- Structured logging: `[START]`, `[STEP]`, `[END]` markers for evaluator parsing

### Validation (`finance_env/validate_submission.py`)

Pre-submission check that runs `inference.py` as a subprocess and verifies:
- `[START]`, `[STEP]`, `[END]` markers present in stdout
- Writes `artifacts/submission/pre_submission_report.json`
- Exits with code 1 on failure

### Deployment (`start.sh`)

Docker entry point:
1. Starts uvicorn (`server/app.py`) in background on port 7860
2. Polls `/health` (up to 30 attempts) before continuing
3. If `API_BASE_URL` and `API_KEY` are set, runs `inference.py`
4. Keeps server alive with `wait`

---

## Data (`finance_env/data/`)

All data is **static, pre-generated JSON — never generated at runtime.**

| File | Used by |
|---|---|
| `task1_transactions.json` | Task 1 grader — 20 SBI Savings transactions (Jan 2024) |
| `task2_multi_source.json`, `task2_ground_truth.json` | Task 2 grader — ~40 txns across 3 sources |
| `task3_history.json`, `task3_goals.json`, `task3_simulation.json` | Task 3 grader |
| `categories.json` | All tasks — exactly 9 canonical categories |

**Critical:** `Transaction.correct_category` and `Transaction.is_cc_settlement` exist in raw data files but **must be stripped** before returning to the agent. `env.py` handles this; graders import raw files directly.

---

## Task Graders (`finance_env/tasks/`)

Three independent modules with no shared state. All return `(score_delta, partial_scores, feedback, done)`.

### Task 1 — Categorize (`task1_categorize.py`)
Legal actions: `categorize`, `finalize`. Max steps: 2.

**`grade_categorize()`** score deltas:
| Condition | Delta |
|---|---|
| Unknown transaction ID | 0.0 |
| Duplicate action on same txn | −0.020 |
| Invalid category string | −0.010 |
| Exact category match | +0.040 |
| Near-miss (parent category match) | +0.015 |
| Wrong category | 0.0 |

**`grade_finalize()`**:
- Terminal bonus: +0.050 (always on finalize)
- Efficiency bonus: +0.050 (all 20 transactions addressed in < 22 steps)
- Partial scores: `correct_labels` (fraction correct), `efficiency`

Uses a hierarchical `parent_category` map for near-miss scoring.

### Task 2 — Reconcile (`task2_reconcile.py`)
Legal actions: `reconcile`, `query`, `finalize`. Max steps: 3.
Valid classifications: `genuine_spend`, `cc_settlement`, `internal_transfer`, `refund`.

**`grade_reconcile()`** score deltas:
| Condition | Delta |
|---|---|
| Unknown transaction ID | 0.0 |
| Duplicate action on same txn | −0.020 |
| Invalid classification value | −0.010 |
| Correct classification | +0.060 |
| Incorrect classification | −0.040 |

**`grade_query()`**:
- Query types: `merchant` (substring match) or `date_range` (`YYYY-MM-DD:YYYY-MM-DD`)
- Returns first 8 matching transactions
- +0.010 if unresolved duplicates found in results, −0.010 if all resolved or no matches (−0.005)

**`grade_finalize()`**:
- F1 score on `excluded_ids` (TP/FP/FN) → max +0.200
- `reconciled_totals` accuracy: +0.010 per source within 2%, +0.005 within 10%
- Partial scores: `classification_accuracy`, `exclusion_f1`, `totals_accuracy`

### Task 3 — Budget (`task3_budget.py`)
Legal actions: `set_budget`, `query`, `finalize`. Max steps: 4.

**`grade_set_budget()`** score deltas:
| Condition | Delta |
|---|---|
| Invalid category | −0.010 |
| Invalid amount | −0.010 |
| Zero budget for historically-spent category | −0.060 |
| Within ±30% of historical average | +0.020 |
| Outside ±30% range | 0.0 |

Mutates `budget_draft` in state.

**`grade_query()`**:
- Returns monthly spend history for requested category
- +0.015 if not yet budgeted (useful query), −0.010 if already set (redundant)
- Partial scores: `realism` (fraction of set categories within ±30% of historical avg)

**`grade_finalize()`**:
| Condition | Delta |
|---|---|
| No prior queries | −0.050 |
| Missing required categories | −0.050 each |
| Budget sum > monthly income (85000) | −0.200 |
| Budget sum ≤ income | +0.100 |
| All categories within historical avg | +0.150 |
| 7+ categories within historical avg | +0.080 |
| 5+ categories within historical avg | +0.040 |
| Savings goal fully met | +0.200 |
| Savings goal 50%+ met | +0.100 |
| Life event buffer fully absorbed | +0.100 |
| Life event buffer 50%+ absorbed | +0.050 |
| Essential category < 50% of historical | −0.080 each |

Partial scores: `budget_validity`, `adherence`, `savings_goal`, `life_event_absorption`, `realism`

---

## Baseline Agent (`baseline/run_baseline.py`)

Calls `env.step()` directly (not via HTTP), supports two LLM providers:

- **OpenAI** (default): `LLM_PROVIDER=openai`, uses `OPENAI_API_KEY` or `API_KEY`, `temperature=0, seed=42`
- **Ollama**: `LLM_PROVIDER=ollama`, uses `OLLAMA_BASE_URL` and `OLLAMA_MODEL`

`TASK` env var filters which tasks to run (default: all 3). Receives the 4-tuple `(Observation, Reward, bool, dict)` directly from `env.step()` and reads `reward.feedback`, `reward.cumulative_score` etc. directly. Invalid JSON responses fall back to a penalty action.

---

## Key Rules (from tech spec Part 0)

- All scores are `float` clamped to `(EPSILON, 1-EPSILON)`; cumulative scores are normalized
- All Pydantic models live in `finance_env/models.py` — import from there, never redefine
- Ground truth data is in `finance_env/data/` — static, committed, never runtime-generated
- The three graders are completely independent — no shared state
- `step()` is the only function that mutates environment state
- `env.py` has no openenv dependency — it is pure Python; openenv conformance lives in `server/app.py` only
- Server is at `server/app.py`, not root `app.py`

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
