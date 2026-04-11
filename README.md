---
title: FinanceEnv
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# FinanceEnv — OpenEnv Hackathon

A reinforcement learning environment for training AI agents to perform **personal financial intelligence tasks**: transaction categorization, cross-source deduplication/reconciliation, and forward budget planning.

---

## The Real Problem

Every bank, credit card, and UPI app categorizes spending **within its own silo**. A person with an HDFC Credit Card, an SBI Savings account, and Paytm UPI ends up with three separate "Food & Dining" totals that overlap, conflict, and double-count each other.

The worst part: when the SBI account pays the HDFC credit card bill, it logs something like `HDFC CREDIT CARD PAYMENT ₹18,450` as an outflow. The HDFC CC already shows every individual merchant charge that makes up that ₹18,450. A naive sum of both sources inflates total spend by ₹18,450 — even before you've added Paytm.

No consumer tool actually solves this today. FinanceEnv trains agents to do exactly this: ingest multi-source transaction data, reconcile across sources, detect duplication, and build a forward-looking financial plan from the resulting clean, unified view.

---

## What is FinanceEnv?

FinanceEnv is a structured **OpenEnv**-compatible environment where an AI agent acts as a personal financial intelligence assistant for a simulated salaried individual. The agent reads multi-source transaction data through a gym-like HTTP interface (`reset` / `step` / `state`), takes structured JSON actions at each step, and receives a score based on the quality of its financial reasoning.

The environment is designed to mirror real fintech workloads:
- **Ambiguous raw data** — merchant strings like `ZOMATO*ORDER 8829` must be mapped to canonical categories
- **Cross-source duplication** — the same economic event appears in 2+ bank feeds
- **Goal-conditioned planning** — budgets must satisfy stated savings targets, reduce overspending, and absorb life-event shocks

All synthetic data represents a single consistent persona: **Ananya Sharma**, a Mumbai software engineer earning ₹85,000/month with three financial accounts.

---

## The Three Tasks

### Task 1 — Transaction Categorization
**Difficulty:** Easy | **Max Steps:** 2 | **Data:** 20 SBI Savings transactions (January 2024)

The agent receives 20 raw bank transactions and must assign each one to exactly one of the 9 canonical spending categories. The challenge is normalizing ambiguous merchant strings (`UPI/NEFT/...`) to the correct semantic category.

**Actions:**
```json
{"action_type": "categorize", "payload": {"transaction_id": "txn_sbi_001", "category": "Savings & Investment"}, "confidence": 0.9}
{"action_type": "finalize", "payload": {}, "confidence": 1.0}
```

**Scoring:**
- +0.040 per exact match
- +0.015 per near-miss (parent category in taxonomy hierarchy)
- −0.010 for invalid category string
- +0.050 finalize bonus (always)
- +0.050 efficiency bonus (all 20 addressed within step threshold)

---

### Task 2 — Cross-Source Reconciliation
**Difficulty:** Medium | **Max Steps:** 3 | **Data:** ~40 transactions across HDFC CC, SBI Savings, Paytm UPI (3 months)

The agent must identify which transactions are genuine spend versus credit card bill payments, internal transfers, or refunds — then submit reconciled monthly category totals that exclude the duplicates.

**Actions:**
```json
{"action_type": "reconcile", "payload": {"transaction_id": "t2s05", "classification": "cc_settlement"}, "confidence": 0.85}
{"action_type": "query", "payload": {"query_type": "merchant", "value": "HDFC"}, "confidence": 0.8}
{"action_type": "finalize", "payload": {"excluded_ids": ["t2s05", "t2s08"], "reconciled_totals": {"2024-01": {"Food & Dining": 1330.0, "Transport & Commute": 2255.0}}}, "confidence": 1.0}
```

**Valid classifications:** `genuine_spend` | `cc_settlement` | `internal_transfer` | `refund`

**Query types:**
- `merchant` — substring match on description
- `date_range` — format `"YYYY-MM-DD:YYYY-MM-DD"`

**Scoring:**
- +0.060 correct classification, −0.040 wrong, −0.020 duplicate action
- Query: +0.010 if unresolved duplicates found, −0.010 if redundant
- Finalize: F1 score on excluded_ids (max +0.200) + totals accuracy (+0.010/+0.005 per category)

---

### Task 3 — Forward Budget Planning
**Difficulty:** Hard | **Max Steps:** 4 | **Data:** 2 months clean spend history + financial goals + simulated Month 3

The agent receives reconciled spending history and must build a realistic ₹85,000/month budget across all 9 categories. The budget is then run against a pre-baked simulation (actual Month 3 spend including a ₹7,500 car repair life event) and scored against adherence, savings goals, and financial resilience.

**Goals:** Save ₹8,000 this month · Reduce Food & Dining by 15% vs last month · Fixed EMI: ₹12,000 · Zero deficit across all categories

**Actions:**
```json
{"action_type": "query", "payload": {"category": "Food & Dining", "months": ["2024-01", "2024-02"]}, "confidence": 0.9}
{"action_type": "set_budget", "payload": {"category": "Food & Dining", "amount": 7700.0}, "confidence": 0.8}
{"action_type": "finalize", "payload": {"budget": {"Food & Dining": 7700, "Transport & Commute": 3000, "Utilities & Bills": 2500, "EMI & Loan Repayment": 12000, "Entertainment & Subscriptions": 1500, "Healthcare": 1000, "Shopping & Apparel": 2000, "Savings & Investment": 8000, "Other": 47300}}, "confidence": 1.0}
```

**Scoring:**
- set_budget: +0.020 within ±30% of historical avg, −0.060 zero budget for a historically-spent category
- finalize: −0.050 no prior queries, −0.050 per missing category, −0.200 if sum > ₹85,000
- Adherence bonus: +0.150 (all 9), +0.080 (7+), +0.040 (5+)
- Savings goal: +0.200 (met), +0.100 (50%+ met)
- Life event absorption: +0.100 (full), +0.050 (50%+)
- Essential underfunding penalty: −0.080 per category (EMI, Utilities, Healthcare) below 50% of historical avg

---

## Canonical Category Taxonomy

Exactly 9 categories. Use exact strings — case-sensitive.

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

---

## API Endpoints

Auto-generated by the OpenEnv library at `POST /reset`, `POST /step`, `GET /state`, `GET /health`, `GET /schema`, `GET /docs`.

### `POST /reset`
Start a new episode.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'
```

Returns an `Observation` with transactions, account balance, task context, and step count.

### `POST /step`
Submit one action.

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "categorize", "payload": {"transaction_id": "txn_sbi_001", "category": "Savings & Investment"}, "confidence": 0.9}'
```

Returns:
```json
{
  "observation": {"transactions": [...], "step_count": 1, ...},
  "reward": {"score": 0.040, "cumulative_score": 0.040, "feedback": "Correct.", "partial_scores": {...}},
  "done": false
}
```

### `GET /state`
Read-only snapshot of internal episode state (step count, addressed IDs, cumulative score, budget draft).

### `GET /health`
Returns `{"status": "ok"}`.

---

## Universal Penalties

Enforced by `env.py` before dispatching to any grader.

| Trigger | Score Delta |
|---|---|
| `action_type` not valid for active task | −0.020 |
| Failed Pydantic validation on payload | −0.010 |
| Same `transaction_id` actioned 3+ times | −0.030 per repeat |
| Max steps exceeded | Hard stop, score capped |

---

## Local Setup

**Requirements:** Python 3.11+

```bash
git clone <repo>
cd financeEnv
python -m venv .venv
pip install -r requirements.txt
```

**Start the server:**
```bash
.venv/bin/uvicorn server.app:app --reload --port 7860
# API docs available at http://localhost:7860/docs
```

**Run the LLM agent (requires server running):**
```bash
API_BASE_URL=https://api.openai.com/v1 API_KEY=sk-... TASK_NAME=task1 \
  .venv/bin/python finance_env/inference.py
```

**Run the local baseline (no server needed, calls env directly):**
```bash
# OpenAI
OPENAI_API_KEY=sk-... .venv/bin/python baseline/run_baseline.py

# Ollama (local LLM)
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b .venv/bin/python baseline/run_baseline.py

# Single task
TASK=task2 OPENAI_API_KEY=sk-... .venv/bin/python baseline/run_baseline.py
```

**Docker:**
```bash
docker build -t financeenv .
docker run -p 7860:7860 financeenv
# Starts server, waits for /health, then runs inference.py if API_BASE_URL + API_KEY are set
```

**Deploy to HuggingFace Spaces:**
```bash
HF_TOKEN=hf_... .venv/bin/python scripts/deploy_hf.py
```

---

## How to Replicate / Extend

1. **Data** lives in `finance_env/data/` as static JSON. To build your own environment, generate equivalent JSON files for your persona and task. All data is loaded once at module import.

2. **Graders** in `finance_env/tasks/` are independent Python modules. Each exports `grade_<action>()` returning `(score_delta, partial_scores, feedback, done)`. Add a new grader by creating a new module and wiring it into `env.py`'s `_dispatch()`.

3. **Models** are Pydantic classes in `finance_env/models.py`. All contracts between agent and environment flow through these — never define them elsewhere.

4. **HTTP server** (`server/app.py`) is a thin adapter: OpenEnv creates a new adapter instance per request, but all calls delegate to a module-level singleton `FinanceEnv()` so state persists across HTTP calls.

5. **Inference agent** (`finance_env/inference.py`) is a generic LLM loop: reset → observe → call LLM → step → repeat. Swap the system prompt and model to run any LLM.

---

## Project Structure

```
finance_env/
├── env.py                  # FinanceEnv: reset / step / state
├── models.py               # Pydantic contracts (Transaction, Observation, Action, Reward, State)
├── inference.py            # LLM agent runner (HTTP-based, evaluator-compatible output)
├── validate_submission.py  # Pre-submission validation (checks [START]/[STEP]/[END] markers)
├── tasks/
│   ├── task1_categorize.py
│   ├── task2_reconcile.py
│   └── task3_budget.py
└── data/
    ├── categories.json
    ├── task1_transactions.json
    ├── task2_multi_source.json
    ├── task2_ground_truth.json
    ├── task3_history.json
    ├── task3_goals.json
    └── task3_simulation.json

server/app.py               # FastAPI server (OpenEnv adapter + singleton)
baseline/
├── run_baseline.py         # Direct env baseline (OpenAI or Ollama)
└── test_e2e_task1.py       # HTTP integration test for Task 1
scripts/
└── deploy_hf.py            # HuggingFace Spaces deployment helper
```

---

## Documentation

- **Developer & architecture guide:** `CLAUDE.md`
- **Testing commands (local + HF Space):** `TESTING.md`
- **OpenEnv standard:** https://openenv.io
