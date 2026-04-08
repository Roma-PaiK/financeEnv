---
title: FinanceEnv
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# FinanceEnv — OpenEnv Hackathon

A reinforcement learning environment for training AI agents to perform **personal financial intelligence tasks**: transaction categorization, cross-source reconciliation, and forward budget planning.

---

## What is FinanceEnv?

FinanceEnv simulates the real problem of multi-source banking: a person with multiple bank accounts, credit cards, and UPI wallets sees fragmented transaction data across silos. An AI agent must:

1. **Task 1: Categorize** — Assign 20 transactions to one of 9 canonical spending categories
2. **Task 2: Reconcile** — Identify duplicate transactions across 3 sources (bank, credit card, UPI) and build a unified spend view
3. **Task 3: Budget** — Plan a forward-looking monthly budget given historical spending and financial goals

All tasks use the same synthetic persona (**Ananya Sharma**, a ₹85,000/month earner) and are scored on correctness and reasoning quality.

---

## Quick Start

### Use the API (OpenEnv Protocol)

```bash
# 1. Reset to start a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'

# 2. Send an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "categorize",
    "payload": {"transaction_id": "txn_001", "category": "Food & Dining"},
    "confidence": 0.95
  }'

# 3. Check environment state
curl http://localhost:7860/state

# 4. Health check
curl http://localhost:7860/health
```

---

## The Three Tasks

### Task 1: Transaction Categorization
**Max Steps:** 25 | **Data:** 20 SBI transactions (Jan 2024)

Classify each transaction into one of 9 canonical categories.

```json
{
  "action_type": "categorize",
  "payload": {"transaction_id": "txn_sbi_001", "category": "Food & Dining"},
  "confidence": 0.9
}
```

**Scoring:** +0.050 per correct categorization

---

### Task 2: Cross-Source Reconciliation
**Max Steps:** 40 | **Data:** 3 months across HDFC Credit Card, SBI Savings, Paytm UPI

Identify duplicates (e.g., SBI paying HDFC bill) and submit reconciled spend totals.

```json
{
  "action_type": "reconcile",
  "payload": {
    "transaction_id": "t2s05",
    "classification": "cc_settlement"
  },
  "confidence": 0.85
}
```

Then finalize:
```json
{
  "action_type": "finalize",
  "payload": {
    "excluded_ids": ["t2s05", "t2s08"],
    "reconciled_totals": {
      "2024-01": {"Food & Dining": 1330.0, ...},
      ...
    }
  },
  "confidence": 1.0
}
```

---

### Task 3: Forward Budget Planning
**Max Steps:** 60 | **Data:** 2 months clean spend + goals

Build a ₹85,000/month budget meeting:
- ₹8,000 savings target
- 15% Food & Dining reduction
- Zero deficit across all categories

```json
{
  "action_type": "set_budget",
  "payload": {"category": "Food & Dining", "amount": 7700},
  "confidence": 0.8
}
```

Finalize with all 9 categories:
```json
{
  "action_type": "finalize",
  "payload": {
    "budget": {
      "Food & Dining": 7700,
      "Transport & Commute": 3000,
      "Utilities & Bills": 2500,
      "EMI & Loan Repayment": 12000,
      "Entertainment & Subscriptions": 1500,
      "Healthcare": 1000,
      "Shopping & Apparel": 2000,
      "Savings & Investment": 8000,
      "Other": 47300
    }
  },
  "confidence": 1.0
}
```

---

## Canonical Categories

Exactly 9 categories (case-sensitive, exact spelling):

- Food & Dining
- Transport & Commute
- Utilities & Bills
- EMI & Loan Repayment
- Entertainment & Subscriptions
- Healthcare
- Shopping & Apparel
- Savings & Investment
- Other

---

## API Endpoints

### `POST /reset`
Start a new episode.

**Request:** `{"task_id": "task1" | "task2" | "task3"}`

**Response:** Observation with transactions, balance, task context

---

### `POST /step`
Submit an action.

**Request:**
```json
{
  "action_type": "categorize" | "reconcile" | "query" | "set_budget" | "finalize",
  "payload": {...},
  "confidence": 0.0-1.0
}
```

**Response:**
```json
{
  "observation": {...},
  "reward": {
    "score": 0.050,
    "cumulative_score": 0.120,
    "feedback": "..."
  },
  "done": false
}
```

---

### `GET /state`
Read-only environment state snapshot.

---

### `GET /health`
Service health check.

---

## Scoring

All scores are **floats in [0.0, 1.0]**. Cumulative scores are normalized across all steps.

### Universal Penalties
- Invalid `action_type`: −0.020
- Failed JSON validation: −0.010
- Duplicate submission (same txn 3+ times): −0.030 per repeat
- Max steps exceeded: Hard stop

---

## Running Your Agent

See `inference.py` for a GPT-4o baseline that runs all 3 tasks against the live API.

```bash
export OPENAI_API_KEY=sk-...
export TASK_NAME=task1
python inference.py
```

Output format:
```
[START] task=task1 env=finance_env_india model=gpt-4o
[STEP] step=1 action={...} reward=0.05 done=false error=null
[END] success=true steps=21 score=1.000 rewards=0.05,0.05,...
```

---

## Project Structure

```
finance_env/
├── env.py                      # FinanceEnv class
├── models.py                   # Pydantic contracts
├── tasks/                      # Task graders (1, 2, 3)
└── data/                       # Static JSON data

app.py                           # FastAPI (this Space)
inference.py                     # Agent runner
CLAUDE.md                        # Developer guide
financeEnv_tech_spec.md         # Full specification
```

---

## Development

### Local setup
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

### Run baseline
```bash
python baseline/run_baseline.py
```

---

## Documentation

- **Full Spec:** See `financeEnv_tech_spec.md`
- **Implementation Guide:** See `CLAUDE.md`
- **OpenEnv Standard:** https://openenv.io

---

## License

MIT License — See LICENSE file
