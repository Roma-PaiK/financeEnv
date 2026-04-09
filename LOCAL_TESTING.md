# FinanceEnv — Local Testing Commands

Base URL: `http://localhost:7860`

---

## Start the Server

```bash
cd /Users/romapai/Documents/Projects/financeEnv
.venv/bin/uvicorn app:app --reload --port 7860
```

---

## Endpoint Checks

### Health
```bash
curl http://localhost:7860/health
```

### Schema (action + observation shapes)
```bash
curl http://localhost:7860/schema | python3 -m json.tool
```

### Docs (open in browser)
```bash
open http://localhost:7860/docs
```

### State (read current episode state)
```bash
curl http://localhost:7860/state | python3 -m json.tool
```

---

## Task 1 — Transaction Categorisation (easy, max 2 steps)

**Reset**
```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}' | python3 -m json.tool
```

**Categorize a transaction**
```bash
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "categorize",
    "payload": {
      "transaction_id": "txn_sbi_001",
      "category": "Savings & Investment"
    },
    "confidence": 0.9
  }' | python3 -m json.tool
```

**Finalize task1**
```bash
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "finalize",
    "payload": {},
    "confidence": 1.0
  }' | python3 -m json.tool
```

---

## Task 2 — Cross-Source Reconciliation (medium, max 3 steps)

**Reset**
```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task2"}' | python3 -m json.tool
```

**Query transactions**
```bash
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "query",
    "payload": {"filter": "source=HDFC_CC"},
    "confidence": 0.8
  }' | python3 -m json.tool
```

**Flag a duplicate/settlement row**
```bash
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "reconcile",
    "payload": {
      "transaction_id": "t2s05",
      "is_duplicate": true,
      "duplicate_of": "t2h01"
    },
    "confidence": 0.85
  }' | python3 -m json.tool
```

**Finalize task2**
```bash
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "finalize",
    "payload": {
      "monthly_totals": {
        "2024-01": {"Food & Dining": 1330.0, "Transport & Commute": 2255.0},
        "2024-02": {"Food & Dining": 1230.0, "Shopping & Apparel": 1600.0},
        "2024-03": {"EMI & Loan Repayment": 12000.0}
      }
    },
    "confidence": 0.8
  }' | python3 -m json.tool
```

---

## Task 3 — Forward Budget Planning (hard, max 4 steps)

**Reset**
```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task3"}' | python3 -m json.tool
```

**Query category history**
```bash
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "query",
    "payload": {"category": "Food & Dining"},
    "confidence": 0.9
  }' | python3 -m json.tool
```

**Set a budget for a category**
```bash
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "set_budget",
    "payload": {
      "category": "EMI & Loan Repayment",
      "amount": 12000.0
    },
    "confidence": 1.0
  }' | python3 -m json.tool
```

**Finalize task3 (must include all 9 categories)**
```bash
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "finalize",
    "payload": {
      "budget": {
        "Food & Dining": 3500.0,
        "Transport & Commute": 2000.0,
        "Utilities & Bills": 2500.0,
        "EMI & Loan Repayment": 12000.0,
        "Entertainment & Subscriptions": 1000.0,
        "Healthcare": 1000.0,
        "Shopping & Apparel": 2000.0,
        "Savings & Investment": 8000.0,
        "Other": 51000.0
      }
    },
    "confidence": 0.9
  }' | python3 -m json.tool
```

---

## Inference Agent (local pre-deploy test)

Runs `inference.py` against the local server. `API_KEY` and `API_BASE_URL` are **mandatory** — no defaults.

### Prerequisites — start the server first
```bash
.venv/bin/uvicorn app:app --reload --port 7860
```

### Task 1
```bash
API_BASE_URL=https://api.openai.com/v1 API_KEY=sk-... TASK_NAME=task1 \
  SPACE_URL=http://localhost:7860 .venv/bin/python inference.py
```

### Task 2
```bash
API_BASE_URL=https://api.openai.com/v1 API_KEY=sk-... TASK_NAME=task2 \
  SPACE_URL=http://localhost:7860 .venv/bin/python inference.py
```

### Task 3
```bash
API_BASE_URL=https://api.openai.com/v1 API_KEY=sk-... TASK_NAME=task3 \
  SPACE_URL=http://localhost:7860 .venv/bin/python inference.py
```

### Baseline agent (calls env.py directly, no server needed)
```bash
API_KEY=sk-... .venv/bin/python baseline/run_baseline.py
```

### Baseline — Ollama
```bash
LLM_PROVIDER=ollama .venv/bin/python baseline/run_baseline.py
```

---

## Canonical Categories (exactly 9)

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

## Penalty Reference

| Trigger | Score Delta |
|---|---|
| Wrong `action_type` for active task | −0.020 |
| Failed Pydantic validation | −0.010 |
| Same `transaction_id` actioned 3+ times | −0.030 |
| Max steps exceeded | Hard stop |
