# FinanceEnv — Testing Reference

Set `BASE_URL` to `http://localhost:7860` for local dev or your HF Space URL for deployment testing.

```bash
BASE_URL="http://localhost:7860"
# BASE_URL="https://romapai-finance-env-india.hf.space"
```

---

## Start the Server (local only)

```bash
.venv/bin/uvicorn server.app:app --reload --port 7860
```

---

## Health & Schema

```bash
curl $BASE_URL/health
curl -s $BASE_URL/schema | python3 -m json.tool
curl $BASE_URL/state | python3 -m json.tool
```

---

## Task 1 — Categorisation (max 2 steps)

```bash
# Reset
curl -s -X POST $BASE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}' | python3 -m json.tool

# Categorize a transaction
curl -s -X POST $BASE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "categorize", "payload": {"transaction_id": "txn_sbi_001", "category": "Savings & Investment"}, "confidence": 0.9}' \
  | python3 -m json.tool

# Finalize
curl -s -X POST $BASE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "finalize", "payload": {}, "confidence": 1.0}' \
  | python3 -m json.tool
```

---

## Task 2 — Reconciliation (max 3 steps)

```bash
# Reset
curl -s -X POST $BASE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task2"}' | python3 -m json.tool

# Query (merchant or date_range)
curl -s -X POST $BASE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "query", "payload": {"query_type": "merchant", "value": "HDFC"}, "confidence": 0.8}' \
  | python3 -m json.tool

# Reconcile a transaction
curl -s -X POST $BASE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "reconcile", "payload": {"transaction_id": "t2s05", "classification": "cc_settlement"}, "confidence": 0.85}' \
  | python3 -m json.tool

# Finalize
curl -s -X POST $BASE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "finalize", "payload": {"excluded_ids": ["t2s05"], "reconciled_totals": {"2024-01": {"Food & Dining": 1330.0}}}, "confidence": 0.8}' \
  | python3 -m json.tool
```

---

## Task 3 — Budget Planning (max 4 steps)

```bash
# Reset
curl -s -X POST $BASE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task3"}' | python3 -m json.tool

# Query historical spend for a category
curl -s -X POST $BASE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "query", "payload": {"category": "Food & Dining", "months": ["2024-01", "2024-02"]}, "confidence": 0.9}' \
  | python3 -m json.tool

# Set budget for a category
curl -s -X POST $BASE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "set_budget", "payload": {"category": "EMI & Loan Repayment", "amount": 12000.0}, "confidence": 1.0}' \
  | python3 -m json.tool

# Finalize (all 9 categories required, sum must be <= 85000)
curl -s -X POST $BASE_URL/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "finalize",
    "payload": {"budget": {
      "Food & Dining": 3500.0,
      "Transport & Commute": 2000.0,
      "Utilities & Bills": 2500.0,
      "EMI & Loan Repayment": 12000.0,
      "Entertainment & Subscriptions": 1000.0,
      "Healthcare": 1000.0,
      "Shopping & Apparel": 2000.0,
      "Savings & Investment": 8000.0,
      "Other": 53000.0
    }},
    "confidence": 0.9
  }' | python3 -m json.tool
```

---

## Inference Agent

Requires the server to be running (or point `ENV_BASE_URL` at the HF Space).

```bash
# Local server
API_BASE_URL=https://api.openai.com/v1 API_KEY=sk-... TASK_NAME=task1 \
  .venv/bin/python finance_env/inference.py

# Against HF Space
API_BASE_URL=https://api.openai.com/v1 API_KEY=sk-... TASK_NAME=task1 \
  ENV_BASE_URL=https://romapai-finance-env-india.hf.space \
  .venv/bin/python finance_env/inference.py
```

> The hackathon evaluator injects `API_KEY` and `API_BASE_URL` automatically — do **not** set these as HF Space secrets.

## Baseline Agent (no server needed)

```bash
# OpenAI
OPENAI_API_KEY=sk-... .venv/bin/python baseline/run_baseline.py

# Ollama
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b .venv/bin/python baseline/run_baseline.py

# Single task
TASK=task1 OPENAI_API_KEY=sk-... .venv/bin/python baseline/run_baseline.py
```

## E2E HTTP Test

```bash
# Runs against HF Space by default; set BASE_URL for local
BASE_URL=http://localhost:7860 .venv/bin/python baseline/test_e2e_task1.py
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Server not reachable | Check `uvicorn server.app:app` is running on the right port |
| `{"detail":"Not Found"}` | Check `/docs` for available endpoints |
| Validation error on `confidence` | Must be a float `0.0–1.0` |
| Penalty on step | Check `feedback` in response — wrong `action_type` or bad payload |
| `No API key found` | Set `API_KEY` or `HF_TOKEN` env var |
