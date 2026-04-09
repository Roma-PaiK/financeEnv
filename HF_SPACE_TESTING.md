# FinanceEnv — HF Space Testing Commands

After pushing to GitHub and deploying to HuggingFace Spaces, use these commands to verify the environment is running correctly.

**Base URL:** Replace `YOUR_HF_SPACE_URL` with your actual HF Space URL (e.g., `https://romapai-finance-env-india.hf.space`)

---

## Health Check

### Health Endpoint
```bash
HF_URL="https://romapai-finance-env-india.hf.space"
curl ${HF_URL}/health
```

### Full Schema Check
```bash
curl -s ${HF_URL}/schema | python3 -m json.tool
```

### API Docs (open in browser)
```bash
# Replace URL and open in browser
https://romapai-finance-env-india.hf.space/docs
```

---

## Run inference.py Against HF Space

`API_KEY` and `API_BASE_URL` are **mandatory** — injected by the hackathon evaluator, or set manually for local testing against the deployed space.

### Task 1
```bash
API_BASE_URL=https://api.openai.com/v1 API_KEY=sk-... TASK_NAME=task1 \
  SPACE_URL=https://romapai-finance-env-india.hf.space .venv/bin/python inference.py
```

### Task 2
```bash
API_BASE_URL=https://api.openai.com/v1 API_KEY=sk-... TASK_NAME=task2 \
  SPACE_URL=https://romapai-finance-env-india.hf.space .venv/bin/python inference.py
```

### Task 3
```bash
API_BASE_URL=https://api.openai.com/v1 API_KEY=sk-... TASK_NAME=task3 \
  SPACE_URL=https://romapai-finance-env-india.hf.space .venv/bin/python inference.py
```

> **Note:** The hackathon evaluator injects `API_KEY` and `API_BASE_URL` automatically — do **not** set these as HF Space secrets, or you will bypass their LiteLLM proxy.

---

## Task 1 — Transaction Categorisation (easy, max 2 steps)

### Reset Task 1
```bash
HF_URL="https://romapai-finance-env-india.hf.space"
curl -s -X POST ${HF_URL}/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}' | python3 -m json.tool
```

### Categorize a Transaction
```bash
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "categorize",
      "payload": {
        "transaction_id": "txn_sbi_001",
        "category": "Savings & Investment"
      },
      "confidence": 0.9
    }
  }' | python3 -m json.tool
```

### Finalize Task 1
```bash
curl -s -X POST ${HF_URL}/step \
    -H "Content-Type: application/json" \
    -d '{
      "action": {
        "action_type": "finalize",
        "payload": {},            
        "confidence": 1.0
      }                  
    }' | python3 -m json.tool
```

### Full Task 1 Test Script
```bash
#!/bin/bash
HF_URL="https://romapai-finance-env-india.hf.space"

echo "=== Task 1: Reset ==="
curl -s -X POST ${HF_URL}/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}' | python3 -m json.tool

echo -e "\n=== Task 1: Categorize ==="
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "categorize",
      "payload": {
        "transaction_id": "txn_sbi_001",
        "category": "Savings & Investment"
      },
      "confidence": 0.9
    }
  }' | python3 -m json.tool

echo -e "\n=== Task 1: Finalize ==="
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "finalize",
      "payload": {},
      "confidence": 1.0
    }
  }' | python3 -m json.tool
```

---

## Task 2 — Cross-Source Reconciliation (medium, max 3 steps)

### Reset Task 2
```bash
HF_URL="https://romapai-finance-env-india.hf.space"
curl -s -X POST ${HF_URL}/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task2"}' | python3 -m json.tool
```

### Query Transactions
```bash
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "query",
      "payload": {
        "query_type": "merchant",
        "value": "HDFC"
      },
      "confidence": 0.8
    }
  }' | python3 -m json.tool
```

### Reconcile a Transaction
```bash
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "reconcile",
      "payload": {
        "transaction_id": "t2s05",
        "classification": "cc_settlement"
      },
      "confidence": 0.85
    }
  }' | python3 -m json.tool
```

### Finalize Task 2
```bash
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "finalize",
      "payload": {
        "reconciled_totals": {
          "2024-01": {"Food & Dining": 1330.0, "Transport & Commute": 2255.0},
          "2024-02": {"Food & Dining": 1230.0, "Shopping & Apparel": 1600.0}
        },
        "excluded_ids": ["t2s05"]
      },
      "confidence": 0.8
    }
  }' | python3 -m json.tool
```

### Full Task 2 Test Script
```bash
#!/bin/bash
HF_URL="https://romapai-finance-env-india.hf.space"

echo "=== Task 2: Reset ==="
curl -s -X POST ${HF_URL}/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task2"}' | python3 -m json.tool

echo -e "\n=== Task 2: Query ==="
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "query",
      "payload": {
        "query_type": "merchant",
        "value": "HDFC"
      },
      "confidence": 0.8
    }
  }' | python3 -m json.tool

echo -e "\n=== Task 2: Reconcile ==="
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "reconcile",
      "payload": {
        "transaction_id": "t2s05",
        "classification": "cc_settlement"
      },
      "confidence": 0.85
    }
  }' | python3 -m json.tool

echo -e "\n=== Task 2: Finalize ==="
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "finalize",
      "payload": {
        "reconciled_totals": {
          "2024-01": {"Food & Dining": 1330.0, "Transport & Commute": 2255.0}
        },
        "excluded_ids": ["t2s05"]
      },
      "confidence": 0.8
    }
  }' | python3 -m json.tool
```

---

## Task 3 — Forward Budget Planning (hard, max 4 steps)

### Reset Task 3
```bash
HF_URL="https://romapai-finance-env-india.hf.space"
curl -s -X POST ${HF_URL}/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task3"}' | python3 -m json.tool
```

### Query Category History
```bash
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "query",
      "payload": {
        "category": "Food & Dining",
        "months": ["2024-01", "2024-02"]
      },
      "confidence": 0.9
    }
  }' | python3 -m json.tool
```

### Set Budget for Category
```bash
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "set_budget",
      "payload": {
        "category": "EMI & Loan Repayment",
        "amount": 12000.0
      },
      "confidence": 1.0
    }
  }' | python3 -m json.tool
```

### Finalize Task 3 (all 9 categories required)
```bash
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
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
          "Other": 41000.0
        }
      },
      "confidence": 0.9
    }
  }' | python3 -m json.tool
```

### Full Task 3 Test Script
```bash
#!/bin/bash
HF_URL="https://romapai-finance-env-india.hf.space"

echo "=== Task 3: Reset ==="
curl -s -X POST ${HF_URL}/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task3"}' | python3 -m json.tool

echo -e "\n=== Task 3: Query ==="
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "query",
      "payload": {
        "category": "Food & Dining",
        "months": ["2024-01", "2024-02"]
      },
      "confidence": 0.9
    }
  }' | python3 -m json.tool

echo -e "\n=== Task 3: Set Budget ==="
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "set_budget",
      "payload": {
        "category": "EMI & Loan Repayment",
        "amount": 12000.0
      },
      "confidence": 1.0
    }
  }' | python3 -m json.tool

echo -e "\n=== Task 3: Finalize ==="
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
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
          "Other": 41000.0
        }
      },
      "confidence": 0.9
    }
  }' | python3 -m json.tool
```

---

## Complete Post-Deploy Verification (All Tasks)

Save as `test_hf_space.sh` and run after deployment:

```bash
#!/bin/bash
set -e

HF_URL="https://romapai-finance-env-india.hf.space"

echo "================================"
echo "FinanceEnv HF Space Test Suite"
echo "================================"

echo -e "\n[1/7] Health Check..."
curl -s ${HF_URL}/health | python3 -m json.tool

echo -e "\n[2/7] Schema Verification..."
curl -s ${HF_URL}/schema | python3 -m json.tool | head -20

echo -e "\n[3/7] Task 1 - Reset..."
curl -s -X POST ${HF_URL}/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}' | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'✓ Task1 Reset: {d.get(\"step_count\", \"?\")} transactions')"

echo "[4/7] Task 2 - Reset..."
curl -s -X POST ${HF_URL}/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task2"}' | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'✓ Task2 Reset: {d.get(\"step_count\", \"?\")} transactions')"

echo "[5/7] Task 3 - Reset..."
curl -s -X POST ${HF_URL}/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task3"}' | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'✓ Task3 Reset: {d.get(\"step_count\", \"?\")} transactions')"

echo "[6/7] Task 1 - Step..."
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "categorize",
      "payload": {"transaction_id": "txn_sbi_001", "category": "Savings & Investment"},
      "confidence": 0.9
    }
  }' | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'✓ Task1 Step: reward={d.get(\"reward\", {}).get(\"score\", \"?\")} cumulative={d.get(\"reward\", {}).get(\"cumulative_score\", \"?\")}')"

echo "[7/7] Task 1 - Finalize..."
curl -s -X POST ${HF_URL}/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "finalize",
      "payload": {},
      "confidence": 1.0
    }
  }' | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'✓ Task1 Finalize: cumulative={d.get(\"reward\", {}).get(\"cumulative_score\", \"?\")} done={d.get(\"done\", \"?\")}')"

echo -e "\n================================"
echo "All checks passed! ✓"
echo "================================"
```

Run it:
```bash
chmod +x test_hf_space.sh
./test_hf_space.sh
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Connection refused | Verify HF Space is deployed and running. Check URL is correct. |
| `{"detail":"Not Found"}` | Endpoint doesn't exist. Check `/docs` to see available endpoints. |
| Validation error | Check `confidence` is a float `0.0–1.0`, not a string. |
| Step fails with penalty | Review feedback in response — likely invalid `action_type` or wrong payload schema. |
| `KeyError: 'API_KEY'` | `API_KEY` env var not set. Set it explicitly or let the hackathon evaluator inject it. |
| `KeyError: 'API_BASE_URL'` | `API_BASE_URL` env var not set. Set it explicitly or let the hackathon evaluator inject it. |
| "No API calls through proxy" | Do **not** set `API_KEY` or `API_BASE_URL` as HF Space secrets — the evaluator injects them. |
