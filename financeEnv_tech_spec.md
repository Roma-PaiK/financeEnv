# FinanceEnv — Technical Specification v2.0
### OpenEnv Hackathon | Personal Financial Intelligence Environment

---

## Part 0 — Quick Reference for AI Agents

> **If you are a coding agent reading this, these are your ground rules:**
> - All scoring is a `float` strictly in `[0.0, 1.0]`. Cumulative scores are normalized before being returned, never clipped.
> - All Pydantic models are defined in `finance_env/models.py`. Import from there; never redefine them.
> - Ground truth data lives in `finance_env/data/`. It is pre-generated, static JSON. Do not generate data at runtime.
> - The three task graders are completely independent modules. They share no state.
> - `step()` is the only function that mutates environment state. `state()` is read-only. `reset()` reinitializes.

---

## Part 1 — Problem Statement & Value Proposition

### The Real Pain Point
Every bank, credit card, and UPI app categorizes spending **within its own silo**. A person with one HDFC credit card, one SBI savings account, and Paytm UPI ends up with three separate "Food & Dining" totals that overlap, conflict, and double-count each other.

Specifically: when the SBI account pays the HDFC CC bill, it logs a `HDFC CREDIT CARD PAYMENT ₹18,450` as an outflow. The HDFC CC shows every merchant charge that makes up that ₹18,450. A naive aggregation of both sources inflates total spend by ₹18,450.

No consumer tool solves this. FinanceEnv trains agents to do exactly this — ingest multi-source transaction data, reconcile across sources, detect duplication, and build a forward-looking plan from the clean, unified view.

### What FinanceEnv Is
A structured OpenEnv environment where an AI agent acts as a personal financial intelligence assistant for a simulated salaried individual. The agent reads multi-source transaction data, takes structured actions, and is scored on the quality of its financial reasoning.

### Why It Is Not a Toy
Every task mirrors something a human financial advisor, a diligent individual, or a fintech reconciliation engine actually does — with data that is genuinely ambiguous, multi-source, and requires reasoning across time.

---

## Part 2 — Data Architecture

### 2.1 The Persona
All synthetic data represents a single persona across three tasks:

```
Name:           Ananya Sharma (fictional)
City:           Mumbai, Maharashtra
Occupation:     Software Engineer
Monthly Income: ₹85,000 (fixed salary credit, 1st of month)
Accounts:       HDFC Credit Card, SBI Savings Account, Paytm UPI
```

### 2.2 Synthetic Data Files (Pre-Generated, Static)

All files live in `finance_env/data/`. They are committed to the repo and never generated at runtime.

```
finance_env/data/
├── task1_transactions.json        # 20 single-source transactions (SBI savings, Jan)
├── task2_multi_source.json        # 3 months of transactions across 3 sources
├── task2_ground_truth.json        # Known duplicate pairs + expected reconciled totals
├── task3_history.json             # 2 months of categorized, reconciled spend history
├── task3_goals.json               # Stated financial goals for the planning task
├── task3_simulation.json          # Pre-baked "Month 3" actual spend for simulation
└── categories.json                # Canonical category taxonomy (shared across tasks)
```

### 2.3 Transaction Schema

Every transaction across all tasks conforms to this schema:

```python
class Transaction(BaseModel):
    id: str                    # Unique ID, e.g. "txn_sbi_001"
    source: str                # "HDFC_CC" | "SBI_SAVINGS" | "PAYTM_UPI"
    description: str           # Raw merchant string, e.g. "ZOMATO*ORDER 8829"
    amount: float              # Positive = debit. Negative = credit (salary, refunds).
    date: str                  # ISO 8601, e.g. "2024-01-15"
    correct_category: str      # HIDDEN from agent. Used by grader only.
    is_cc_settlement: bool     # True if this is a credit card bill payment row.
                               # HIDDEN from agent. Used by Task 2 grader.
```

> **Critical implementation note:** `correct_category` and `is_cc_settlement` are included in the raw data file but **must be stripped** from the `Observation` object before it is returned to the agent. The `env.py` layer handles this stripping. The grader imports the raw file directly.

### 2.4 Category Taxonomy (Canonical, Shared)

Defined in `categories.json`. Exactly 9 categories. No other strings are valid.

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

## Part 3 — OpenEnv Pydantic Models

Defined in `finance_env/models.py`. These are the **only** contracts between the environment and the agent.

### 3.1 Observation

```python
class Observation(BaseModel):
    transactions: List[Transaction]    # Stripped of hidden fields
    account_balance: float             # INR
    monthly_income: float              # Fixed: 85000.0
    current_month: str                 # e.g. "2024-01"
    task_id: str                       # "task1" | "task2" | "task3"
    task_context: str                  # Natural language description of objective
    step_count: int                    # Steps taken so far this episode
    sources_present: List[str]         # Which sources are in this observation
                                       # e.g. ["HDFC_CC", "SBI_SAVINGS"]
```

### 3.2 Action

```python
class Action(BaseModel):
    action_type: Literal[
        "categorize",           # Task 1 only
        "reconcile",            # Task 2 only
        "query",                # Task 2 and Task 3
        "set_budget",           # Task 3 only
        "finalize"              # Task 2 and Task 3
    ]
    payload: Dict[str, Any]     # Schema varies by action_type. See Section 5 per task.
    confidence: float           # 0.0–1.0. Required. Omitting forfeits calibration bonus.
```

> **Action payload schemas are defined per task in Section 5.** The `env.py` dispatcher validates that the `action_type` is legal for the active `task_id` before calling the grader.

### 3.3 Reward

```python
class Reward(BaseModel):
    score: float                       # Score delta for THIS step. 0.0–1.0.
    partial_scores: Dict[str, float]   # Breakdown. Keys are standardized per task.
    feedback: str                      # Plain English explanation of this step's score.
    done: bool                         # True if episode is complete.
    cumulative_score: float            # Normalized running total for this episode.
```

### 3.4 State

```python
class State(BaseModel):
    task_id: str
    step_count: int
    cumulative_score: float
    done: bool
    addressed_ids: List[str]           # Transaction IDs already acted on this episode.
    budget_draft: Dict[str, float]     # Task 3 only: current in-progress budget.
```

---

## Part 4 — Core Environment (`env.py`)

### 4.1 Public API

```python
class FinanceEnv:
    def reset(self, task_id: str) -> Observation: ...
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]: ...
    def state(self) -> State: ...
```

### 4.2 `reset(task_id)` Contract

- Clears all episode state: `step_count = 0`, `cumulative_score = 0.0`, `addressed_ids = []`, `budget_draft = {}`
- Loads the appropriate data file for `task_id`
- Strips hidden fields from transaction objects
- Returns a clean `Observation`
- Raises `ValueError` if `task_id` not in `["task1", "task2", "task3"]`

### 4.3 `step(action)` Contract

```
1. Validate action_type is legal for current task_id → if not, return penalty reward, do NOT advance state
2. Validate action payload schema → if invalid Pydantic, return penalty reward
3. Check for repeated action on same transaction_id → if 3+ repeats, apply loop penalty
4. Dispatch to correct grader module
5. Get (score_delta, partial_scores, feedback, done) from grader
6. Compute new cumulative_score = clamp(cumulative_score + score_delta, 0.0, 1.0)
7. Increment step_count
8. If step_count >= max_steps for task → set done = True, feedback += " [MAX STEPS REACHED]"
9. Return (new_observation, Reward, done, {})
```

### 4.4 Normalization Contract

Each grader returns raw `score_delta` values. The `env.py` accumulator ensures `cumulative_score` never exceeds `1.0`. Individual `score_delta` values returned in the `Reward.score` field are the raw grader values (may sum to more than 1.0 in theory; the cap is enforced only on `cumulative_score`).

### 4.5 Max Steps Per Task

```
task1: 25 steps
task2: 40 steps
task3: 60 steps
```

### 4.6 Universal Penalties (All Tasks)

| Trigger | Score Delta | Implementation Location |
|---|---|---|
| `action_type` not legal for active task | −0.020 | `env.py` dispatcher |
| Failed Pydantic validation on payload | −0.010 | `env.py` dispatcher |
| Same `transaction_id` actioned 3+ times | −0.030 per repeat | `env.py` dispatcher |
| Max steps exceeded | Hard stop, score capped | `env.py` step counter |

---

## Part 5 — The Three Tasks

---

### Task 1 — Transaction Categorisation (Easy)

**Max Steps:** 25 | **Data:** `task1_transactions.json` | **Grader:** `tasks/task1_categorize.py`

#### Objective
The agent is given 20 raw transactions from a single source (SBI Savings, January 2024). It must assign each to the correct category from the canonical taxonomy.

#### Observation at Reset
`transactions`: 20 Transaction objects (stripped of `correct_category`), single source `SBI_SAVINGS`.

#### Action Payload Schema

```python
# action_type: "categorize"
payload = {
    "transaction_id": str,    # Must match a valid ID in current episode's transactions
    "category": str           # Must be one of the 9 canonical categories
}
```

#### Grader Logic

```
For each "categorize" action:
  - If transaction_id not in episode data → feedback: "Unknown ID", score_delta: 0.0
  - If transaction_id already in addressed_ids → score_delta: -0.020, feedback: "Duplicate action"
  - If category not in canonical taxonomy → score_delta: -0.010, feedback: "Invalid category"
  - If category == correct_category → score_delta: +0.040, feedback: "Correct."
  - If category == parent_category[correct_category] (near-miss) → score_delta: +0.015
  - Else → score_delta: 0.0, feedback: "Incorrect. Expected [Y], received [X]."

On "finalize" action (or when all 20 addressed):
  - done = True
  - If all 20 addressed AND step_count < 22 → efficiency_bonus: +0.050
  - terminal_bonus: +0.050 (always on finalize, regardless of accuracy)
```

#### Scoring Weights (Normalized to 1.0)

| Component | Max Contribution |
|---|---|
| 20 correct categories × 0.040 | 0.800 |
| Efficiency bonus (done < 22 steps) | 0.050 |
| Terminal bonus | 0.050 |
| Near-miss partial credit | up to 0.100 (replaces 0.040 per near-miss) |

> Total max (all correct + efficient) = 0.80 + 0.05 + 0.05 = 0.90. Remaining 0.10 headroom absorbed by `cumulative_score` cap at 1.0.

#### Partial Scores Keys

```python
partial_scores = {
    "correct_labels": float,    # Count of exact matches so far / 20
    "efficiency": float,        # 0.0 or 0.050 (only set on finalize)
}
```

#### Expected GPT-4o Baseline: 0.74 – 0.82

---

### Task 2 — Cross-Source Deduplication & Reconciliation (Medium)

**Max Steps:** 40 | **Data:** `task2_multi_source.json`, `task2_ground_truth.json` | **Grader:** `tasks/task2_reconcile.py`

#### Objective
The agent receives 3 months of transactions across **three sources**: HDFC Credit Card, SBI Savings, and Paytm UPI. It must identify all instances of double-counting (primarily: SBI rows that are credit card bill payments for charges already listed in the HDFC CC feed), classify each row correctly, and produce a reconciled monthly spend total per category.

#### The Core Problem to Solve
```
HDFC CC Feed:  ZOMATO*8829    ₹450   (Jan 12)
               SWIGGY*2210    ₹320   (Jan 18)
               ... 14 more charges totaling ₹18,450

SBI Savings:   HDFC CC PAYMENT  ₹18,450  (Feb 5)  ← THIS IS A DUPLICATE
```

The `HDFC CC PAYMENT` row in SBI is not a spend event — it is the settlement of the CC bill. Counting both the CC charges AND the SBI payment inflates "total January spend" by ₹18,450.

#### Observation at Reset
`transactions`: 3 months of transactions, all three sources mixed. `sources_present: ["HDFC_CC", "SBI_SAVINGS", "PAYTM_UPI"]`.

#### Action Payload Schemas

```python
# action_type: "query"
# Request additional context on a specific merchant or date range
payload = {
    "query_type": "merchant" | "date_range",
    "value": str    # merchant name substring OR "YYYY-MM-DD:YYYY-MM-DD"
}

# action_type: "reconcile"
# Flag a transaction row with a classification
payload = {
    "transaction_id": str,
    "classification": "genuine_spend" | "cc_settlement" | "internal_transfer" | "refund",
    "linked_source": str | None   # If cc_settlement, the source this settles (e.g. "HDFC_CC")
}

# action_type: "finalize"
# Submit the complete reconciliation with unified category totals
payload = {
    "reconciled_totals": {
        "2024-01": {"Food & Dining": float, "Transport & Commute": float, ...},
        "2024-02": {...},
        "2024-03": {...}
    },
    "excluded_ids": List[str]   # IDs the agent classified as non-spend (settlements, transfers)
}
```

#### Grader Logic

```
For each "reconcile" action:
  - If classification == correct value in ground truth → score_delta: +0.060
  - If classification wrong → score_delta: -0.040, feedback: "Misclassified [txn_id]."
  - If transaction already addressed → score_delta: -0.020

For "query" actions:
  - Returns matching transactions from the observation as additional context
  - score_delta: +0.010 if the queried merchant/range contains unresolved duplicates
  - score_delta: -0.010 if all transactions in that query range are already resolved
  - (Useful query = narrows an unresolved problem. Redundant = costs the agent.)

For "finalize" action:
  - Grade excluded_ids:
      true_positives = len(excluded_ids ∩ ground_truth_duplicates)
      false_positives = len(excluded_ids - ground_truth_duplicates)
      false_negatives = len(ground_truth_duplicates - excluded_ids)
      precision = TP / (TP + FP) if any exclusions else 0
      recall = TP / (TP + FN)
      f1 = 2 * precision * recall / (precision + recall) if denominator > 0 else 0
      score_delta for exclusion accuracy: f1 * 0.200

  - Grade reconciled_totals per month:
      For each category in each month:
        abs_error_pct = abs(submitted - ground_truth) / ground_truth
        if abs_error_pct <= 0.02: score_delta += 0.010  (near-perfect)
        elif abs_error_pct <= 0.10: score_delta += 0.005  (close)
        else: 0.0
      (Max from totals accuracy: ~0.180 across all categories and months)

  - done = True
```

#### Partial Scores Keys

```python
partial_scores = {
    "classification_accuracy": float,  # Running: correct classifications / total actioned
    "exclusion_f1": float,             # Set on finalize
    "totals_accuracy": float,          # Set on finalize: avg category accuracy across months
}
```

#### Expected GPT-4o Baseline: 0.48 – 0.60

---

### Task 3 — Forward Budget Planning with Life Event Shock (Hard)

**Max Steps:** 60 | **Data:** `task3_history.json`, `task3_goals.json`, `task3_simulation.json` | **Grader:** `tasks/task3_budget.py`

#### Objective
Given 2 months of clean (already reconciled) transaction history, stated financial goals, and fixed obligations, the agent must construct a realistic category-level monthly budget. After `finalize`, the environment runs a **pre-baked simulation** of the next month against the agent's plan — including a deterministic life event shock — and scores adherence.

#### The Simulation Design
`task3_simulation.json` is a static, pre-generated "Month 3" actual spend record. It represents realistic (not perfectly obedient) spending behavior. It includes exactly one life event shock:

```json
{
  "life_event": {
    "description": "Car repair at service center",
    "amount": 7500.0,
    "category": "Other",
    "day": 15
  }
}
```

The agent does not know about the life event in advance. The budget it constructs will be tested against this static ground truth. Agents that build in a buffer in "Other" or "Savings & Investment" will absorb the shock gracefully.

#### Observation at Reset
`transactions`: 2 months of pre-reconciled history (from `task3_history.json`) tagged with `correct_category`. Goals are embedded in `task_context` as plain text. Example:
```
"Save ₹8,000 this month. Reduce Food & Dining spend by 15% vs last month. 
 Maintain zero deficit across all categories. Fixed EMI: ₹12,000."
```

#### Action Payload Schemas

```python
# action_type: "query"
# Explore history for a specific category
payload = {
    "category": str,           # Must be in canonical taxonomy
    "months": List[str]        # e.g. ["2024-01", "2024-02"]
}

# action_type: "set_budget"
# Set or revise allocation for a single category
payload = {
    "category": str,
    "amount": float   # INR, must be >= 0
}

# action_type: "finalize"
# Lock the budget and trigger simulation
payload = {
    "budget": Dict[str, float]   # Full category → amount mapping
                                 # Must include all 9 categories
}
```

#### Grader Logic — Pre-Finalize (Step-by-Step Signals)

```
For each "set_budget" action:
  - If category not in taxonomy → score_delta: -0.010
  - If amount == 0 AND historical avg spend > 0 for that category → score_delta: -0.060
    feedback: "Unrealistic: [Category] had ₹[X] historical spend. Cannot be zero."
  - If amount within ±30% of historical average for that category → score_delta: +0.020
    feedback: "Realistic allocation for [Category]. Behavioural baseline respected."
  - Multiple revisions to same category are allowed with no penalty.

For each "query" action:
  - Returns historical spend data for requested category and months.
  - score_delta: +0.015 if this category has not been set_budget yet
  - score_delta: -0.010 if category has already been finalized in budget_draft
```

#### Grader Logic — Finalize (Simulation)

```
Pre-simulation validation:
  - budget_sum = sum(budget.values())
  - If budget_sum > monthly_income (₹85,000): score_delta: -0.200, done=True
    feedback: "Invalid: budget exceeds income by ₹[X]. Plan rejected."
  - If any required category missing: score_delta: -0.050 per missing category
  - If "finalize" called with NO prior "query" actions: score_delta: -0.050
    feedback: "No exploration. Budget submitted blindly."

Budget validity bonus (if sum <= income):
  score_delta: +0.100

Run simulation against task3_simulation.json:
  For each category in simulation:
    actual_spend = simulation[category]
    budgeted = budget[category]
    if actual_spend <= budgeted: category_adhered = True
    else: category_adhered = False, overspend = actual_spend - budgeted

  Adherence score:
    adhered_count = count of categories where actual <= budgeted
    if adhered_count == 9: score_delta += 0.150  (perfect adherence)
    elif adhered_count >= 7: score_delta += 0.080
    elif adhered_count >= 5: score_delta += 0.040
    else: score_delta += 0.000

  Savings goal:
    sim_savings = monthly_income - sum(simulation.values())
    if sim_savings >= goal_amount: score_delta += 0.200
    elif sim_savings >= goal_amount * 0.5: score_delta += 0.100
    else: score_delta += 0.000

  Life event absorption:
    life_event_amount = 7500.0
    buffer = budget["Other"] + max(0, budget["Savings & Investment"] - goal_savings_amount)
    if buffer >= life_event_amount: score_delta += 0.100
      feedback: "Life event absorbed. Budget had sufficient buffer."
    elif buffer >= life_event_amount * 0.5: score_delta += 0.050
    else: score_delta += 0.000
      feedback: "Life event unabsorbed. ₹7,500 repair exceeded available buffer."

  Essential category underfunding check:
    essential = ["EMI & Loan Repayment", "Utilities & Bills", "Healthcare"]
    for each essential category:
      if budget[cat] < historical_avg[cat] * 0.5:
        score_delta -= 0.080 each
        feedback: "Risk: [Category] critically underfunded."

  done = True
```

#### Partial Scores Keys

```python
partial_scores = {
    "budget_validity": float,      # 0.0 or 0.100
    "adherence": float,            # 0.0 – 0.150
    "savings_goal": float,         # 0.0, 0.100, or 0.200
    "life_event_absorption": float,# 0.0, 0.050, or 0.100
    "realism": float,              # Running: realistic set_budget actions / total
}
```

#### Expected GPT-4o Baseline: 0.35 – 0.48

---

## Part 6 — Baseline Inference Script (`baseline/run_baseline.py`)

### Logic

```
1. Load OPENAI_API_KEY from os.environ. Raise ValueError if missing.
2. For each task_id in ["task1", "task2", "task3"]:
   a. env.reset(task_id) → initial observation
   b. Build system prompt:
      - Include task_context from observation
      - Include full Action schema as JSON comment
      - Instruction: "Respond ONLY with a valid JSON object matching the Action schema."
   c. Loop until done:
      - Call OpenAI chat completions with response_format: {"type": "json_object"}
      - Parse JSON response → instantiate Action via Pydantic
      - If Pydantic validation fails: log error, submit penalty action
      - env.step(action) → (obs, reward, done, info)
      - Append reward.feedback to conversation history
      - Log step: task_id, step_count, score_delta, cumulative_score
3. Print final scores table:
   | Task | Description           | Difficulty | Score |
   | task1 | Categorisation       | Easy       | X.XX  |
   | task2 | Reconciliation       | Medium     | X.XX  |
   | task3 | Budget Planning      | Hard       | X.XX  |
```

### Reproducibility Requirement
The baseline script must produce **deterministic scores** on each run. This is guaranteed because:
- Synthetic data is static (pre-generated, committed to repo)
- Graders are deterministic (no random elements)
- OpenAI API calls use `temperature=0` and `seed=42`

---

## Part 7 — Deployment

### Project File Structure

```
financeenv/
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
├── app.py                              # HF Spaces FastAPI entry point
├── finance_env/
│   ├── __init__.py
│   ├── models.py                       # All Pydantic models
│   ├── env.py                          # FinanceEnv class
│   ├── data/
│   │   ├── task1_transactions.json
│   │   ├── task2_multi_source.json
│   │   ├── task2_ground_truth.json
│   │   ├── task3_history.json
│   │   ├── task3_goals.json
│   │   ├── task3_simulation.json
│   │   └── categories.json
│   └── tasks/
│       ├── task1_categorize.py
│       ├── task2_reconcile.py
│       └── task3_budget.py
└── baseline/
    └── run_baseline.py
```

### openenv.yaml

```yaml
name: FinanceEnv
version: 1.0.0
description: >
  Multi-source personal finance environment. Trains agents to categorize
  transactions, reconcile cross-source duplicates, and build realistic budgets.
tasks:
  - id: task1
    name: Transaction Categorisation
    difficulty: easy
    max_steps: 25
  - id: task2
    name: Cross-Source Reconciliation
    difficulty: medium
    max_steps: 40
  - id: task3
    name: Forward Budget Planning
    difficulty: hard
    max_steps: 60
action_types:
  - categorize
  - reconcile
  - query
  - set_budget
  - finalize
tags:
  - openenv
  - finance
  - real-world
```

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### app.py (FastAPI — Minimal)

Expose three endpoints. No UI required for spec compliance.

```
POST /reset         → body: {"task_id": str}          → returns Observation JSON
POST /step          → body: Action JSON                → returns {observation, reward, done, info}
GET  /state         → returns State JSON
```

---

## Part 8 — Shared Interfaces (API Contracts Between Streams)

### Interface A→C: Grader Return Type
Every grader module (`task1_categorize.py`, etc.) must return this exact dict to `env.py`:

```python
{
    "score_delta": float,          # Raw score change for this action
    "partial_scores": dict,        # Keys defined per task in Section 5
    "feedback": str,               # Plain English explanation
    "done": bool,                  # Whether episode is complete
    "addressed_ids": List[str]     # Updated list of acted-upon IDs
}
```

### Interface C→B: HTTP Response from FastAPI

`POST /step` returns:

```json
{
  "observation": {
    "transactions": [...],
    "account_balance": 42300.0,
    "monthly_income": 85000.0,
    "current_month": "2024-01",
    "task_id": "task1",
    "task_context": "...",
    "step_count": 3,
    "sources_present": ["SBI_SAVINGS"]
  },
  "reward": {
    "score": 0.04,
    "partial_scores": {"correct_labels": 0.25, "efficiency": 0.0},
    "feedback": "Correct. Category confirmed as Food & Dining.",
    "done": false,
    "cumulative_score": 0.12
  },
  "done": false,
  "info": {}
}
```

### Interface B→C: Action Payload from Frontend/Agent

`POST /step` body:

```json
{
  "action_type": "categorize",
  "payload": {
    "transaction_id": "txn_sbi_004",
    "category": "Food & Dining"
  },
  "confidence": 0.85
}
```

---

## Part 9 — Workstream Breakdown (3-Way Split)

See Part 10 for the full atomic task checklist per stream.

| Stream | Owner | Core Responsibility |
|---|---|---|
| **Stream A** | Dev 1 | Data generation, all grader logic, env.py core |
| **Stream B** | Dev 2 | Baseline inference script, openenv.yaml, README |
| **Stream C** | Dev 3 | FastAPI app.py, Dockerfile, HF deployment, models.py |

**Dependency rule:** Stream C depends on Stream A's `models.py` and grader return contract. Stream B depends on Stream C's running endpoints. Stream A and Stream B can proceed fully in parallel.

---

## Part 10 — Atomic Task Checklists

### Stream A — Core Logic & Graders

> Owner builds: synthetic data, models.py, env.py, all three task graders.

**A1 — Data Generation**

- [ ] `A1.1` Write a Python script `scripts/generate_data.py` that creates `task1_transactions.json`: 20 Transaction objects for a single SBI Savings source (January 2024). Each object must include `correct_category` (hidden from agent), `is_cc_settlement: false` for all records, and realistic Indian merchant description strings (e.g. "ZOMATO*ORDER 8829", "IRCTC BOOKING 4992", "HDFC EMI AUTO-DEBIT"). Commit the generated JSON to `finance_env/data/`.

- [ ] `A1.2` Extend `generate_data.py` to create `task2_multi_source.json`: 3 months (Jan–Mar 2024) of transactions across three sources: HDFC_CC, SBI_SAVINGS, PAYTM_UPI. Include exactly 6 credit card settlement rows in SBI_SAVINGS (2 per month) where `is_cc_settlement: true` and the amount matches the sum of HDFC_CC charges in the prior billing period. Include 4 internal UPI transfer rows. Ensure every non-settlement/non-transfer row has a `correct_category`.

- [ ] `A1.3` Write `generate_data.py` section for `task2_ground_truth.json`. This file must contain: (a) a list of all transaction IDs where `is_cc_settlement: true` or `classification == "internal_transfer"`, (b) the correct reconciled monthly category totals for Jan, Feb, Mar after removing those IDs from the spend calculation.

- [ ] `A1.4` Write `generate_data.py` section for `task3_history.json`: 2 months of pre-categorized, pre-reconciled transactions representing Ananya's clean spending history (no duplicates, no settlements). Include salary credit rows (₹85,000 on 1st of each month).

- [ ] `A1.5` Create `task3_goals.json` with this structure: `{"savings_target": 8000, "category_reduction_targets": {"Food & Dining": 0.15}, "fixed_obligations": {"EMI & Loan Repayment": 12000}}`.

- [ ] `A1.6` Create `task3_simulation.json`: static pre-baked "Month 3" actual spend per category. Must include `"life_event": {"description": "Car repair", "amount": 7500, "category": "Other", "day": 15}`. The non-life-event spend should represent realistic human behavior (not perfectly budget-adherent — allow 1–2 categories to slightly overspend). Compute and include `"actual_savings"` field.

- [ ] `A1.7` Create `categories.json` as a simple JSON array of the 9 canonical category strings.

**A2 — Pydantic Models**

- [ ] `A2.1` Write `finance_env/models.py` defining `Transaction`, `Observation`, `Action`, `Reward`, and `State` exactly as specified in Part 3. Add a `@validator` on `Action.action_type` that confirms it is one of the 5 allowed literals. Add a `@validator` on `Action.confidence` that enforces `0.0 <= confidence <= 1.0`.

- [ ] `A2.2` Add a `strip_hidden_fields()` method to `Transaction` (or a utility function in `models.py`) that returns a copy of the Transaction with `correct_category` set to `None` and `is_cc_settlement` set to `None`. This function is called by `env.py` before returning Observations.

**A3 — Core Environment**

- [ ] `A3.1` Write `finance_env/env.py` implementing `FinanceEnv` with `reset(task_id)`, `step(action)`, and `state()` per the contracts in Part 4. Implement the dispatcher that validates `action_type` legality per task_id. Implement universal penalties. Implement `cumulative_score` normalization (clamp at 1.0).

- [ ] `A3.2` Write unit tests in `tests/test_env.py`. Test: `reset()` clears state correctly, illegal `action_type` returns penalty reward, invalid payload returns penalty reward, `step_count` increments, `done=True` at max steps.

**A4 — Task 1 Grader**

- [ ] `A4.1` Write `finance_env/tasks/task1_categorize.py` implementing the grader logic from Section 5 Task 1. The function signature must be: `def grade(action: Action, episode_state: dict, ground_truth: dict) -> dict`. Return value must match the grader return type in Part 8 Interface A→C. Load `task1_transactions.json` and `categories.json` internally.

- [ ] `A4.2` Write unit tests for Task 1 grader in `tests/test_task1.py`. Test: exact match scores +0.040, near-miss scores +0.015, wrong category scores 0.0, duplicate action scores -0.020, invalid category string scores -0.010, efficiency bonus triggers correctly.

**A5 — Task 2 Grader**

- [ ] `A5.1` Write `finance_env/tasks/task2_reconcile.py`. Implement F1 scoring for `excluded_ids` on `finalize`. Implement per-category accuracy scoring for `reconciled_totals`. Implement query usefulness logic (useful = query contains at least one unresolved transaction). Load `task2_multi_source.json` and `task2_ground_truth.json` internally.

- [ ] `A5.2` Write unit tests for Task 2 grader in `tests/test_task2.py`. Test: correct `cc_settlement` classification scores +0.060, wrong classification scores -0.040, finalize with perfect exclusions scores F1=1.0 → +0.200, redundant query scores -0.010.

**A6 — Task 3 Grader**

- [ ] `A6.1` Write `finance_env/tasks/task3_budget.py`. Implement pre-simulation step rewards (`set_budget` realism scoring). Implement the full finalize simulation logic from Section 5 Task 3: budget validity check, adherence scoring against `task3_simulation.json`, savings goal evaluation, life event absorption check, essential category underfunding penalties.

- [ ] `A6.2` Write unit tests for Task 3 grader in `tests/test_task3.py`. Test: budget > income returns -0.200 and done=True, zero allocation for a historically-used category returns -0.060, perfect adherence scores +0.150, life event buffer ≥ ₹7,500 scores +0.100.

---

### Stream B — Baseline Script & Documentation

> Owner builds: baseline inference script, openenv.yaml, README. Requires Stream C endpoints to be running for end-to-end testing but can be built against the contract independently.

**B1 — Baseline Inference Script**

- [ ] `B1.1` Write `baseline/run_baseline.py`. Implement the loop described in Part 6: load API key from env, reset environment, build system prompt from observation, call OpenAI API with `temperature=0, seed=42, response_format={"type": "json_object"}`, parse response to Action, call `env.step()`, repeat until done. Log each step to console with format: `[task1][step 3] score_delta=+0.04 cumulative=0.12 feedback="Correct."`.

- [ ] `B1.2` Add error handling to `run_baseline.py`: if Pydantic validation fails on model response, log the raw response and submit a fallback action `{"action_type": "finalize", "payload": {}, "confidence": 0.0}`. This ensures the script never crashes on a malformed LLM response.

- [ ] `B1.3` Add final summary output to `run_baseline.py` that prints a Markdown table of results:
```
| Task  | Description              | Difficulty | Score |
|-------|--------------------------|------------|-------|
| task1 | Transaction Categorisation | Easy     | 0.78  |
| task2 | Cross-Source Reconciliation | Medium  | 0.54  |
| task3 | Forward Budget Planning   | Hard      | 0.41  |
```
Also write results to `baseline/results.json` for reproducibility.

- [ ] `B1.4` Write a `baseline/prompts.py` file containing the system prompt templates for each task. Prompts must include: (a) task context from observation, (b) full Action schema, (c) canonical category list for Task 1, (d) instruction to output only valid JSON.

**B2 — Documentation**

- [ ] `B2.1` Write `openenv.yaml` exactly as specified in Part 7.

- [ ] `B2.2` Write `README.md` with these sections: (1) Environment Description & Motivation — lead with the multi-source pain point, (2) Persona (Ananya's profile), (3) Action Space — table of all action_types with payloads, (4) Observation Space — table of all fields, (5) Task Descriptions — one subsection per task with objective, difficulty, max_steps, (6) Setup Instructions — docker build + run, pip install, (7) Baseline Scores — table from B1.3.

- [ ] `B2.3` Add a `CONTRIBUTING.md` (optional but high signal to judges): explain how to add a new task by implementing the grader return interface.

---

### Stream C — Integration, API & Deployment

> Owner builds: models.py (can use A2's output directly), FastAPI app, Dockerfile, HF deployment.

**C1 — FastAPI Application**

- [ ] `C1.1` Write `app.py` implementing a FastAPI application with three endpoints: `POST /reset`, `POST /step`, `GET /state`. Each endpoint instantiates `FinanceEnv` from `finance_env/env.py`. The app must maintain a single environment instance in memory (module-level singleton is fine for this use case). Add basic error handling: if `env.step()` raises an exception, return HTTP 422 with the error message.

- [ ] `C1.2` Add request/response models to `app.py` using the Pydantic models from `models.py` for automatic OpenAPI documentation generation. This means FastAPI will generate `/docs` automatically — which is useful evidence for judges that the API is spec-compliant.

- [ ] `C1.3` Add a `GET /health` endpoint that returns `{"status": "ok", "tasks": ["task1", "task2", "task3"]}`. This is used by HF Spaces to confirm the container is running.

- [ ] `C1.4` Write `requirements.txt` with pinned versions: `fastapi==0.110.0`, `uvicorn==0.27.1`, `pydantic==2.6.3`, `openai==1.12.0`. Run `pip freeze` to capture any transitive dependencies.

**C2 — Containerization**

- [ ] `C2.1` Write `Dockerfile` as specified in Part 7. Verify locally: `docker build -t financeenv .` completes without errors. `docker run -p 7860:7860 financeenv` starts successfully and `curl localhost:7860/health` returns 200.

- [ ] `C2.2` Create a `.dockerignore` file that excludes: `__pycache__`, `*.pyc`, `.git`, `tests/`, `scripts/`, `.env`. This keeps the container image small.

- [ ] `C2.3` Test full Docker roundtrip: `docker run`, `curl POST /reset {"task_id": "task1"}`, `curl POST /step` with a valid Action, verify response matches the contract in Part 8.

**C3 — HF Spaces Deployment**

- [ ] `C3.1` Create a Hugging Face Space named `financeenv` with SDK set to `docker`. Push the repo. Verify the Space builds successfully from the Dockerfile.

- [ ] `C3.2` Add the `openenv` tag to the HF Space metadata. Add a Space README header with `tags: [openenv, finance, real-world]`.

- [ ] `C3.3` Confirm that `GET /health` returns 200 on the deployed Space URL. Document the Space URL in `README.md`.

**C4 — Integration Testing**

- [ ] `C4.1` Write `tests/test_integration.py`. This test hits the live FastAPI endpoints (not the HF Space — the local Docker container). Test the full lifecycle: `POST /reset task1` → verify Observation shape, `POST /step` with a valid categorize action → verify Reward shape, `GET /state` → verify State shape.

- [ ] `C4.2` Run `openenv validate` against the deployed environment. Confirm it passes. Document the output in `README.md`.

---

## Part 11 — Integration & Sync Plan

### Merge Order

```
Day 1–3: Streams A and B work fully in parallel. No coordination needed.
Day 4:   Stream C integrates Stream A's modules (import env.py, models.py into app.py).
         Stream B runs baseline script against Stream C's local Docker container.
Day 5:   Integration test, HF deploy, README finalize, openenv validate.
```

### The One Shared File to Coordinate Early
`finance_env/models.py` is the only file that all three streams depend on. Recommendation: **Dev (Stream C) writes the initial models.py on Day 1 based on Part 3 of this spec**, commits it to the repo, and Dev A imports from it rather than redefining. This prevents model divergence.

### How to Test Without Blocking Each Other
- Stream A: run grader unit tests directly against grader functions (`python -m pytest tests/test_task1.py`). No HTTP needed.
- Stream B: run baseline script against a **mock server** (`python -m pytest --mock-server`) that returns dummy Observations. The mock is a 20-line FastAPI app hardcoded to return valid Observation/Reward shapes.
- Stream C: test FastAPI endpoints with hardcoded dummy graders (grader functions that return `{"score_delta": 0.05, "done": False, ...}`) before Stream A's real graders are ready.

### Definition of "Done" for Integration
The build is integration-complete when all three pass simultaneously:
1. `python -m pytest tests/` — all unit and integration tests pass
2. `python baseline/run_baseline.py` — prints final scores table without errors
3. `openenv validate` — passes