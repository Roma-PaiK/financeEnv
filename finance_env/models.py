from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    id: str
    source: str                          # "HDFC_CC" | "SBI_SAVINGS" | "PAYTM_UPI"
    description: str                     # Raw merchant string
    amount: float                        # Positive = debit, Negative = credit
    date: str                            # ISO 8601, e.g. "2024-01-15"
    # Hidden fields — present in raw data files, stripped from Observation by env.py
    correct_category: Optional[str] = None
    is_cc_settlement: Optional[bool] = None


class Observation(BaseModel):
    transactions: List[Transaction]
    account_balance: float
    monthly_income: float                # Fixed: 85000.0
    current_month: str                   # e.g. "2024-01"
    task_id: str                         # "task1" | "task2" | "task3"
    task_context: str                    # Natural language description of objective
    step_count: int
    sources_present: List[str]


class Action(BaseModel):
    action_type: Literal["categorize", "reconcile", "query", "set_budget", "finalize"]
    payload: Dict[str, Any]
    confidence: float = Field(..., ge=0.0, le=1.0)


class Reward(BaseModel):
    score: float                         # Score delta for THIS step
    partial_scores: Dict[str, float]
    feedback: str
    done: bool
    cumulative_score: float              # Normalized running total


class State(BaseModel):
    task_id: str
    step_count: int
    cumulative_score: float
    done: bool
    addressed_ids: List[str]
    budget_draft: Dict[str, float]       # Task 3 only
