"""Pydantic contracts shared across env, graders, and server. Never redefine these elsewhere."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Transaction(BaseModel):
    """One bank/UPI/CC transaction row shown to the agent."""
    id: str
    source: str          # "HDFC_CC" | "SBI_SAVINGS" | "PAYTM_UPI"
    description: str     # Raw merchant string, e.g. "ZOMATO*ORDER 8829"
    amount: float        # Positive = debit, negative = credit
    date: str            # ISO 8601, e.g. "2024-01-15"
    # Present in raw data files but stripped from Observation by env.py before returning to agent
    correct_category: Optional[str] = None
    is_cc_settlement: Optional[bool] = None


class Observation(BaseModel):
    """What the agent sees at each step."""
    transactions: List[Transaction]
    account_balance: float
    monthly_income: float        # Fixed: 85000.0
    current_month: str           # e.g. "2024-01"
    task_id: str                 # "task1" | "task2" | "task3"
    task_context: str            # Natural language description of the task objective
    step_count: int
    sources_present: List[str]
    done: bool = False           # Set by server adapter after each step
    reward: Optional[float] = None  # Cumulative score, set by server adapter


class Action(BaseModel):
    """What the agent submits at each step."""
    action_type: Literal["categorize", "reconcile", "query", "set_budget", "finalize"]
    payload: Dict[str, Any]
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator('confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        # Accept confidence as a string (some LLMs return "0.9" instead of 0.9)
        if isinstance(v, str):
            return float(v)
        return v


class Reward(BaseModel):
    """Step result returned by env.py. Not exposed directly over HTTP."""
    score: float                  # Delta for this step only
    partial_scores: Dict[str, float]
    feedback: str
    done: bool
    cumulative_score: float       # Normalized running total across all steps


class State(BaseModel):
    """Internal episode state. Read-only via GET /state."""
    task_id: str
    step_count: int
    cumulative_score: float
    done: bool
    addressed_ids: List[str]
    budget_draft: Dict[str, float]  # Task 3 only: category → allocated amount
