"""
FinanceEnv — Core Environment

Public API:
    env = FinanceEnv()
    obs  = env.reset(task_id)          # "task1" | "task2" | "task3"
    obs, reward, done, info = env.step(action)
    state = env.state()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from finance_env.models import Action, Observation, Reward, State, Transaction

DATA_DIR = Path(__file__).parent / "data"

TASK_DATA: Dict[str, List[str]] = {
    "task1": ["task1_transactions.json"],
    "task2": ["task2_multi_source.json", "task2_ground_truth.json"],
    "task3": ["task3_history.json", "task3_goals.json", "task3_simulation.json"],
}

MAX_STEPS: Dict[str, int] = {
    "task1": 2,
    "task2": 3,
    "task3": 4,
}

LEGAL_ACTIONS: Dict[str, List[str]] = {
    "task1": ["categorize", "finalize"],
    "task2": ["reconcile", "query", "finalize"],
    "task3": ["query", "set_budget", "finalize"],
}

TASK_CONTEXT: Dict[str, str] = {
    "task1": (
        "You are given 20 transactions from Ananya Sharma's SBI Savings account for January 2024. "
        "Assign each transaction to exactly one of the 9 canonical categories. "
        "Use 'categorize' for each transaction, then 'finalize' when done."
    ),
    "task2": (
        "You are given 3 months of transactions across HDFC Credit Card, SBI Savings, and Paytm UPI. "
        "Identify and flag duplicate/settlement rows, then submit reconciled monthly category totals."
    ),
    "task3": (
        "You are given 2 months of pre-reconciled spend history for Ananya Sharma (Jan–Feb 2024). "
        "Financial goals: Save ₹8,000 this month. Reduce Food & Dining by 15% vs last month. "
        "Maintain zero deficit across all categories. Fixed EMI: ₹12,000. Income: ₹85,000/month. "
        "Use 'query' to explore category history, 'set_budget' to allocate per category, "
        "then 'finalize' with all 9 categories to lock the plan."
    ),
}

HIDDEN_FIELDS = {"correct_category", "is_cc_settlement"}


def _strip_hidden(txn_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in txn_dict.items() if k not in HIDDEN_FIELDS}


class FinanceEnv:
    def __init__(self) -> None:
        self._task_id: str = ""
        self._raw_transactions: List[Dict[str, Any]] = []
        self._state: State = State(
            task_id="",
            step_count=0,
            cumulative_score=0.0,
            done=False,
            addressed_ids=[],
            budget_draft={},
        )
        self._correct_count: int = 0  # task1/task2: tracks exact match count for finalize
        self._query_count: int = 0    # task3: tracks number of query actions taken

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        if task_id not in TASK_DATA:
            raise ValueError(f"Unknown task_id '{task_id}'. Must be one of: {list(TASK_DATA)}")

        self._task_id = task_id
        self._raw_transactions = self._load_transactions(task_id)
        self._correct_count = 0
        self._query_count = 0

        self._state = State(
            task_id=task_id,
            step_count=0,
            cumulative_score=0.0,
            done=False,
            addressed_ids=[],
            budget_draft={},
        )

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # --- Universal penalty: illegal action_type for this task ---
        if action.action_type not in LEGAL_ACTIONS[self._task_id]:
            reward = self._make_reward(
                score_delta=-0.020,
                partial_scores={},
                feedback=(
                    f"'{action.action_type}' is not a valid action for {self._task_id}. "
                    f"Legal actions: {LEGAL_ACTIONS[self._task_id]}"
                ),
                done=False,
            )
            self._state.step_count += 1
            return self._build_observation(), reward, False, {}

        # --- Universal penalty: repeated transaction (3+ times) ---
        if action.action_type in ("categorize", "reconcile"):
            txn_id = action.payload.get("transaction_id", "")
            repeat_count = self._state.addressed_ids.count(txn_id)
            if repeat_count >= 2:
                reward = self._make_reward(
                    score_delta=-0.030,
                    partial_scores={},
                    feedback=f"Loop penalty: '{txn_id}' has been acted on {repeat_count + 1} times.",
                    done=False,
                )
                self._state.step_count += 1
                self._check_max_steps(reward)
                return self._build_observation(), reward, reward.done, {}

        # --- Dispatch to task grader ---
        score_delta, partial_scores, feedback, done = self._dispatch(action)

        # Track addressed IDs and correct count
        # Task 1: categorize actions
        if action.action_type == "categorize" and score_delta != -0.010:
            txn_id = action.payload.get("transaction_id", "")
            if txn_id and txn_id not in self._state.addressed_ids:
                self._state.addressed_ids.append(txn_id)
                if score_delta == 0.040:
                    self._correct_count += 1
            # Auto-finalize when all transactions addressed
            if len(self._state.addressed_ids) == len(self._raw_transactions) and not done:
                from finance_env.tasks.task1_categorize import grade_finalize
                fin_delta, fin_partial, fin_feedback, done = grade_finalize(
                    self._state.addressed_ids,
                    self._state.step_count + 1,
                    self._correct_count,
                )
                score_delta += fin_delta
                partial_scores.update(fin_partial)
                feedback += " " + fin_feedback

        # Task 2: reconcile actions
        if action.action_type == "reconcile" and score_delta not in (-0.010, -0.020):
            txn_id = action.payload.get("transaction_id", "")
            if txn_id and txn_id not in self._state.addressed_ids:
                self._state.addressed_ids.append(txn_id)
                if score_delta == 0.060:
                    self._correct_count += 1

        reward = self._make_reward(score_delta, partial_scores, feedback, done)
        self._state.step_count += 1
        self._state.done = done
        self._check_max_steps(reward)

        return self._build_observation(), reward, reward.done, {}

    def state(self) -> State:
        return self._state.model_copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(self, action: Action) -> Tuple[float, Dict[str, Any], str, bool]:
        if self._task_id == "task1":
            return self._dispatch_task1(action)
        elif self._task_id == "task2":
            return self._dispatch_task2(action)
        elif self._task_id == "task3":
            return self._dispatch_task3(action)
        raise NotImplementedError(f"{self._task_id} grader not yet implemented.")

    def _dispatch_task1(self, action: Action) -> Tuple[float, Dict[str, Any], str, bool]:
        from finance_env.tasks.task1_categorize import grade_categorize, grade_finalize

        if action.action_type == "categorize":
            return grade_categorize(action.payload, self._state.addressed_ids)

        if action.action_type == "finalize":
            return grade_finalize(
                self._state.addressed_ids,
                self._state.step_count,
                self._correct_count,
            )

        # Should never reach here — caught by legal action check above
        raise ValueError(f"Unexpected action_type '{action.action_type}' in task1 dispatch.")

    def _dispatch_task2(self, action: Action) -> Tuple[float, Dict[str, Any], str, bool]:
        from finance_env.tasks.task2_reconcile import (
            grade_reconcile,
            grade_query,
            grade_finalize,
        )

        if action.action_type == "reconcile":
            return grade_reconcile(action.payload, self._state.addressed_ids, self._correct_count)

        if action.action_type == "query":
            return grade_query(action.payload, self._state.addressed_ids, self._correct_count)

        if action.action_type == "finalize":
            return grade_finalize(action.payload, self._state.addressed_ids, self._correct_count)

        # Should never reach here
        raise ValueError(f"Unexpected action_type '{action.action_type}' in task2 dispatch.")

    def _dispatch_task3(self, action: Action) -> Tuple[float, Dict[str, Any], str, bool]:
        from finance_env.tasks.task3_budget import grade_set_budget, grade_query, grade_finalize

        if action.action_type == "set_budget":
            # grade_set_budget mutates _state.budget_draft in place
            return grade_set_budget(action.payload, self._state.budget_draft)

        if action.action_type == "query":
            self._query_count += 1
            return grade_query(action.payload, self._state.budget_draft)

        if action.action_type == "finalize":
            return grade_finalize(action.payload, self._state.budget_draft, self._query_count)

        # Should never reach here
        raise ValueError(f"Unexpected action_type '{action.action_type}' in task3 dispatch.")

    def _make_reward(
        self,
        score_delta: float,
        partial_scores: Dict[str, Any],
        feedback: str,
        done: bool,
    ) -> Reward:
        # Clamp cumulative_score to exclusive interval (1e-6, 1 - 1e-6)
        eps = 1e-6
        new_cumulative = min(1.0 - eps, max(eps, self._state.cumulative_score + score_delta))
        self._state.cumulative_score = new_cumulative
        return Reward(
            score=score_delta,
            partial_scores=partial_scores,
            feedback=feedback,
            done=done,
            cumulative_score=new_cumulative,
        )

    def _check_max_steps(self, reward: Reward) -> None:
        if self._state.step_count >= MAX_STEPS[self._task_id]:
            reward.done = True
            reward.feedback += " [MAX STEPS REACHED]"
            self._state.done = True

    def _load_transactions(self, task_id: str) -> List[Dict[str, Any]]:
        primary_file = DATA_DIR / TASK_DATA[task_id][0]
        with open(primary_file) as f:
            return json.load(f)

    def _build_observation(self) -> Observation:
        stripped = [Transaction(**_strip_hidden(t)) for t in self._raw_transactions]
        sources = list({t["source"] for t in self._raw_transactions})
        current_month = (
            max(t["date"][:7] for t in self._raw_transactions)
            if self._raw_transactions else ""
        )
        balance = 85000.0 - sum(t["amount"] for t in self._raw_transactions if t["amount"] > 0)

        return Observation(
            transactions=stripped,
            account_balance=round(balance, 2),
            monthly_income=85000.0,
            current_month=current_month,
            task_id=self._task_id,
            task_context=TASK_CONTEXT.get(self._task_id, ""),
            step_count=self._state.step_count,
            sources_present=sources,
        )
