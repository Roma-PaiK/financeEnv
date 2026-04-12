"""
Task 1 grader — Transaction Categorisation.
Scores "categorize" and "finalize" actions against ground truth.
Returns: (score_delta, partial_scores, feedback, done)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATA_FILE = Path(__file__).parent.parent / "data" / "task1_transactions.json"
CATEGORIES_FILE = Path(__file__).parent.parent / "data" / "categories.json"

MAX_TRANSACTIONS = 20
EFFICIENCY_STEP_THRESHOLD = 22  # finalize before this step to earn efficiency bonus

EPSILON = 1e-6

# Maps each category to its "parent" for near-miss scoring (+0.015 instead of 0.0)
PARENT_CATEGORY: Dict[str, str | None] = {
    "Food & Dining":                "Shopping & Apparel",
    "Shopping & Apparel":           "Other",
    "Entertainment & Subscriptions":"Shopping & Apparel",
    "Transport & Commute":          "Utilities & Bills",
    "EMI & Loan Repayment":         "Utilities & Bills",
    "Utilities & Bills":            "Other",
    "Healthcare":                   "Other",
    "Savings & Investment":         "Other",
    "Other":                        None,
}

# Module-level cache — loaded once on first use
_GROUND_TRUTH: Dict[str, str] = {}  # {transaction_id: correct_category}
_CATEGORIES: List[str] = []


def _ensure_loaded() -> None:
    global _GROUND_TRUTH, _CATEGORIES
    if not _GROUND_TRUTH:
        with open(DATA_FILE) as f:
            _GROUND_TRUTH = {t["id"]: t["correct_category"] for t in json.load(f)}
    if not _CATEGORIES:
        with open(CATEGORIES_FILE) as f:
            _CATEGORIES = json.load(f)


def _clamp(value: float) -> float:
    return min(1.0 - EPSILON, max(EPSILON, value))


def grade_categorize(
    payload: Dict[str, Any],
    addressed_ids: List[str],
) -> Tuple[float, Dict[str, float], str, bool]:
    """
    Score one "categorize" action.
    Returns (+0.040 exact, +0.015 near-miss, -0.010 invalid, -0.020 duplicate, 0.0 wrong).
    """
    _ensure_loaded()

    txn_id: str = payload.get("transaction_id", "")
    category: str = payload.get("category", "")

    if txn_id not in _GROUND_TRUTH:
        return 0.0, {}, f"Unknown transaction ID: '{txn_id}'.", False

    if txn_id in addressed_ids:
        return -0.020, {}, f"Duplicate: '{txn_id}' already categorized.", False

    if category not in _CATEGORIES:
        return -0.010, {}, f"Invalid category: '{category}'.", False

    correct = _GROUND_TRUTH[txn_id]
    correct_so_far = len(addressed_ids) + 1  # include this action

    if category == correct:
        partial = {"correct_labels": _clamp(correct_so_far / MAX_TRANSACTIONS), "efficiency": _clamp(0.0)}
        return 0.040, partial, "Correct.", False

    if category == PARENT_CATEGORY.get(correct):
        partial = {"correct_labels": _clamp(len(addressed_ids) / MAX_TRANSACTIONS), "efficiency": _clamp(0.0)}
        return 0.015, partial, f"Partial credit. '{category}' is related but '{correct}' was expected.", False

    partial = {"correct_labels": _clamp(len(addressed_ids) / MAX_TRANSACTIONS), "efficiency": _clamp(0.0)}
    return 0.0, partial, f"Incorrect. Expected '{correct}', got '{category}'.", False


def grade_finalize(
    addressed_ids: List[str],
    step_count: int,
    correct_count: int,
) -> Tuple[float, Dict[str, float], str, bool]:
    """
    Score the "finalize" action. Always ends the episode.
    +0.050 terminal bonus always. +0.050 efficiency bonus if all 20 addressed in < 22 steps.
    """
    _ensure_loaded()

    score_delta = 0.050  # terminal bonus always awarded

    efficiency = 0.0
    if len(addressed_ids) == MAX_TRANSACTIONS and step_count < EFFICIENCY_STEP_THRESHOLD:
        efficiency = 0.050
        score_delta += efficiency

    partial: Dict[str, float] = {
        "correct_labels": _clamp(correct_count / MAX_TRANSACTIONS),
        "efficiency": _clamp(efficiency),
    }

    feedback = (
        f"Episode complete. {correct_count}/{MAX_TRANSACTIONS} correct. "
        f"{len(addressed_ids)}/{MAX_TRANSACTIONS} addressed."
    )
    if efficiency:
        feedback += f" Efficiency bonus (finished in {step_count} steps)."

    return score_delta, partial, feedback, True
