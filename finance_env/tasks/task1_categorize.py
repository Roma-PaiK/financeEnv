"""
Task 1 Grader — Transaction Categorisation

Scores "categorize" and "finalize" actions against the ground truth in the raw
transaction data. Called by env.py after action validation.

Returns: (score_delta, partial_scores, feedback, done)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATA_FILE = Path(__file__).parent.parent / "data" / "task1_transactions.json"
CATEGORIES_FILE = Path(__file__).parent.parent / "data" / "categories.json"

MAX_TRANSACTIONS = 20
EFFICIENCY_STEP_THRESHOLD = 22  # must finalize before this step count to earn bonus

EPSILON = 1e-6  # Clamp scores to exclusive interval (EPSILON, 1 - EPSILON)


def _clamp_score(value: float) -> float:
    """Clamp score to exclusive open interval (EPSILON, 1 - EPSILON)."""
    return min(1.0 - EPSILON, max(EPSILON, value))

# Hierarchical parent for near-miss scoring.
# If the agent submits parent_category[correct] instead of correct → +0.015 partial.
PARENT_CATEGORY: Dict[str, str | None] = {
    "Food & Dining": "Shopping & Apparel",
    "Shopping & Apparel": "Other",
    "Entertainment & Subscriptions": "Shopping & Apparel",
    "Transport & Commute": "Utilities & Bills",
    "EMI & Loan Repayment": "Utilities & Bills",
    "Utilities & Bills": "Other",
    "Healthcare": "Other",
    "Savings & Investment": "Other",
    "Other": None,
}


def _load_ground_truth() -> Dict[str, str]:
    """Returns {transaction_id: correct_category} from raw data file."""
    with open(DATA_FILE) as f:
        raw = json.load(f)
    return {txn["id"]: txn["correct_category"] for txn in raw}


def _load_categories() -> List[str]:
    with open(CATEGORIES_FILE) as f:
        return json.load(f)


# Module-level cache so we don't re-read on every step
_GROUND_TRUTH: Dict[str, str] = {}
_CATEGORIES: List[str] = []


def _ensure_loaded() -> None:
    global _GROUND_TRUTH, _CATEGORIES
    if not _GROUND_TRUTH:
        _GROUND_TRUTH = _load_ground_truth()
    if not _CATEGORIES:
        _CATEGORIES = _load_categories()


def grade_categorize(
    payload: Dict[str, Any],
    addressed_ids: List[str],
) -> Tuple[float, Dict[str, float], str, bool]:
    """Grade a single 'categorize' action."""
    _ensure_loaded()

    txn_id: str = payload.get("transaction_id", "")
    category: str = payload.get("category", "")

    # Unknown transaction ID
    if txn_id not in _GROUND_TRUTH:
        return 0.0, {}, f"Unknown transaction ID: '{txn_id}'.", False

    # Already addressed
    if txn_id in addressed_ids:
        return (
            -0.020,
            {},
            f"Duplicate action: '{txn_id}' has already been categorized.",
            False,
        )

    # Invalid category string
    if category not in _CATEGORIES:
        return (
            -0.010,
            {},
            f"Invalid category: '{category}'. Must be one of the 9 canonical categories.",
            False,
        )

    correct = _GROUND_TRUTH[txn_id]

    # Exact match
    if category == correct:
        correct_count = len(addressed_ids) + 1  # +1 for this action
        partial = {
            "correct_labels": _clamp_score(correct_count / MAX_TRANSACTIONS),
            "efficiency": _clamp_score(0.0),
        }
        return 0.040, partial, "Correct.", False

    # Near-miss: agent submitted the parent of the correct category
    if category == PARENT_CATEGORY.get(correct):
        correct_count = len(addressed_ids)
        partial = {
            "correct_labels": _clamp_score(correct_count / MAX_TRANSACTIONS),
            "efficiency": _clamp_score(0.0),
        }
        return (
            0.015,
            partial,
            f"Partial credit. '{category}' is related but '{correct}' was expected.",
            False,
        )

    # Wrong
    correct_count = len(addressed_ids)
    partial = {
        "correct_labels": _clamp_score(correct_count / MAX_TRANSACTIONS),
        "efficiency": _clamp_score(0.0),
    }
    return (
        0.0,
        partial,
        f"Incorrect. Expected '{correct}', received '{category}'.",
        False,
    )


def grade_finalize(
    addressed_ids: List[str],
    step_count: int,
    correct_count: int,
) -> Tuple[float, Dict[str, float], str, bool]:
    """Grade a 'finalize' action. Always ends the epiode."""
    _ensure_loaded()

    score_delta = 0.050  # terminal bonus always awarded

    efficiency = 0.0
    if len(addressed_ids) == MAX_TRANSACTIONS and step_count < EFFICIENCY_STEP_THRESHOLD:
        efficiency = 0.050
        score_delta += efficiency

    partial: Dict[str, float] = {
        "correct_labels": _clamp_score(correct_count / MAX_TRANSACTIONS),
        "efficiency": _clamp_score(efficiency),
    }

    addressed = len(addressed_ids)
    feedback_parts = [
        f"Episode complete. {correct_count}/{MAX_TRANSACTIONS} correct categorizations.",
        f"{addressed}/{MAX_TRANSACTIONS} transactions addressed.",
    ]
    if efficiency:
        feedback_parts.append(f"Efficiency bonus awarded (finished in {step_count} steps).")

    return score_delta, partial, " ".join(feedback_parts), True
