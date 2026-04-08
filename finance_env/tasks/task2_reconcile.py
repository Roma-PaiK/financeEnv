"""
Task 2 Grader — Cross-Source Deduplication & Reconciliation

Scores "reconcile", "query", and "finalize" actions against the ground truth.
Called by env.py after action validation.

Returns: (score_delta, partial_scores, feedback, done)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

MULTI_SOURCE_FILE = Path(__file__).parent.parent / "data" / "task2_multi_source.json"
GROUND_TRUTH_FILE = Path(__file__).parent.parent / "data" / "task2_ground_truth.json"

VALID_CLASSIFICATIONS: Set[str] = {"genuine_spend", "cc_settlement", "internal_transfer", "refund"}

# Module-level cache
_RAW_TRANSACTIONS: List[Dict[str, Any]] = []
_GROUND_TRUTH: Dict[str, Any] = {}
_TXN_INDEX: Dict[str, Dict[str, Any]] = {}  # id → transaction dict


def _ensure_loaded() -> None:
    global _RAW_TRANSACTIONS, _GROUND_TRUTH, _TXN_INDEX
    if not _RAW_TRANSACTIONS:
        with open(MULTI_SOURCE_FILE) as f:
            _RAW_TRANSACTIONS = json.load(f)
        _TXN_INDEX = {t["id"]: t for t in _RAW_TRANSACTIONS}
    if not _GROUND_TRUTH:
        with open(GROUND_TRUTH_FILE) as f:
            _GROUND_TRUTH = json.load(f)


def _clamp(v: float) -> float:
    """Clamp to the exclusive open interval (0, 1)."""
    eps = 1e-6
    return min(1.0 - eps, max(eps, v))


def _classification_accuracy(addressed_ids: List[str], extra_correct: int = 0) -> float:
    """Compute running classification accuracy from addressed_ids + extra_correct for current step."""
    _ensure_loaded()
    total = len(addressed_ids) + (1 if extra_correct else 0)
    if total == 0:
        return _clamp(0.0)
    correct = sum(
        1 for aid in addressed_ids
        if _GROUND_TRUTH["classifications"].get(aid) is not None
        # We can't know what was submitted for each addressed ID, so we track
        # via correct_count passed from env.py instead.
        # This function is only called with the env-tracked correct_count.
    )
    return _clamp(correct / total)


def grade_reconcile(
    payload: Dict[str, Any],
    addressed_ids: List[str],
    correct_count: int,
) -> Tuple[float, Dict[str, float], str, bool]:
    """Grade a single 'reconcile' action."""
    _ensure_loaded()

    txn_id: str = payload.get("transaction_id", "")
    classification: str = payload.get("classification", "")

    # Unknown transaction
    if txn_id not in _TXN_INDEX:
        return 0.0, {}, f"Unknown transaction ID: '{txn_id}'.", False

    # Already addressed (duplicate action)
    if txn_id in addressed_ids:
        partial = {"classification_accuracy": _clamp(correct_count / max(len(addressed_ids), 1))}
        return -0.020, partial, f"Duplicate action: '{txn_id}' already reconciled.", False

    # Invalid classification string
    if classification not in VALID_CLASSIFICATIONS:
        return (
            -0.010,
            {},
            f"Invalid classification: '{classification}'. "
            f"Must be one of: {sorted(VALID_CLASSIFICATIONS)}.",
            False,
        )

    # Compare against ground truth
    correct_cls = _GROUND_TRUTH["classifications"].get(txn_id, "genuine_spend")
    total_actioned = len(addressed_ids) + 1  # +1 for this action

    if classification == correct_cls:
        new_correct = correct_count + 1
        partial = {"classification_accuracy": _clamp(new_correct / total_actioned)}
        txn = _TXN_INDEX[txn_id]
        return (
            0.060,
            partial,
            f"Correct. '{txn_id}' ({txn['description']}) classified as '{classification}'.",
            False,
        )
    else:
        partial = {"classification_accuracy": _clamp(correct_count / total_actioned)}
        txn = _TXN_INDEX[txn_id]
        return (
            -0.040,
            partial,
            f"Misclassified '{txn_id}' ({txn['description']}). "
            f"Expected '{correct_cls}', got '{classification}'.",
            False,
        )


def grade_query(
    payload: Dict[str, Any],
    addressed_ids: List[str],
    correct_count: int,
) -> Tuple[float, Dict[str, float], str, bool]:
    """Grade a 'query' action and return context feedback."""
    _ensure_loaded()

    query_type: str = payload.get("query_type", "")
    value: str = payload.get("value", "")

    if query_type not in ("merchant", "date_range"):
        return (
            -0.010,
            {},
            f"Invalid query_type: '{query_type}'. Use 'merchant' or 'date_range'.",
            False,
        )

    # Find matching transactions
    matches: List[Dict[str, Any]] = []
    if query_type == "merchant":
        matches = [
            t for t in _RAW_TRANSACTIONS
            if value.lower() in t["description"].lower()
        ]
    else:  # date_range
        try:
            start_str, end_str = value.split(":")
        except ValueError:
            return (
                -0.010,
                {},
                "Invalid date_range format. Use 'YYYY-MM-DD:YYYY-MM-DD'.",
                False,
            )
        matches = [
            t for t in _RAW_TRANSACTIONS
            if start_str <= t["date"] <= end_str
        ]

    # Build feedback summary (visible fields only — no hidden fields)
    visible_fields = ("id", "source", "description", "amount", "date")
    match_lines = [
        f"  [{t['id']}] {t['date']} | {t['source']} | {t['description']} | ₹{t['amount']}"
        for t in matches[:8]
    ]
    suffix = f" (showing first 8 of {len(matches)})" if len(matches) > 8 else ""

    duplicate_ids: Set[str] = set(_GROUND_TRUTH["duplicate_ids"])
    unresolved_dups = [
        t for t in matches
        if t["id"] in duplicate_ids and t["id"] not in addressed_ids
    ]

    total_actioned = max(len(addressed_ids), 1)
    partial = {"classification_accuracy": _clamp(correct_count / total_actioned)}

    if not matches:
        return (
            -0.005,
            partial,
            f"Query '{value}' matched no transactions.",
            False,
        )

    if unresolved_dups:
        dup_note = f" ⚠ {len(unresolved_dups)} unresolved duplicate(s) in results."
        score_delta = 0.010
    else:
        dup_note = " All duplicates in this range are already resolved."
        score_delta = -0.010

    feedback = (
        f"Query '{value}' returned {len(matches)} transaction(s){suffix}:{dup_note}\n"
        + "\n".join(match_lines)
    )
    return score_delta, partial, feedback, False


def grade_finalize(
    payload: Dict[str, Any],
    addressed_ids: List[str],
    correct_count: int,
) -> Tuple[float, Dict[str, float], str, bool]:
    """Grade the 'finalize' action. Always ends the episode."""
    _ensure_loaded()

    excluded_ids: List[str] = payload.get("excluded_ids", [])
    reconciled_totals: Dict[str, Any] = payload.get("reconciled_totals", {})

    score_delta = 0.0
    feedback_parts: List[str] = []

    # ── 1. F1 score on excluded_ids ──────────────────────────────────────────
    ground_truth_dups: Set[str] = set(_GROUND_TRUTH["duplicate_ids"])
    excluded_set: Set[str] = set(excluded_ids)

    tp = len(excluded_set & ground_truth_dups)
    fp = len(excluded_set - ground_truth_dups)
    fn = len(ground_truth_dups - excluded_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    exclusion_score = _clamp(f1 * 0.200)
    score_delta += exclusion_score
    feedback_parts.append(
        f"Exclusion F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn}) → +{exclusion_score:.4f}."
    )

    # ── 2. Reconciled totals accuracy ────────────────────────────────────────
    gt_totals = _GROUND_TRUTH["reconciled_totals"]
    totals_score = 0.0
    category_hits = 0
    category_total = 0

    for month, gt_cats in gt_totals.items():
        submitted_month: Dict[str, float] = reconciled_totals.get(month, {})
        for category, gt_amount in gt_cats.items():
            if gt_amount == 0.0:
                continue  # skip zero ground-truth categories in scoring
            category_total += 1
            submitted = float(submitted_month.get(category, 0.0))
            abs_error_pct = abs(submitted - gt_amount) / gt_amount
            if abs_error_pct <= 0.02:
                totals_score += 0.010
                category_hits += 1
            elif abs_error_pct <= 0.10:
                totals_score += 0.005

    totals_score = _clamp(totals_score)
    score_delta += totals_score
    feedback_parts.append(
        f"Totals accuracy: {category_hits}/{category_total} near-perfect → +{totals_score:.4f}."
    )

    # ── 3. Classification accuracy summary ───────────────────────────────────
    total_actioned = max(len(addressed_ids), 1)
    cls_acc = _clamp(correct_count / total_actioned)
    feedback_parts.append(
        f"Classification accuracy: {correct_count}/{len(addressed_ids)} correct reconcile actions."
    )

    partial: Dict[str, float] = {
        "classification_accuracy": cls_acc,
        "exclusion_f1":    _clamp(f1),
        "totals_accuracy": totals_score,
    }

    return score_delta, partial, " ".join(feedback_parts), True
