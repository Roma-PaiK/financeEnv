"""
Task 2 grader — Cross-Source Deduplication & Reconciliation.
Scores "reconcile", "query", and "finalize" actions against ground truth.
Returns: (score_delta, partial_scores, feedback, done)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

MULTI_SOURCE_FILE = Path(__file__).parent.parent / "data" / "task2_multi_source.json"
GROUND_TRUTH_FILE = Path(__file__).parent.parent / "data" / "task2_ground_truth.json"

VALID_CLASSIFICATIONS: Set[str] = {"genuine_spend", "cc_settlement", "internal_transfer", "refund"}

# Module-level cache — loaded once on first use
_RAW_TRANSACTIONS: List[Dict[str, Any]] = []
_GROUND_TRUTH: Dict[str, Any] = {}
_TXN_INDEX: Dict[str, Dict[str, Any]] = {}  # id → transaction dict for O(1) lookup


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
    eps = 1e-6
    return min(1.0 - eps, max(eps, v))


def grade_reconcile(
    payload: Dict[str, Any],
    addressed_ids: List[str],
    correct_count: int,
) -> Tuple[float, Dict[str, float], str, bool]:
    """
    Score one "reconcile" action.
    Returns (+0.060 correct, -0.040 wrong, -0.020 duplicate, -0.010 invalid classification).
    """
    _ensure_loaded()

    txn_id: str = payload.get("transaction_id", "")
    classification: str = payload.get("classification", "")

    if txn_id not in _TXN_INDEX:
        return 0.0, {}, f"Unknown transaction ID: '{txn_id}'.", False

    if txn_id in addressed_ids:
        partial = {"classification_accuracy": _clamp(correct_count / max(len(addressed_ids), 1))}
        return -0.020, partial, f"Duplicate: '{txn_id}' already reconciled.", False

    if classification not in VALID_CLASSIFICATIONS:
        return -0.010, {}, f"Invalid classification: '{classification}'. Must be one of: {sorted(VALID_CLASSIFICATIONS)}.", False

    correct_cls = _GROUND_TRUTH["classifications"].get(txn_id, "genuine_spend")
    total = len(addressed_ids) + 1
    txn = _TXN_INDEX[txn_id]

    if classification == correct_cls:
        partial = {"classification_accuracy": _clamp((correct_count + 1) / total)}
        return 0.060, partial, f"Correct. '{txn_id}' ({txn['description']}) → '{classification}'.", False

    partial = {"classification_accuracy": _clamp(correct_count / total)}
    return -0.040, partial, f"Wrong. '{txn_id}' ({txn['description']}): expected '{correct_cls}', got '{classification}'.", False


def grade_query(
    payload: Dict[str, Any],
    addressed_ids: List[str],
    correct_count: int,
) -> Tuple[float, Dict[str, float], str, bool]:
    """
    Score a "query" action. Returns matching transactions as feedback.
    +0.010 if unresolved duplicates found in results, -0.010 if redundant, -0.005 if no matches.
    """
    _ensure_loaded()

    query_type: str = payload.get("query_type", "")
    value: str = payload.get("value", "")

    if query_type not in ("merchant", "date_range"):
        return -0.010, {}, f"Invalid query_type: '{query_type}'. Use 'merchant' or 'date_range'.", False

    if query_type == "merchant":
        matches = [t for t in _RAW_TRANSACTIONS if value.lower() in t["description"].lower()]
    else:
        try:
            start_str, end_str = value.split(":")
        except ValueError:
            return -0.010, {}, "Invalid date_range format. Use 'YYYY-MM-DD:YYYY-MM-DD'.", False
        matches = [t for t in _RAW_TRANSACTIONS if start_str <= t["date"] <= end_str]

    match_lines = [
        f"  [{t['id']}] {t['date']} | {t['source']} | {t['description']} | ₹{t['amount']}"
        for t in matches[:8]
    ]
    suffix = f" (showing first 8 of {len(matches)})" if len(matches) > 8 else ""

    total_actioned = max(len(addressed_ids), 1)
    partial = {"classification_accuracy": _clamp(correct_count / total_actioned)}

    if not matches:
        return -0.005, partial, f"Query '{value}' matched no transactions.", False

    duplicate_ids: Set[str] = set(_GROUND_TRUTH["duplicate_ids"])
    unresolved = [t for t in matches if t["id"] in duplicate_ids and t["id"] not in addressed_ids]

    if unresolved:
        note = f" ⚠ {len(unresolved)} unresolved duplicate(s) in results."
        score_delta = 0.010
    else:
        note = " All duplicates in this range are resolved."
        score_delta = -0.010

    feedback = f"Query '{value}' → {len(matches)} transaction(s){suffix}:{note}\n" + "\n".join(match_lines)
    return score_delta, partial, feedback, False


def grade_finalize(
    payload: Dict[str, Any],
    addressed_ids: List[str],
    correct_count: int,
) -> Tuple[float, Dict[str, float], str, bool]:
    """
    Score the "finalize" action. Always ends the episode.
    Scores: F1 on excluded_ids (max +0.200) + reconciled_totals accuracy.
    """
    _ensure_loaded()

    excluded_ids: List[str] = payload.get("excluded_ids", [])
    reconciled_totals: Dict[str, Any] = payload.get("reconciled_totals", {})

    score_delta = 0.0
    feedback_parts: List[str] = []

    # F1 score on excluded_ids vs ground truth duplicate IDs
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
    feedback_parts.append(f"Exclusion F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn}) → +{exclusion_score:.4f}.")

    # Reconciled totals accuracy — +0.010 within 2%, +0.005 within 10% per category
    gt_totals = _GROUND_TRUTH["reconciled_totals"]
    totals_score = 0.0
    category_hits = 0
    category_total = 0

    for month, gt_cats in gt_totals.items():
        submitted_month: Dict[str, float] = reconciled_totals.get(month, {})
        for category, gt_amount in gt_cats.items():
            if gt_amount == 0.0:
                continue
            category_total += 1
            submitted = float(submitted_month.get(category, 0.0))
            err_pct = abs(submitted - gt_amount) / gt_amount
            if err_pct <= 0.02:
                totals_score += 0.010
                category_hits += 1
            elif err_pct <= 0.10:
                totals_score += 0.005

    totals_score = _clamp(totals_score)
    score_delta += totals_score
    feedback_parts.append(f"Totals: {category_hits}/{category_total} near-perfect → +{totals_score:.4f}.")

    total_actioned = max(len(addressed_ids), 1)
    cls_acc = _clamp(correct_count / total_actioned)
    feedback_parts.append(f"Classification: {correct_count}/{len(addressed_ids)} correct.")

    partial: Dict[str, float] = {
        "classification_accuracy": cls_acc,
        "exclusion_f1":    _clamp(f1),
        "totals_accuracy": totals_score,
    }
    return score_delta, partial, " ".join(feedback_parts), True
