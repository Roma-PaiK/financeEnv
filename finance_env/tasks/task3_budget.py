"""
Task 3 Grader — Forward Budget Planning with Life Event Shock

Scores "set_budget", "query", and "finalize" actions.
Called by env.py after action validation.

Returns: (score_delta, partial_scores, feedback, done)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

HISTORY_FILE = Path(__file__).parent.parent / "data" / "task3_history.json"
GOALS_FILE = Path(__file__).parent.parent / "data" / "task3_goals.json"
SIMULATION_FILE = Path(__file__).parent.parent / "data" / "task3_simulation.json"
CATEGORIES_FILE = Path(__file__).parent.parent / "data" / "categories.json"

MONTHLY_INCOME = 85000.0
ESSENTIAL_CATEGORIES = ["EMI & Loan Repayment", "Utilities & Bills", "Healthcare"]

_HISTORY: List[Dict[str, Any]] = []
_GOALS: Dict[str, Any] = {}
_SIMULATION: Dict[str, Any] = {}
_CATEGORIES: List[str] = []
_HISTORICAL_AVG: Dict[str, float] = {}


def _ensure_loaded() -> None:
    global _HISTORY, _GOALS, _SIMULATION, _CATEGORIES, _HISTORICAL_AVG
    if not _HISTORY:
        with open(HISTORY_FILE) as f:
            _HISTORY = json.load(f)
    if not _GOALS:
        with open(GOALS_FILE) as f:
            _GOALS = json.load(f)
    if not _SIMULATION:
        with open(SIMULATION_FILE) as f:
            _SIMULATION = json.load(f)
    if not _CATEGORIES:
        with open(CATEGORIES_FILE) as f:
            _CATEGORIES = json.load(f)
    if not _HISTORICAL_AVG:
        _HISTORICAL_AVG = _compute_historical_avg()


def _compute_historical_avg() -> Dict[str, float]:
    """Compute average monthly debit spend per category from history."""
    monthly_totals: Dict[str, Dict[str, float]] = {}
    for txn in _HISTORY:
        if txn["amount"] <= 0:
            continue  # skip credits (salary etc.)
        month = txn["date"][:7]
        cat = txn.get("correct_category", "Other")
        if month not in monthly_totals:
            monthly_totals[month] = {}
        monthly_totals[month][cat] = monthly_totals[month].get(cat, 0.0) + txn["amount"]

    if not monthly_totals:
        return {cat: 0.0 for cat in _CATEGORIES}

    n_months = len(monthly_totals)
    avgs: Dict[str, float] = {}
    for cat in _CATEGORIES:
        total = sum(m.get(cat, 0.0) for m in monthly_totals.values())
        avgs[cat] = round(total / n_months, 2)
    return avgs


def _clamp(v: float) -> float:
    eps = 1e-6
    return min(1.0 - eps, max(eps, v))


def _realism_score(budget_draft: Dict[str, float]) -> float:
    """Fraction of set categories whose allocation is within ±30% of historical avg."""
    if not budget_draft:
        return _clamp(0.0)
    realistic = sum(
        1
        for cat, amt in budget_draft.items()
        if _HISTORICAL_AVG.get(cat, 0.0) > 0
        and abs(amt - _HISTORICAL_AVG[cat]) / _HISTORICAL_AVG[cat] <= 0.30
    )
    return _clamp(realistic / len(budget_draft))


def grade_set_budget(
    payload: Dict[str, Any],
    budget_draft: Dict[str, float],
) -> Tuple[float, Dict[str, float], str, bool]:
    """Grade a 'set_budget' action. Mutates budget_draft in place."""
    _ensure_loaded()

    category: str = payload.get("category", "")
    amount = payload.get("amount", None)

    if category not in _CATEGORIES:
        partial = {"realism": _realism_score(budget_draft)}
        return (
            -0.010,
            partial,
            f"Invalid category: '{category}'. Must be one of the 9 canonical categories.",
            False,
        )

    if amount is None or not isinstance(amount, (int, float)) or float(amount) < 0:
        partial = {"realism": _realism_score(budget_draft)}
        return (
            -0.010,
            partial,
            f"Invalid amount for '{category}'. Must be a non-negative number.",
            False,
        )

    amount = float(amount)
    hist_avg = _HISTORICAL_AVG.get(category, 0.0)

    # Zero budget for a historically-spent category
    if amount == 0.0 and hist_avg > 0:
        budget_draft[category] = amount
        partial = {"realism": _realism_score(budget_draft)}
        return (
            -0.060,
            partial,
            (
                f"Unrealistic: '{category}' had ₹{hist_avg:.0f} average historical spend. "
                "Cannot be zero."
            ),
            False,
        )

    # Update draft (revisions allowed, no penalty)
    budget_draft[category] = amount

    # Within ±30% of historical average → realistic
    if hist_avg > 0 and abs(amount - hist_avg) / hist_avg <= 0.30:
        partial = {"realism": _realism_score(budget_draft)}
        return (
            0.020,
            partial,
            (
                f"Realistic allocation for '{category}' (₹{amount:.0f}). "
                "Behavioural baseline respected."
            ),
            False,
        )

    partial = {"realism": _realism_score(budget_draft)}
    return (
        0.0,
        partial,
        (
            f"Allocation noted for '{category}': ₹{amount:.0f}. "
            f"(Historical avg: ₹{hist_avg:.0f})"
        ),
        False,
    )


def grade_query(
    payload: Dict[str, Any],
    budget_draft: Dict[str, float],
) -> Tuple[float, Dict[str, float], str, bool]:
    """Grade a 'query' action and return historical category spend data."""
    _ensure_loaded()

    category: str = payload.get("category", "")
    months: List[str] = payload.get("months", [])

    if category not in _CATEGORIES:
        return (
            -0.010,
            {"realism": _realism_score(budget_draft)},
            f"Invalid category: '{category}'. Must be one of the 9 canonical categories.",
            False,
        )

    # Compute per-month spending for the requested category
    monthly_data: Dict[str, float] = {}
    for txn in _HISTORY:
        if txn["amount"] <= 0:
            continue
        month = txn["date"][:7]
        if months and month not in months:
            continue
        if txn.get("correct_category") == category:
            monthly_data[month] = monthly_data.get(month, 0.0) + txn["amount"]

    lines = [f"  {m}: ₹{v:.0f}" for m, v in sorted(monthly_data.items())]
    avg = _HISTORICAL_AVG.get(category, 0.0)
    summary = "\n".join(lines) if lines else "  No data for requested months."

    feedback = (
        f"Historical spend for '{category}':\n{summary}\n"
        f"  2-month average: ₹{avg:.0f}/month"
    )

    # Score: useful if category not yet drafted, redundant if already set
    if category in budget_draft:
        score_delta = -0.010
        feedback += f"\n  Note: '{category}' already set to ₹{budget_draft[category]:.0f} in draft."
    else:
        score_delta = 0.015

    partial = {"realism": _realism_score(budget_draft)}
    return score_delta, partial, feedback, False


def grade_finalize(
    payload: Dict[str, Any],
    budget_draft: Dict[str, float],
    query_count: int,
) -> Tuple[float, Dict[str, float], str, bool]:
    """Grade the 'finalize' action. Runs simulation. Always ends the episode."""
    _ensure_loaded()

    # The submitted budget — fall back to current draft if payload budget is empty
    submitted: Dict[str, float] = {
        k: float(v) for k, v in payload.get("budget", {}).items()
    }
    budget = submitted if submitted else dict(budget_draft)

    score_delta = 0.0
    feedback_parts: List[str] = []
    partial: Dict[str, float] = {}

    # ── 0. No prior queries penalty ──────────────────────────────────────────
    if query_count == 0:
        score_delta -= 0.050
        feedback_parts.append("No exploration detected. Budget submitted blindly. (-0.050)")

    # ── 1. Missing categories penalty ────────────────────────────────────────
    missing = [cat for cat in _CATEGORIES if cat not in budget]
    for cat in missing:
        score_delta -= 0.050
        feedback_parts.append(f"Missing category: '{cat}'. (-0.050)")

    # ── 2. Budget sum validation ──────────────────────────────────────────────
    budget_sum = sum(budget.values())
    if budget_sum > MONTHLY_INCOME:
        score_delta -= 0.200
        partial = {
            "budget_validity": _clamp(0.0),
            "adherence": _clamp(0.0),
            "savings_goal": _clamp(0.0),
            "life_event_absorption": _clamp(0.0),
            "realism": _realism_score(budget_draft),
        }
        feedback_parts.append(
            f"INVALID: Budget (₹{budget_sum:.0f}) exceeds income (₹{MONTHLY_INCOME:.0f}) "
            f"by ₹{budget_sum - MONTHLY_INCOME:.0f}. Plan rejected. (-0.200)"
        )
        return score_delta, partial, "\n".join(feedback_parts), True

    # ── 3. Budget validity bonus ──────────────────────────────────────────────
    score_delta += 0.100
    partial["budget_validity"] = _clamp(0.100)
    feedback_parts.append(
        f"Budget valid: ₹{budget_sum:.0f} / ₹{MONTHLY_INCOME:.0f} income. (+0.100)"
    )

    # ── 4. Simulation ─────────────────────────────────────────────────────────
    actual_spend: Dict[str, float] = _SIMULATION["actual_spend"]

    # Adherence check
    adhered_categories = [
        cat for cat, actual in actual_spend.items()
        if budget.get(cat, 0.0) >= actual
    ]
    adhered_count = len(adhered_categories)
    total_cats = len(actual_spend)

    if adhered_count == total_cats:
        adherence_score = 0.150
    elif adhered_count >= 7:
        adherence_score = 0.080
    elif adhered_count >= 5:
        adherence_score = 0.040
    else:
        adherence_score = 0.0

    score_delta += adherence_score
    partial["adherence"] = _clamp(adherence_score)

    overspent_lines = [
        f"    {cat}: budgeted ₹{budget.get(cat, 0):.0f}, actual ₹{actual:.0f}"
        for cat, actual in actual_spend.items()
        if budget.get(cat, 0.0) < actual
    ]
    adherence_detail = (
        f"Adherence: {adhered_count}/{total_cats} categories within budget. "
        f"(+{adherence_score:.3f})"
    )
    if overspent_lines:
        adherence_detail += "\n  Overspent:\n" + "\n".join(overspent_lines)
    feedback_parts.append(adherence_detail)

    # Savings goal check
    goals_savings = float(_GOALS.get("savings_goal", 8000.0))
    sim_savings = MONTHLY_INCOME - sum(actual_spend.values())

    if sim_savings >= goals_savings:
        savings_score = 0.200
        feedback_parts.append(
            f"Savings goal met: ₹{sim_savings:.0f} remaining (goal: ₹{goals_savings:.0f}). (+0.200)"
        )
    elif sim_savings >= goals_savings * 0.5:
        savings_score = 0.100
        feedback_parts.append(
            f"Partial savings: ₹{sim_savings:.0f} remaining (goal: ₹{goals_savings:.0f}). (+0.100)"
        )
    else:
        savings_score = 0.0
        feedback_parts.append(
            f"Savings goal missed: ₹{sim_savings:.0f} remaining (goal: ₹{goals_savings:.0f}). (+0.000)"
        )
    score_delta += savings_score
    partial["savings_goal"] = _clamp(savings_score)

    # Life event absorption check
    life_event_amount = float(_SIMULATION["life_event"]["amount"])
    excess_savings = max(0.0, budget.get("Savings & Investment", 0.0) - goals_savings)
    buffer = budget.get("Other", 0.0) + excess_savings

    if buffer >= life_event_amount:
        life_score = 0.100
        feedback_parts.append(
            f"Life event absorbed (₹{life_event_amount:.0f} car repair). "
            f"Available buffer: ₹{buffer:.0f}. (+0.100)"
        )
    elif buffer >= life_event_amount * 0.5:
        life_score = 0.050
        feedback_parts.append(
            f"Life event partially absorbed. Buffer ₹{buffer:.0f} < shock ₹{life_event_amount:.0f}. (+0.050)"
        )
    else:
        life_score = 0.0
        feedback_parts.append(
            f"Life event unabsorbed. ₹{life_event_amount:.0f} car repair exceeded buffer "
            f"(₹{buffer:.0f}). (+0.000)"
        )
    score_delta += life_score
    partial["life_event_absorption"] = _clamp(life_score)

    # Essential category underfunding check
    for ess_cat in ESSENTIAL_CATEGORIES:
        hist_avg = _HISTORICAL_AVG.get(ess_cat, 0.0)
        if hist_avg > 0 and budget.get(ess_cat, 0.0) < hist_avg * 0.5:
            score_delta -= 0.080
            feedback_parts.append(
                f"Risk: '{ess_cat}' critically underfunded "
                f"(budgeted ₹{budget.get(ess_cat, 0):.0f}, minimum ₹{hist_avg * 0.5:.0f}). (-0.080)"
            )

    partial["realism"] = _realism_score(budget_draft)

    return score_delta, partial, "\n".join(feedback_parts), True
