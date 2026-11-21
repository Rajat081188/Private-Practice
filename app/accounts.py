"""Simulated account data and refund tools (agent PRD sections 6.3-6.7).

Everything here is deterministic application code. The model may call get_user_account and
validate_refund, but it never decides eligibility, and it cannot call process_refund or
submit_to_support: the Refund Agent's state machine calls those only after validation (and,
for process_refund, after explicit confirmation).

No real CRM, billing or payment system is touched. Refunds only update the in-memory data below.
"""
from __future__ import annotations

import copy
import os
import threading
import uuid

REFUND_WINDOW_DAYS = 7
APPROVAL_LIMIT = 499  # refunds at or below this can be executed here after confirmation (FR-36, FR-37)

# Hard-coded for the demonstration (FR-14..FR-18a). user_123 is the PRD's example user.
# user_456 exists so the "too old" and "already refunded" scenarios can be shown live.
_SEED_USERS = {
    "user_123": {
        "name": "Rahul",
        "plan": "Pro",
        "plan_amount": 999,
        "recharges": [
            {"recharge_id": "rch_456", "recharge_amount": 999, "recharge_status": "completed",
             "recharge_age_days": 2, "refund_status": "not_refunded"},
            {"recharge_id": "rch_457", "recharge_amount": 499, "recharge_status": "completed",
             "recharge_age_days": 1, "refund_status": "not_refunded"},
        ],
    },
    "user_456": {
        "name": "Priya",
        "plan": "Basic",
        "plan_amount": 299,
        "recharges": [
            {"recharge_id": "rch_900", "recharge_amount": 299, "recharge_status": "completed",
             "recharge_age_days": 15, "refund_status": "not_refunded"},
            {"recharge_id": "rch_901", "recharge_amount": 199, "recharge_status": "completed",
             "recharge_age_days": 3, "refund_status": "refunded"},
        ],
    },
}

_lock = threading.Lock()
USERS: dict = copy.deepcopy(_SEED_USERS)
SUPPORT_QUEUE: list[dict] = []  # requests handed to the Support Team; their review is out of scope (FR-40a)


def session_user_id() -> str:
    """Stands in for the authenticated session's user (FR-15, FR-16). The user is never asked for it."""
    return os.environ.get("NOVACELL_DEMO_USER", "user_123")


def reset_demo_data() -> None:
    global USERS
    with _lock:
        USERS = copy.deepcopy(_SEED_USERS)
        SUPPORT_QUEUE.clear()


def get_user_account(user_id: str) -> dict:
    """Tool (exposed to the model): the user's plan and recharges."""
    with _lock:
        user = USERS.get(user_id)
        if not user:
            return {"found": False, "user_id": user_id}
        return {"found": True, "user_id": user_id, **copy.deepcopy(user)}


def _inr(x: float) -> str:
    return f"₹{x:,.0f}" if float(x).is_integer() else f"₹{x:,.2f}"


def _check(r: dict, amount: float) -> str | None:
    """Return the reason a refund of `amount` against recharge `r` is not allowed, or None."""
    if r["recharge_status"] != "completed":
        return f"The recharge is {r['recharge_status']}, and only completed recharges can be refunded."
    if r["refund_status"] == "refunded":
        return "This recharge has already been refunded."
    if r["recharge_age_days"] > REFUND_WINDOW_DAYS:
        return (f"The recharge was {r['recharge_age_days']} days ago, outside the "
                f"{REFUND_WINDOW_DAYS}-day refund window.")
    if amount > r["recharge_amount"]:
        return (f"The requested amount ({_inr(amount)}) exceeds the recharge amount "
                f"({_inr(r['recharge_amount'])}).")
    return None


def _select_recharge(recharges: list[dict], amount: float) -> dict | None:
    """Pick the recharge a refund applies to, deterministically (FR-27a).

    1. A recharge whose amount matches exactly is that recharge, even if already refunded.
    2. Otherwise, the most recent recharge that could cover the amount and is refundable.
    3. Otherwise, the most recent recharge that could cover it (so the reason names a real problem).
    4. Otherwise (amount larger than every recharge), the largest recharge.
    """
    if not recharges:
        return None
    exact = sorted((r for r in recharges if r["recharge_amount"] == amount), key=lambda r: r["recharge_age_days"])
    if exact:
        refundable = [r for r in exact if _check(r, amount) is None]
        return (refundable or exact)[0]
    covering = sorted((r for r in recharges if r["recharge_amount"] >= amount),
                      key=lambda r: (r["recharge_age_days"], r["recharge_amount"]))
    for r in covering:
        if _check(r, amount) is None:
            return r
    if covering:
        return covering[0]
    return max(recharges, key=lambda r: r["recharge_amount"])


def validate_refund(user_id: str, amount: float) -> dict:
    """Tool (exposed to the model): deterministic eligibility check (FR-24..FR-30).

    Decides eligibility only. Whether it can be executed here is the approval limit's job.
    """
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return {"eligible": False, "reason": "The refund amount is not a number.", "requested_amount": amount}
    base = {"requested_amount": amount}
    if amount <= 0:
        return {**base, "eligible": False, "reason": "The refund amount must be greater than zero."}
    with _lock:
        user = USERS.get(user_id)
        if not user:
            return {**base, "eligible": False, "reason": "No account was found."}
        r = _select_recharge(user["recharges"], amount)
        if r is None:
            return {**base, "eligible": False, "reason": "There are no recharges on this account."}
        reason = _check(r, amount)
        result = {**base, "recharge_amount": r["recharge_amount"], "recharge_id": r["recharge_id"],
                  "eligible": reason is None, "reason": reason or "Refund is eligible."}
        return result


def eligible_recharges(user_id: str) -> list[dict]:
    with _lock:
        user = USERS.get(user_id) or {"recharges": []}
        return [copy.deepcopy(r) for r in sorted(user["recharges"], key=lambda r: r["recharge_age_days"])
                if _check(r, r["recharge_amount"]) is None]


# ---- Not exposed to the model (FR-23a). Called only by the Refund Agent's state machine. ----

def process_refund(user_id: str, amount: float, recharge_id: str) -> dict:
    """Simulated refund (FR-23, FR-33, FR-34). Re-validates so it can never run on an ineligible refund."""
    with _lock:
        user = USERS.get(user_id)
        r = next((x for x in (user or {}).get("recharges", []) if x["recharge_id"] == recharge_id), None)
        if r is None:
            return {"success": False, "reason": "Recharge not found."}
        reason = _check(r, float(amount))
        if reason:
            return {"success": False, "reason": reason}
        if float(amount) > APPROVAL_LIMIT:
            return {"success": False, "reason": "Above the approval limit; must go to the Support Team."}
        r["refund_status"] = "refunded"  # a refund consumes the whole recharge (FR-28a)
        return {"success": True, "refund_id": "ref_" + uuid.uuid4().hex[:6], "amount": float(amount),
                "recharge_id": recharge_id, "status": "refunded"}


def submit_to_support(user_id: str, amount: float, recharge_id: str) -> dict:
    """Hand an above-limit refund to the Support Team. Does not change refund_status (scenario 6)."""
    ticket = {"request_id": "SR-" + uuid.uuid4().hex[:6].upper(), "user_id": user_id, "amount": float(amount),
              "recharge_id": recharge_id, "status": "submitted_for_review"}
    with _lock:
        SUPPORT_QUEUE.append(ticket)
    return {"submitted": True, **ticket}
