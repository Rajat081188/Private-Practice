"""Deterministic routing rules that run before the model (PRD section 4 escalation triggers).

Rules, not the model, decide the safety-critical routes so they are predictable and testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts), re.I)


# Emergency numbers only count with calling context: "₹999" or "a refund of 112" is money, not an emergency.
EMERGENCY = _rx(r"\b(call(ing)?|dial(ling|ing)?|ring|phone|reach|contact)\b[^.?!\d]{0,20}\b(911|112|999)\b",
                r"(?<![₹$£€\d.,])\b(911|112|999)\b(?!\s*(rs|inr|rupees?)\b)[^.?!]{0,25}\b(emergenc|ambulance|police|fire)",
                r"\bemergency (call|services|number)\b", r"\bambulance\b", r"\bsuicid")
LOST_STOLEN = _rx(r"\b(lost|stolen|stole|missing|theft)\b.{0,40}\b(phone|sim|mobile|device|handset)\b",
                  r"\b(phone|sim|mobile|device|handset)\b.{0,30}\b(lost|stolen|was taken|got taken)\b")
FRAUD = _rx(r"\bfraud", r"\bscam", r"\bhack(ed)?\b", r"\bsim[- ]?swap", r"\bunauthori[sz]ed\b",
            r"someone (else )?(has |had )?(changed|accessed|logged into|is using) my",
            r"\bcompromised\b", r"\bidentity theft\b")
HUMAN = _rx(r"\b(talk|speak|chat) (to|with) (a |an |someone|somebody|person|human|agent|real)",
            r"\b(human|real person|live agent|representative|operator)\b", r"\bcustomer service agent\b",
            r"^\s*(agent|human)\s*[.!?]*\s*$")
# Refunds are no longer here: refund requests go to the Refund Agent (agent PRD), and questions about
# refunds go through normal retrieval.
ACCOUNT_ACTION = _rx(r"\bdispute\b", r"\bcompensat", r"\bgoodwill\b", r"\bcredit (me|my account|back)\b",
                     r"\bcancel (my|the) (contract|account|plan|line)\b",
                     r"\bunbar\b", r"\bwaive\b", r"\bchargeback\b")
# Questions only answerable with the customer's own account data.
ACCOUNT_DATA = _rx(
    r"\b(what('?s| is| are)|how much|how many|check|show|tell me|see)\b.{0,25}\bmy\b.{0,25}"
    r"\b(balance|credit|data left|remaining|usage|bill amount|invoice|plan details|account number|due amount|contract end)",
    r"\bhow much (data|credit|money|minutes) (do i have|have i got|is left|left)\b",
    r"\bwhy (is|was) my (bill|invoice)\b.{0,20}\$\s?\d",
    r"\b(my|the) (last|latest|current) (bill|invoice)\b.{0,30}\b(say|show|amount|total|cost)",
    r"\bam i (on|eligible|due|in contract)\b",
)
FRUSTRATION = _rx(r"\buseless\b", r"\bridiculous\b", r"\bterrible\b", r"\bawful\b", r"\bfurious\b", r"\bangry\b",
                  r"\bnot helpful\b", r"\bwaste of (my )?time\b", r"\bwtf\b", r"\bfed up\b", r"\bunacceptable\b",
                  r"(still|it) (doesn'?t|does not|isn'?t) work", r"\bthis is a joke\b")
INJECTION = _rx(r"ignore (all |any |the |your )?(previous|prior|above) (instructions|rules)",
                r"\bsystem prompt\b", r"\byou are now\b", r"\bpretend (to be|you are)\b", r"\bjailbreak\b",
                r"\bdeveloper mode\b")


@dataclass
class Route:
    kind: str  # answer | account_data | escalate | emergency | decline
    reasons: list[str] = field(default_factory=list)
    offer_handoff: bool = False
    urgent_banner: str | None = None


def _norm(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def is_repeat(message: str, previous_user_messages: list[str]) -> bool:
    cur = _norm(message)
    if len(cur) < 3:
        return False
    for prev in previous_user_messages[-3:]:
        p = _norm(prev)
        if p and len(cur & p) / len(cur | p) >= 0.7:
            return True
    return False


def route(message: str, previous_user_messages: list[str] | None = None) -> Route:
    prev = previous_user_messages or []
    if EMERGENCY.search(message):
        return Route("emergency", ["emergency"], urgent_banner=(
            "If this is an emergency, call your local emergency number now (112 or 911). "
            "Emergency calls work even without credit or roaming."))
    if FRAUD.search(message):
        return Route("escalate", ["fraud_or_security"], offer_handoff=True, urgent_banner=(
            "For your security, an agent needs to verify your identity. "
            "If you think someone has taken over your number, call 611 now."))
    if INJECTION.search(message):
        return Route("decline", ["possible_prompt_injection"])
    if HUMAN.search(message):
        return Route("escalate", ["asked_for_human"], offer_handoff=True)
    if ACCOUNT_ACTION.search(message):
        return Route("escalate", ["needs_account_action"], offer_handoff=True)

    r = Route("answer")
    if LOST_STOLEN.search(message):
        # Answerable from the FAQ, but blocking the SIM is urgent and needs a person.
        r.reasons.append("lost_or_stolen")
        r.offer_handoff = True
        r.urgent_banner = "Call 611 right away so we can block your SIM and protect your number."
    if ACCOUNT_DATA.search(message):
        r.kind = "account_data"
        r.reasons.append("personal_account_data")
        r.offer_handoff = True
    if FRUSTRATION.search(message):
        r.reasons.append("frustration")
        r.offer_handoff = True
    if is_repeat(message, prev):
        r.reasons.append("repeated_question")
        r.offer_handoff = True
    return r
