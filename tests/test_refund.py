"""Agentic refund capability (agent_PRD.md): intent routing, Refund Agent, guardrails, the 7 demo scenarios."""
import json
from types import SimpleNamespace

import pytest

from app import accounts, assistant, intent, llm, refund_agent


@pytest.fixture(autouse=True)
def fresh_accounts(monkeypatch):
    accounts.reset_demo_data()
    monkeypatch.delenv("NOVACELL_DEMO_USER", raising=False)
    yield
    accounts.reset_demo_data()


def run(msg, sid=None, **kw):
    events = list(assistant.respond(msg, sid, **kw))
    return events[0], "".join(e["text"] for e in events if e["type"] == "delta"), events[-1]


def recharge(rid):
    return next(r for r in accounts.USERS["user_123"]["recharges"] if r["recharge_id"] == rid)


@pytest.fixture
def spy(monkeypatch):
    calls = {"process": 0, "support": 0}
    real_p, real_s = accounts.process_refund, accounts.submit_to_support

    def p(*a, **k):
        calls["process"] += 1
        return real_p(*a, **k)

    def s(*a, **k):
        calls["support"] += 1
        return real_s(*a, **k)
    monkeypatch.setattr(accounts, "process_refund", p)
    monkeypatch.setattr(accounts, "submit_to_support", s)
    return calls


# ---------- Intent detection ----------

@pytest.mark.parametrize("msg,expected", [
    ("Why is my internet slow?", "requirement_inquiry"),
    ("I want a refund of ₹999.", "refund_request"),
    ("Give me a refund of ₹2,000.", "refund_request"),
    ("I want a refund.", "refund_request"),
    ("Please cancel my recharge and give my money back", "refund_request"),
    ("refund my last recharge", "refund_request"),
    ("How do I cancel a recharge?", "requirement_inquiry"),
    ("What is your refund policy?", "requirement_inquiry"),
    ("Should I do that before I travel?", "requirement_inquiry"),
])
def test_rule_intent(msg, expected):
    assert intent.rule_intent(msg) == expected


def test_llm_intent_is_used_and_validated(monkeypatch):
    monkeypatch.setattr(llm, "complete_json", lambda s, u: {"intent": "refund_request"})
    assert intent.classify("I need my money back for the recharge") == ("refund_request", "llm")
    monkeypatch.setattr(llm, "complete_json", lambda s, u: {"intent": "something_else"})
    assert intent.classify("Why is my internet slow?") == ("requirement_inquiry", "rules")


# ---------- Validation rules (deterministic) ----------

def test_validation_rules():
    v = accounts.validate_refund("user_123", 499)
    assert v == {"requested_amount": 499.0, "recharge_amount": 499, "recharge_id": "rch_457",
                 "eligible": True, "reason": "Refund is eligible."}
    assert accounts.validate_refund("user_123", 999)["recharge_id"] == "rch_456"
    assert not accounts.validate_refund("user_123", 0)["eligible"]
    assert not accounts.validate_refund("user_123", -5)["eligible"]
    over = accounts.validate_refund("user_123", 2000)
    assert not over["eligible"] and "exceeds the recharge amount" in over["reason"]
    # A partial amount applies to the most recent recharge that can cover it.
    assert accounts.validate_refund("user_123", 300)["recharge_id"] == "rch_457"


def test_exact_match_identifies_recharge_even_if_refunded():
    recharge("rch_457")["refund_status"] = "refunded"
    v = accounts.validate_refund("user_123", 499)
    assert v["recharge_id"] == "rch_457" and not v["eligible"] and "already been refunded" in v["reason"]


def test_process_refund_cannot_bypass_rules():
    assert not accounts.process_refund("user_123", 999, "rch_456")["success"]  # above the approval limit
    recharge("rch_457")["recharge_age_days"] = 15
    assert not accounts.process_refund("user_123", 499, "rch_457")["success"]


def test_model_cannot_call_execution_tools():
    names = {t["function"]["name"] for t in refund_agent.TOOLS}
    assert names == {"get_user_account", "validate_refund"}


# ---------- The seven demonstration scenarios (rules path: no model configured in tests) ----------

def test_scenario_1_successful_refund(spy):
    meta, text, done = run("I want a refund of ₹499.")
    assert meta["route"] == "refund" and meta["intent"] == "refund_request"
    r = done["refund"]
    assert r["state"] == "awaiting_confirmation" and r["amount"] == 499
    assert "Refund eligible: ₹499" in text and spy["process"] == 0
    assert recharge("rch_457")["refund_status"] == "not_refunded"

    _, text, done = run("Yes, proceed", meta["session_id"])
    assert done["refund"]["state"] == "refund_processed" and done["refund"]["refund_id"].startswith("ref_")
    assert spy["process"] == 1 and recharge("rch_457")["refund_status"] == "refunded"
    assert recharge("rch_456")["refund_status"] == "not_refunded"  # the other recharge is untouched
    steps = done["refund"]["steps"]
    for expected in ("[Intent] refund_request", "[Agent] get_user_account", "[Agent] validate_refund(amount=499)",
                     "[Validation] eligible=True", "[Agent] awaiting_user_confirmation", "[User] confirmed",
                     "[Agent] process_refund(amount=499)", "[Refund] success"):
        assert expected in steps


def test_scenario_2_amount_exceeds_recharge(spy):
    _, text, done = run("Give me a refund of ₹2,000.")
    steps = done["refund"]["steps"]
    assert any(s.startswith("[Validation] eligible=False") and "exceeds the recharge amount" in s for s in steps)
    # Not one of the recharges: the real options are offered, and no person.
    assert "₹2,000 isn't one of your recharges" in text and "₹999 and ₹499" in text
    assert done["refund"]["state"] == "awaiting_amount" and not done["handoff"]["offer"]
    assert spy["process"] == 0 and spy["support"] == 0


@pytest.mark.parametrize("msg,asked", [("I want a refund of ₹1000", "₹1,000"), ("I want a refund of ₹300", "₹300"),
                                       ("I want a refund of ₹750", "₹750")])
def test_amount_not_a_recharge_offers_the_listed_ones(spy, msg, asked):
    meta, text, done = run(msg)
    assert f"{asked} isn't one of your recharges" in text
    assert "Your recharges that can be refunded are ₹999 and ₹499. Which one" in text
    assert "Talk to a person" not in text and not done["handoff"]["offer"]
    assert done["refund"]["state"] == "awaiting_amount" and spy["process"] == 0 and spy["support"] == 0
    assert done["refund"]["options"] == [999, 499]  # shown as buttons
    _, _, done = run("₹499", meta["session_id"])  # picking a listed recharge continues normally
    assert done["refund"]["state"] == "awaiting_confirmation" and done["refund"]["amount"] == 499


def test_unmatched_amount_with_one_refundable_recharge_offers_it(spy):
    recharge("rch_456")["refund_status"] = "refunded"  # only ₹499 left
    meta, text, done = run("I want a refund of ₹1000")
    assert "Your recharge that can be refunded is ₹499. Would you like that one refunded?" in text
    _, _, done = run("yes", meta["session_id"])
    assert done["refund"]["state"] == "awaiting_confirmation" and spy["process"] == 0


def test_unmatched_amount_with_nothing_refundable(monkeypatch, spy):
    monkeypatch.setenv("NOVACELL_DEMO_USER", "user_456")
    _, text, done = run("I want a refund of ₹1000")
    assert "None of your recharges can be refunded right now" in text
    assert done["refund"]["state"] == "refund_rejected" and not done["handoff"]["offer"]


@pytest.mark.parametrize("setup,reason", [("old", "7-day refund window"), ("refunded", "already been refunded")])
def test_real_recharge_that_cant_be_refunded_still_offers_a_person(setup, reason):
    r = recharge("rch_457")
    if setup == "old":
        r["recharge_age_days"] = 15
    else:
        r["refund_status"] = "refunded"
    _, text, done = run("I want a refund of ₹499")
    assert done["refund"]["state"] == "refund_rejected" and reason in text
    assert "Talk to a person" in text and done["handoff"]["offer"]
    assert "refund_rejected" in done["handoff"]["reasons"]


def test_scenario_3_recharge_too_old(spy):
    for r in accounts.USERS["user_123"]["recharges"]:
        r["recharge_age_days"] = 15
    _, text, done = run("I want a refund of ₹499.")
    assert done["refund"]["state"] == "refund_rejected" and "7-day refund window" in text
    assert spy["process"] == 0


def test_scenario_4_already_refunded(spy):
    recharge("rch_457")["refund_status"] = "refunded"
    _, text, done = run("I want a refund of ₹499.")
    assert done["refund"]["state"] == "refund_rejected" and "already been refunded" in text
    assert spy["process"] == 0


def test_scenario_5_rag_query_unchanged():
    meta, text, done = run("Why is my mobile internet slow?")
    assert meta["route"] == "answer" and meta["intent"] == "requirement_inquiry"
    assert "refund" not in done and done["citations"][0]["id"] == "FAQ-2"


def test_scenario_6_above_limit_goes_to_support(spy):
    _, text, done = run("I want a refund of ₹999.")
    r = done["refund"]
    assert r["state"] == "refund_submitted_for_review" and r["request_id"].startswith("SR-")
    assert "submitted to our Support Team" in text and "process it or take the necessary action" in text
    for claim in ("approved", "rejected", "processed"):
        assert claim not in text.replace("process it", "")
    assert spy["process"] == 0 and spy["support"] == 1
    assert recharge("rch_456")["refund_status"] == "not_refunded"
    assert accounts.SUPPORT_QUEUE[0]["amount"] == 999


@pytest.mark.parametrize("user,msg,amount", [("user_123", "I want a refund of ₹999", 999),
                                             ("user_789", "I want a refund of ₹599", 599),
                                             ("user_789", "I want a refund of ₹1,499", 1499)])
def test_above_limit_offers_a_person(spy, user, msg, amount):
    _, text, done = run(msg, customer_id=user)
    assert done["refund"]["state"] == "refund_submitted_for_review" and done["refund"]["amount"] == amount
    assert "Talk to a person" in text and done["handoff"]["offer"]
    assert "refund_above_limit" in done["handoff"]["reasons"] and spy["process"] == 0


@pytest.mark.parametrize("user,msg", [("user_123", "I want a refund of ₹499"), ("user_789", "I want a refund of ₹149"),
                                      ("user_321", "I want a refund of ₹99")])
def test_at_or_below_limit_is_refunded_here_after_confirmation(spy, user, msg):
    meta, text, done = run(msg, customer_id=user)
    assert done["refund"]["state"] == "awaiting_confirmation" and not done["handoff"]["offer"]
    _, _, done = run("Yes, proceed", meta["session_id"], customer_id=user)
    assert done["refund"]["state"] == "refund_processed" and spy["process"] == 1 and spy["support"] == 0


def test_scenario_7_follow_up_gets_context(monkeypatch):
    seen = []

    def fake_stream(system, user):
        seen.append(user)
        yield "text", "Yes, turn it on at least 24 hours before you travel [FAQ-9]."
        yield "stop", "end"
    monkeypatch.setattr(assistant, "has_credentials", lambda: True)
    monkeypatch.setattr(assistant.llm, "stream_answer", fake_stream)
    meta, _, _ = run("How do I activate international roaming?")
    meta2, _, done = run("Should I do that before I travel?", meta["session_id"])
    assert meta2["intent"] == "requirement_inquiry" and done["grounded"]
    assert "How do I activate international roaming?" in seen[-1]  # previous exchange passed as context


# ---------- Confirmation, amounts, clearing ----------

def test_original_request_and_ambiguous_replies_do_not_execute(spy):
    meta, _, _ = run("Refund ₹499 now, I confirm")
    sid = meta["session_id"]
    for reply in ("ok", "maybe", "sure"):
        _, text, done = run(reply, sid)
        assert done["refund"]["state"] == "awaiting_confirmation" and "clear answer" in text
    assert spy["process"] == 0


def test_cancel_does_not_execute(spy):
    meta, _, _ = run("I want a refund of ₹499")
    _, text, done = run("No, cancel", meta["session_id"])
    assert done["refund"]["state"] == "refund_cancelled" and spy["process"] == 0
    assert recharge("rch_457")["refund_status"] == "not_refunded"


def test_button_from_an_older_refund_changes_nothing(spy):
    meta, _, first = run("I want a refund of ₹149", customer_id="user_789")
    sid = meta["session_id"]
    _, _, second = run("Actually make it ₹299", sid, customer_id="user_789")  # supersedes the first request
    assert second["refund"]["id"] != first["refund"]["id"]
    _, text, done = run("Yes, proceed", sid, action="confirm", refund_id=first["refund"]["id"], customer_id="user_789")
    assert "earlier request" in text and spy["process"] == 0
    assert done["refund"]["state"] == "awaiting_confirmation"


def test_only_whole_recharges_are_refunded(spy):
    # A partial amount is not refunded against a bigger recharge any more; the options are offered instead.
    _, text, done = run("I want a refund of ₹300")
    assert done["refund"]["state"] == "awaiting_amount" and spy["process"] == 0
    assert recharge("rch_457")["refund_status"] == "not_refunded"


def test_button_confirm(spy):
    meta, _, done = run("I want a refund of ₹499")
    _, _, done = run("Confirm Refund", meta["session_id"], action="confirm", refund_id=done["refund"]["id"])
    assert done["refund"]["state"] == "refund_processed" and spy["process"] == 1


def test_stale_button_after_clear_does_nothing(spy):
    meta, _, done = run("I want a refund of ₹499")
    sid, rid = meta["session_id"], done["refund"]["id"]
    assistant.clear_session(sid)
    _, text, _ = run("Yes, proceed", sid, action="confirm", refund_id=rid)
    assert "no longer waiting" in text and spy["process"] == 0
    assert assistant.get_session(sid).turns[:-1] == []


def test_missing_amount_with_two_recharges_asks_which():
    meta, text, done = run("I want a refund.")
    assert done["refund"]["state"] == "awaiting_amount"
    assert "₹999 and ₹499" in text and "Which amount" in text
    _, text, done = run("₹499", meta["session_id"])
    assert done["refund"]["state"] == "awaiting_confirmation" and done["refund"]["amount"] == 499


def test_missing_amount_with_one_recharge_offers_it(spy):
    recharge("rch_456")["refund_status"] = "refunded"
    meta, text, done = run("I want a refund")
    assert done["refund"]["state"] == "awaiting_amount" and "one recent recharge of ₹499" in text
    _, _, done = run("yes", meta["session_id"])
    assert done["refund"]["state"] == "awaiting_confirmation" and spy["process"] == 0


def test_refunding_one_recharge_leaves_the_other_refundable():
    meta, _, _ = run("I want a refund of ₹499")
    run("Yes, proceed", meta["session_id"])
    assert accounts.validate_refund("user_123", 999)["eligible"]


def test_question_while_awaiting_confirmation_keeps_refund_pending(spy):
    meta, _, _ = run("I want a refund of ₹499")
    sid = meta["session_id"]
    meta2, _, _ = run("How do I enable Wi-Fi calling?", sid)
    assert meta2["route"] == "answer"
    _, _, done = run("Yes, proceed", sid)
    assert done["refund"]["state"] == "refund_processed" and spy["process"] == 1


def test_safety_routes_take_priority_over_refunds():
    meta, _, _ = run("I was hacked and someone recharged, refund it")
    assert meta["route"] == "escalate"


def test_alternate_demo_user(monkeypatch, spy):
    monkeypatch.setenv("NOVACELL_DEMO_USER", "user_456")
    _, text, done = run("I want a refund of ₹299")
    assert done["refund"]["state"] == "refund_rejected" and "15 days ago" in text
    _, text, done = run("I want a refund of ₹199")
    assert "already been refunded" in text and spy["process"] == 0


# ---------- Extra demo customers and the picker ----------

def test_rahul_keeps_exactly_the_prd_recharges():
    amounts = sorted(r["recharge_amount"] for r in accounts.USERS["user_123"]["recharges"])
    assert amounts == [499, 999]


@pytest.mark.parametrize("user,amount,eligible,reason", [
    ("user_456", 399, False, "pending"),
    ("user_456", 249, False, "failed"),
    ("user_456", 799, False, "7-day refund window"),
    ("user_456", 199, False, "already been refunded"),
    ("user_789", 149, True, None),
    ("user_789", 299, True, None),
    ("user_789", 599, True, None),
    ("user_789", 1499, True, None),
    ("user_321", 99, True, None),
    ("user_321", 1999, True, None),
    ("user_321", 349, False, "already been refunded"),
    ("user_321", 599, False, "pending"),
    ("user_321", 449, False, "7-day refund window"),
])
def test_new_demo_recharges(user, amount, eligible, reason):
    v = accounts.validate_refund(user, amount)
    assert v["eligible"] is eligible
    if reason:
        assert reason in v["reason"]


def test_picker_switches_customer_and_starts_fresh(spy):
    meta, _, done = run("I want a refund of ₹149", customer_id="user_789")
    sid = meta["session_id"]
    assert meta["customer_id"] == "user_789" and done["refund"]["state"] == "awaiting_confirmation"
    # Switching customer drops the pending refund, so it can't be confirmed for someone else.
    meta, text, done = run("Yes, proceed", sid, customer_id="user_123")
    assert meta["customer_id"] == "user_123" and spy["process"] == 0
    assert assistant.get_session(sid).refund is None or assistant.get_session(sid).refund.user_id == "user_123"
    assert accounts.USERS["user_789"]["recharges"][0]["refund_status"] == "not_refunded"


def test_unknown_customer_is_ignored():
    meta, _, _ = run("I want a refund of ₹499", customer_id="user_does_not_exist")
    assert meta["customer_id"] == "user_123"


def test_new_customer_flows(spy):
    meta, _, done = run("I want a refund of ₹1,499", customer_id="user_789")
    assert done["refund"]["state"] == "refund_submitted_for_review" and done["handoff"]["offer"]
    meta, text, done = run("I want a refund", customer_id="user_789")
    assert "₹1,499, ₹599, ₹299 and ₹149. Which amount" in text  # every refundable recharge, largest first
    meta, _, done = run("I want a refund of ₹299", customer_id="user_789")
    _, _, done = run("Yes, proceed", meta["session_id"], customer_id="user_789")
    assert done["refund"]["state"] == "refund_processed" and spy["process"] == 1
    customers = {c["user_id"]: c for c in accounts.demo_customers()}
    r299 = next(r for r in customers["user_789"]["recharges"] if r["recharge_amount"] == 299)
    assert r299["refund_status"] == "refunded" and not r299["refundable"]


# ---------- Model-driven tool loop, with a fake model ----------

def _call(cid, name, args):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _script(monkeypatch, *turns):
    """Make chat_with_tools return the given turns in order (each a list of tool calls, or [] to stop)."""
    seq = iter(turns)
    seen = []

    def fake(messages, tools):
        seen.append(messages)
        calls = next(seq, [])
        return SimpleNamespace(content="", tool_calls=calls or None)
    monkeypatch.setattr(llm, "chat_with_tools", fake)
    monkeypatch.setattr(llm, "complete_json", lambda s, u: {"intent": "refund_request"})
    return seen


def test_model_drives_tools(monkeypatch, spy):
    seen = _script(monkeypatch,
                   [_call("1", "get_user_account", {"user_id": "user_123"})],
                   [_call("2", "validate_refund", {"user_id": "user_123", "amount": 499})],
                   [])
    meta, _, done = run("refund my ₹499 recharge please")
    assert meta["intent"] == "refund_request"
    assert done["refund"]["state"] == "awaiting_confirmation" and spy["process"] == 0
    tool_msgs = [m for m in seen[-1] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["user_id"] == "user_123"
    assert json.loads(tool_msgs[1]["content"])["eligible"] is True


def test_model_cannot_switch_user_or_invent_amount(monkeypatch, spy):
    _script(monkeypatch,
            [_call("1", "get_user_account", {"user_id": "user_999"})],
            [_call("2", "validate_refund", {"user_id": "user_999", "amount": 50})],
            [])
    _, text, done = run("I want a refund")
    steps = done["refund"]["steps"]
    assert any("ignored user_id='user_999'" in s for s in steps)
    assert any("not on the account" in s for s in steps)
    assert done["refund"]["state"] == "awaiting_amount" and "₹999 and ₹499" in text


def test_customer_amount_beats_model_amount(monkeypatch):
    _script(monkeypatch, [_call("1", "validate_refund", {"user_id": "user_123", "amount": 999})], [])
    _, _, done = run("I want a refund of ₹499")
    assert done["refund"]["amount"] == 499 and done["refund"]["state"] == "awaiting_confirmation"


def test_model_may_resolve_last_recharge_from_account(monkeypatch):
    _script(monkeypatch,
            [_call("1", "get_user_account", {"user_id": "user_123"})],
            [_call("2", "validate_refund", {"user_id": "user_123", "amount": 499})], [])
    _, _, done = run("refund my last recharge")
    assert done["refund"]["amount"] == 499 and done["refund"]["state"] == "awaiting_confirmation"


def test_model_failure_falls_back_to_rules(monkeypatch):
    def boom(messages, tools):
        raise llm.ServiceError("RateLimitError")
    monkeypatch.setattr(llm, "chat_with_tools", boom)
    monkeypatch.setattr(llm, "complete_json", lambda s, u: {"intent": "refund_request"})
    _, _, done = run("I want a refund of ₹499")
    assert done["refund"]["state"] == "awaiting_confirmation"
    assert any("model unavailable" in s for s in done["refund"]["steps"])
