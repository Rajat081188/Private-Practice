import re

import pytest

from app import config
from app.ingest import load_all
from app.pii import mask
from app.search import live_index


def test_all_three_sources_load():
    passages, errors = load_all()
    assert errors == []
    sources = {p.source for p in passages}
    assert sources == {"faq", "ticket", "guide"}
    assert sum(p.source == "faq" for p in passages) == 25


def test_unresolved_ticket_excluded():
    ids = {p.id for p in load_all()[0]}
    assert "TK-001" in ids and "TK-020" not in ids  # TK-020 is 'escalated', not resolved


def test_internal_guide_content_filtered():
    guide = [p for p in load_all()[0] if p.source == "guide"]
    assert not any(p.meta["section"] == 8 for p in guide), "fraud/security section is internal-only"
    for p in guide:
        assert not re.search(r"\bagents?\b", p.text, re.I), p.id
        assert "subscriber management system" not in p.text


@pytest.mark.parametrize("text,kind", [
    ("call me on +1 415 555 0134 please", "PHONE"),
    ("my email is jo.smith@example.com", "EMAIL"),
    ("card 4111 1111 1111 1111", "CARD"),
    ("account number AC-778812", "ACCOUNT"),
    ("sim serial 8944123456789012345", "SIM_SERIAL"),
])
def test_pii_masked(text, kind):
    masked, found = mask(text)
    assert kind in found
    assert f"[{kind}]" in masked


def test_service_numbers_not_masked():
    masked, found = mask("Dial *123# or call 611")
    assert found == [] and "611" in masked


@pytest.mark.parametrize("q,expected", [
    ("why is my internet so slow", "FAQ-2"),
    ("I am going to Spain next week", "FAQ-9"),
    ("my eSIM QR code won't scan", "TK-009"),
    ("why was I charged extra this month", "FAQ-16"),
    ("my calls keep dropping", "FAQ-6"),
    ("how do I enable wifi calling", "FAQ-7"),
    ("how do I keep my number when switching", "FAQ-13"),
])
def test_retrieval_top3(q, expected):
    assert expected in [h.passage.id for h in live_index.search(q)[:3]]


@pytest.mark.parametrize("q", ["who won the world cup", "write me a poem about cats", "best pizza near me"])
def test_out_of_scope_below_threshold(q):
    hits = live_index.search(q)
    assert not hits or hits[0].score < config.load_rules()["min_retrieval_score"]
