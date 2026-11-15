import json
from datetime import datetime

from fastapi.testclient import TestClient

from app import config, store
from app.server import app

client = TestClient(app)


def sse(resp):
    return [json.loads(l[6:]) for l in resp.text.split("\n\n") if l.startswith("data: ")]


def test_health():
    h = client.get("/api/health").json()
    assert h["ok"] and h["passages"] > 50 and h["source_errors"] == []


def test_chat_stream_and_feedback():
    events = sse(client.post("/api/chat", json={"message": "How do I enable Wi-Fi calling?"}))
    assert events[0]["type"] == "meta" and events[-1]["type"] == "done"
    done = events[-1]
    assert client.post("/api/feedback", json={"answer_id": done["answer_id"], "helpful": False,
                                              "comment": "too long"}).json()["ok"]
    stats = client.get("/api/ops/stats").json()
    assert stats["feedback_count"] >= 1 and stats["answers"] >= 1


def test_handoff_carries_transcript():
    events = sse(client.post("/api/chat", json={"message": "I want to talk to a human"}))
    sid = events[0]["session_id"]
    r = client.post("/api/handoff", json={"session_id": sid, "reasons": ["asked_for_human"]}).json()
    assert r["handoff_id"].startswith("HO-") and r["transcript"][0]["customer"] == "I want to talk to a human"


def test_handoff_channel_follows_chat_hours():
    assert store.create_handoff("s", [], [], now=datetime(2026, 11, 16, 12))["channel"] == "live_chat"
    out = store.create_handoff("s", [], [], now=datetime(2026, 11, 16, 23))
    assert out["channel"] == "ticket" and "8am" in out["message"]


def test_bad_faq_upload_rejected_and_live_content_unchanged():
    before = config.FAQ_PATH.read_bytes()
    r = client.post("/api/ops/upload/faq", files={"file": ("faq.csv", b"q,a\nhi,there\n", "text/csv")})
    assert r.status_code == 400 and "missing columns" in r.json()["detail"]
    assert config.FAQ_PATH.read_bytes() == before


def test_faq_upload_goes_live_and_rolls_back():
    original = config.FAQ_PATH.read_bytes()
    new = original + b'26,Do you offer 5G home broadband?,"Yes. Check availability in MyTelecom > Home Broadband.",data\n'
    r = client.post("/api/ops/upload/faq", files={"file": ("faq.csv", new, "text/csv")})
    assert r.status_code == 200 and r.json()["entries"] == 26
    hits = client.get("/api/ops/search", params={"q": "5G home broadband"}).json()
    assert hits[0]["id"] == "FAQ-26"

    version = client.get("/api/ops/sources").json()["sources"][0]["versions"][0]
    assert client.post(f"/api/ops/rollback/faq/{version}").status_code == 200
    assert config.FAQ_PATH.read_bytes() == original
    hits = client.get("/api/ops/search", params={"q": "5G home broadband"}).json()
    assert all(h["id"] != "FAQ-26" for h in hits)


def test_rules_upload_validated():
    r = client.post("/api/ops/upload/rules", files={"file": ("rules.json", b'{"x": 1}', "application/json")})
    assert r.status_code == 400
