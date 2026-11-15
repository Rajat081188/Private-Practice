"""Append-only JSONL logs for answers, feedback and hand-offs (questions are PII-masked before they get here)."""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter
from datetime import datetime

from . import config

_lock = threading.Lock()


def _path(name: str):
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return config.RUNTIME_DIR / f"{name}.jsonl"


def _append(name: str, record: dict) -> None:
    record = {"ts": time.time(), **record}
    with _lock, open(_path(name), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read(name: str) -> list[dict]:
    p = _path(name)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def log_answer(record: dict) -> None:
    _append("answers", record)


def log_feedback(answer_id: str, helpful: bool, comment: str = "") -> None:
    _append("feedback", {"answer_id": answer_id, "helpful": helpful, "comment": comment[:500]})


def create_handoff(session_id: str, reasons: list[str], transcript: list[tuple[str, str]], now: datetime | None = None) -> dict:
    rules = config.load_rules()
    hours = rules.get("live_chat_hours", {"start": 8, "end": 22})
    now = now or datetime.now()
    live = hours["start"] <= now.hour < hours["end"]
    record = {
        "handoff_id": "HO-" + uuid.uuid4().hex[:8].upper(),
        "session_id": session_id,
        "reasons": reasons,
        "channel": "live_chat" if live else "ticket",
        "transcript": [{"customer": u, "assistant": a} for u, a in transcript],
    }
    _append("handoffs", record)
    if live:
        msg = (f"You're in the queue for live chat (reference {record['handoff_id']}). The agent will see this "
               "conversation, so you won't need to repeat yourself.")
    else:
        msg = f"{rules.get('out_of_hours_message', '')} Your reference is {record['handoff_id']}."
    return {**record, "message": msg}


def stats() -> dict:
    answers, feedback, handoffs = _read("answers"), _read("feedback"), _read("handoffs")
    n = len(answers)
    ups = sum(1 for f in feedback if f["helpful"])
    reasons = Counter(r for a in answers for r in a.get("reasons", []))
    unanswered = [a["question"] for a in answers if a.get("abstained")][-20:]
    low_rated_ids = {f["answer_id"] for f in feedback if not f["helpful"]}
    low_rated = [{"question": a["question"], "citations": a.get("citations", []),
                  "comment": next((f.get("comment", "") for f in feedback
                                   if f["answer_id"] == a["answer_id"] and not f["helpful"]), "")}
                 for a in answers if a["answer_id"] in low_rated_ids][-20:]
    sessions = {a["session_id"] for a in answers}
    handoff_sessions = {h["session_id"] for h in handoffs}
    ttfts = sorted(a["ttft_ms"] for a in answers if a.get("ttft_ms"))
    return {
        "answers": n,
        "sessions": len(sessions),
        "containment_rate": round(1 - len(handoff_sessions & sessions) / len(sessions), 3) if sessions else None,
        "handoffs": len(handoffs),
        "abstentions": sum(1 for a in answers if a.get("abstained")),
        "ungrounded": sum(1 for a in answers if not a.get("grounded", True)),
        "helpful_rate": round(ups / len(feedback), 3) if feedback else None,
        "feedback_count": len(feedback),
        "p95_ttft_ms": ttfts[int(0.95 * (len(ttfts) - 1))] if ttfts else None,
        "routes": Counter(a["route"] for a in answers),
        "top_reasons": reasons.most_common(10),
        "unanswered": unanswered,
        "low_rated": low_rated,
    }
