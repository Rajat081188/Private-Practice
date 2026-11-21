"""Golden-set evaluation (PRD section 7).

Offline (free):   python -m eval.run_eval
  - routing accuracy, retrieval hit@3 for answerable questions, decline rate for out-of-scope ones.
With the model:   python -m eval.run_eval --llm      (needs GROQ_API_KEY or ANTHROPIC_API_KEY)
  - adds grounded-answer rate, citation validity, and correct abstention.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import assistant, config  # noqa: E402
from app.guardrails import route  # noqa: E402
from app.intent import rule_intent  # noqa: E402
from app.search import live_index  # noqa: E402

GOLDEN = Path(__file__).with_name("golden.jsonl")


def predicted_route(q: str) -> str:
    """Offline prediction: guardrails, then intent rules (the live app uses the model for intent)."""
    r = route(q)
    if r.kind in ("emergency", "decline") or {"fraud_or_security", "asked_for_human"} & set(r.reasons):
        return r.kind
    if rule_intent(q) == "refund_request":
        return "refund"
    if r.kind != "answer" and r.kind != "account_data":
        return r.kind
    hits = live_index.search(q)
    if not hits or hits[0].score < config.load_rules()["min_retrieval_score"]:
        return "decline"
    return r.kind


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also run full answers through the model")
    args = ap.parse_args()
    cases = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]

    route_ok, hit_n, hit_ok, failures = 0, 0, 0, []
    for c in cases:
        pr = predicted_route(c["q"])
        if pr == c["route"]:
            route_ok += 1
        else:
            failures.append(f"route  {c['q']!r}: expected {c['route']}, got {pr}")
        if c.get("sources"):
            hit_n += 1
            top = [h.passage.id for h in live_index.search(c["q"])[:3]]
            if set(top) & set(c["sources"]):
                hit_ok += 1
            else:
                failures.append(f"recall {c['q']!r}: expected one of {c['sources']}, top3 {top}")

    print(f"Cases: {len(cases)}")
    print(f"Routing accuracy:     {route_ok}/{len(cases)} = {route_ok / len(cases):.0%}")
    print(f"Retrieval hit@3:      {hit_ok}/{hit_n} = {hit_ok / hit_n:.0%}")

    if args.llm:
        if not assistant.has_credentials():
            print("--llm needs GROQ_API_KEY or ANTHROPIC_API_KEY in the environment or .env.")
            return 2
        grounded = abst_ok = abst_n = ans_n = refund_ok = refund_n = 0
        for c in cases:
            events = list(assistant.respond(c["q"]))
            done = events[-1]
            if c["route"] == "refund" or events[0].get("route") == "refund":
                refund_n += c["route"] == "refund"
                ok = (events[0].get("route") == "refund") == (c["route"] == "refund")
                refund_ok += ok and c["route"] == "refund"
                if not ok:
                    failures.append(f"intent  {c['q']!r}: expected {c['route']}, got {events[0].get('route')}")
                continue
            if c["route"] == "decline":
                abst_n += 1
                abst_ok += bool(done.get("abstained"))
                if not done.get("abstained"):
                    failures.append(f"abstain {c['q']!r}: bot answered instead of declining")
            elif c["route"] in ("answer", "account_data"):
                ans_n += 1
                grounded += bool(done["grounded"] and not done.get("abstained"))
                if not done["grounded"]:
                    text = "".join(e["text"] for e in events if e["type"] == "delta")
                    failures.append(f"ground  {c['q']!r}: unverified answer: {text[:120]}")
        print(f"Grounded answers:     {grounded}/{ans_n} = {grounded / ans_n:.0%}   (PRD target >= 95%)")
        print(f"Correct abstention:   {abst_ok}/{abst_n} = {abst_ok / abst_n:.0%}   (PRD target >= 90%)")
        print(f"Refunds to agent:     {refund_ok}/{refund_n} (model intent detection)")

    if failures:
        print("\nFailures:")
        for f in failures:
            print("  " + f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
