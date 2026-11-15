# NovaCell Support Assistant (v1)

A self-serve chat assistant that answers Tier-1 telecom questions using only NovaCell's FAQ, resolved tickets and PDF guide, cites its sources, declines when unsure, and hands off to a person. Built from the PRD in `PRD.md`.

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows; use .venv/bin/python on macOS/Linux
.venv/Scripts/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Put your model key in a `.env` file in the project folder (git-ignored; see `.env.example`): `GROQ_API_KEY=gsk_...` for Groq, or `ANTHROPIC_API_KEY=sk-ant-...` for Claude.

- Customer chat: http://127.0.0.1:8000/
- Support ops console: http://127.0.0.1:8000/ops (set `NOVACELL_OPS_TOKEN` to require an `X-Ops-Token` header)

**Modes.** With `GROQ_API_KEY` set, answers are written by `openai/gpt-oss-120b` on Groq (reasoning effort `low`, reasoning hidden from customers). With only `ANTHROPIC_API_KEY` set, Claude (`claude-opus-5-5`) is used instead. Without a key the app runs in *offline mode*: every guardrail, escalation and search still works, and answers show the best-matching source passage word for word. The provider code is in `app/llm.py`.

Settings: `NOVACELL_PROVIDER` (`groq` or `anthropic`, to force one), `GROQ_MODEL`, `GROQ_REASONING_EFFORT`, `NOVACELL_MODEL`, `NOVACELL_EFFORT`, `NOVACELL_DATA_DIR`, `NOVACELL_RUNTIME_DIR`.

**Groq rate limits.** The current key allows 8,000 tokens per minute, and each answer uses about 2,000–3,000, so roughly three answers a minute. Requests over the limit wait and retry, which shows up as slow answers. A paid Groq tier is needed for real traffic.

## How a question is handled

1. **PII masking** (`app/pii.py`): phone, email, card, account and SIM-serial numbers are removed before anything is logged or sent to the model.
2. **Guardrails** (`app/guardrails.py`): fixed rules, not the model, pick the route: emergency, fraud/security, a request for a person, account actions (refunds, disputes), prompt-injection attempts, personal-data questions, lost/stolen phones, frustration, and repeated questions.
3. **Retrieval** (`app/search.py`): BM25 over all three sources with telecom synonyms. Below `min_retrieval_score` the bot says it isn't sure and offers a person, without calling the model.
4. **Grounded answer** (`app/assistant.py`): the model sees only the retrieved passages and must cite them as `[FAQ-10]`, `[TK-004]`, `[GUIDE-4.2]`, or reply `NO_ANSWER`. Citations are checked after the answer streams; a missing or invented citation marks the answer unverified and offers a hand-off. On the Claude path, server-side refusal fallbacks (`fallbacks: "default"`) are enabled.
5. **Hand-off** (`app/store.py`): during live-chat hours (8am–10pm) the case is queued for live chat with the transcript; outside them a ticket is logged. Records go to `runtime/handoffs.jsonl`.

## Content ops (no engineering release)

`/ops` lets support ops replace the FAQ CSV, tickets DB, guide PDF or `data/content_rules.json`. Each upload is validated first (a bad file is rejected and the live one is untouched), the previous file is kept for one-click rollback, and the index rebuilds on the next question. The console also shows containment, helpful rate, hand-offs, questions the bot declined, low-rated answers, and a retrieval preview.

`content_rules.json` controls the confidence threshold, source priority (FAQ > guide > tickets), chat hours, and which guide content is internal-only. Guide section 8 (fraud and network security) and sentences about agent procedures are kept out of customer answers. Tickets are used as past-case patterns only; the unresolved ticket (TK-020) is excluded.

## Tests and evaluation

```bash
.venv/Scripts/python -m pytest -q          # 56 tests, no API calls (the model is faked)
.venv/Scripts/python -m eval.run_eval      # golden set: routing + retrieval, free
.venv/Scripts/python -m eval.run_eval --llm   # adds grounded-answer and abstention rates; needs a key
```

Current offline results on the 50-case starter set: routing 100%, retrieval hit@3 100%.

## Gaps against the PRD

- Retrieval is keyword (BM25) with synonyms, not hybrid keyword + semantic. Fine for 67 passages; add embeddings as content grows.
- The golden set has 50 cases; the PRD calls for at least 300 real customer questions before launch.
- Live chat and ticketing are simulated with JSONL records; connect them to the real queue.
- Sessions are in memory, and the ops console has only an optional shared token, so run it on a trusted network.
- The sources use "MyTelecom" and telecom.example.com (an open question in the PRD).
