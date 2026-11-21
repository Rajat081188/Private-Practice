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

## Refund Agent (agent_PRD.md)

Every message first goes through intent detection (`app/intent.py`): the model returns `{"intent": "requirement_inquiry" | "refund_request"}` as JSON, with keyword rules as a fallback. Inquiries take the RAG pipeline above, unchanged. Refund requests go to the Refund Agent (`app/refund_agent.py`):

1. The model calls `get_user_account` and `validate_refund` (the only tools it can see). The signed-in user defaults to `user_123`, Rahul, whose two recharges (₹499 and ₹999) are fixed by the PRD. A "Demo customer" picker in the chat switches to other simulated customers (switching starts a fresh conversation), and "Recharges on this account" lists each recharge and whether it can be refunded:

   | Customer | Recharges |
   |---|---|
   | Rahul (`user_123`) | ₹499 (1 day), ₹999 (2 days) |
   | Priya (`user_456`) | none refundable: ₹399 pending, ₹249 failed, ₹199 already refunded, ₹299 and ₹799 older than 7 days |
   | Anjali (`user_789`) | all refundable: ₹149, ₹299 (here), ₹599, ₹1,499 (via Support Team) |
   | Vikram (`user_321`) | mixed: ₹99, ₹1,999 refundable; ₹349 refunded, ₹599 pending, ₹449 8 days old |
2. Application code checks the amount the model used against what the customer said and the account data, so it can't invent one. If no amount was given, the customer is asked (or offered the single eligible recharge).
3. `validate_refund` (`app/accounts.py`) decides eligibility with fixed rules: completed, within 7 days, more than ₹0, not above the recharge, and not already refunded. Only a whole recharge can be refunded: an amount that isn't one of the customer's recharges (₹1,000, ₹300) gets "that isn't one of your recharges" plus buttons for the ones that can be refunded, and no "Talk to a person" offer. A real recharge that can't be refunded (too old, already refunded, pending) is explained and does offer a person.
4. Up to ₹499: the chat shows a "Confirm Refund / Cancel" card, and `process_refund` runs only after an explicit yes. Above ₹499: `submit_to_support` queues it for the Support Team and the turn ends; the app never reports their decision.

A state machine (`refund_requested` through `refund_submitted_for_review`) makes it impossible to process a refund before validation and confirmation. Each step is logged to the console (`[Intent]`, `[Agent]`, `[Validation]`, `[Refund]`) and to `runtime/answers.jsonl`. Refunds are simulated in memory: restart the server, or `POST /api/ops/demo-accounts/reset`, to restore the demo data. "Clear conversation" resets the history and any pending refund.

Differences from the agent PRD's stack table: this app has no LangChain, LangGraph or Streamlit, so the workflow is plain Python and the existing FastAPI chat page; and it uses the existing Groq model (`openai/gpt-oss-120b`), since Qwen3.6-27B isn't available on the current key.

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
