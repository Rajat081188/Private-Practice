"""FastAPI app: customer chat, hand-off, feedback, and the support-ops content console."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import accounts, assistant, config, llm, store
from .ingest import load_faq, load_guide, load_tickets
from .search import live_index

# Workflow steps ([Intent], [Agent], [Validation], [Refund]) go to the server console (agent PRD section 12).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("novacell.agent").setLevel(logging.INFO)

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="NovaCell Support Assistant")

SOURCES = {
    "faq": (config.FAQ_PATH, ".csv"),
    "tickets": (config.TICKETS_PATH, ".db"),
    "guide": (config.GUIDE_PATH, ".pdf"),
    "rules": (config.RULES_PATH, ".json"),
}


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    action: Literal["confirm", "cancel"] | None = None  # from the refund confirmation buttons
    refund_id: str | None = None


class SessionIn(BaseModel):
    session_id: str


class FeedbackIn(BaseModel):
    answer_id: str
    helpful: bool
    comment: str = ""


class HandoffIn(BaseModel):
    session_id: str
    reasons: list[str] = []


def ops_auth(x_ops_token: str | None = Header(default=None)):
    token = os.environ.get("NOVACELL_OPS_TOKEN")
    if token and x_ops_token != token:
        raise HTTPException(401, "Ops token required")


# Browsers must recheck the pages, or they keep showing an old copy after an update.
NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/")
def chat_page():
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


@app.get("/ops")
def ops_page():
    return FileResponse(STATIC / "ops.html", headers=NO_CACHE)


@app.get("/api/health")
def health():
    idx = live_index.get()
    return {"ok": True, "passages": len(idx.passages), "source_errors": live_index.errors,
            "mode": "llm" if assistant.has_credentials() else "offline",
            "provider": llm.provider(), "model": llm.model_name()}


@app.post("/api/chat")
def chat(body: ChatIn):
    def events():
        for ev in assistant.respond(body.message, body.session_id, body.action, body.refund_id):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/session/clear")
def clear_session(body: SessionIn):
    assistant.clear_session(body.session_id)
    return {"ok": True}


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    store.log_feedback(body.answer_id, body.helpful, body.comment)
    return {"ok": True}


@app.post("/api/handoff")
def handoff(body: HandoffIn):
    session = assistant.get_session(body.session_id)
    return store.create_handoff(session.id, body.reasons, session.turns)


# ---------- Ops console ----------

def _validate(kind: str, path: Path) -> int:
    if kind == "faq":
        return len(load_faq(path))
    if kind == "tickets":
        return len(load_tickets(path))
    if kind == "guide":
        return len(load_guide(path, config.load_rules()))
    rules = json.loads(path.read_text(encoding="utf-8"))
    for key in ("min_retrieval_score", "live_chat_hours", "source_priority"):
        if key not in rules:
            raise ValueError(f"rules file is missing '{key}'")
    return len(rules)


def _versions(kind: str) -> list[str]:
    d = config.VERSIONS_DIR / kind
    return sorted((p.name for p in d.glob("*")), reverse=True) if d.exists() else []


@app.get("/api/ops/sources", dependencies=[Depends(ops_auth)])
def sources():
    idx = live_index.get()
    counts = {}
    for p in idx.passages:
        counts[p.source] = counts.get(p.source, 0) + 1
    out = []
    for kind, (path, _) in SOURCES.items():
        st = path.stat() if path.exists() else None
        out.append({
            "kind": kind, "file": path.name, "exists": bool(st),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)) if st else None,
            "passages": counts.get({"tickets": "ticket"}.get(kind, kind)),
            "versions": _versions(kind)[:10],
        })
    return {"sources": out, "errors": live_index.errors, "total_passages": len(idx.passages)}


@app.post("/api/ops/upload/{kind}", dependencies=[Depends(ops_auth)])
async def upload(kind: str, file: UploadFile = File(...)):
    if kind not in SOURCES:
        raise HTTPException(404, "Unknown source")
    target, ext = SOURCES[kind]
    if not (file.filename or "").lower().endswith(ext):
        raise HTTPException(400, f"Expected a {ext} file")
    staging = config.DATA_DIR / ".staging"
    staging.mkdir(exist_ok=True)
    tmp = staging / f"{kind}{ext}"
    tmp.write_bytes(await file.read())
    try:
        n = _validate(kind, tmp)
        if n == 0:
            raise ValueError("file contains no usable entries")
    except Exception as e:  # noqa: BLE001 - shown to the ops user
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"Rejected, the live content is unchanged: {e}")
    # Keep the current file so ops can roll back.
    if target.exists():
        vdir = config.VERSIONS_DIR / kind
        vdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, vdir / f"{time.strftime('%Y%m%d-%H%M%S')}{ext}")
    os.replace(tmp, target)
    live_index.get()
    return {"ok": True, "entries": n, "message": f"Live now: {n} entries from {file.filename}."}


@app.post("/api/ops/rollback/{kind}/{version}", dependencies=[Depends(ops_auth)])
def rollback(kind: str, version: str):
    if kind not in SOURCES or version not in _versions(kind):
        raise HTTPException(404, "Unknown version")
    target, ext = SOURCES[kind]
    src = config.VERSIONS_DIR / kind / version
    vdir = config.VERSIONS_DIR / kind
    if target.exists():
        shutil.copy2(target, vdir / f"{time.strftime('%Y%m%d-%H%M%S')}-before-rollback{ext}")
    shutil.copy2(src, target)
    os.utime(target)  # force the index to notice
    live_index.get()
    return {"ok": True, "message": f"Restored {kind} from {version}."}


@app.get("/api/ops/demo-accounts", dependencies=[Depends(ops_auth)])
def demo_accounts():
    return {"session_user": accounts.session_user_id(), "users": accounts.USERS,
            "support_queue": accounts.SUPPORT_QUEUE}


@app.post("/api/ops/demo-accounts/reset", dependencies=[Depends(ops_auth)])
def reset_demo_accounts():
    accounts.reset_demo_data()
    return {"ok": True, "message": "Demo accounts restored: both recharges are refundable again."}


@app.get("/api/ops/stats", dependencies=[Depends(ops_auth)])
def ops_stats():
    return store.stats()


@app.get("/api/ops/search", dependencies=[Depends(ops_auth)])
def ops_search(q: str):
    return [{"id": h.passage.id, "source": h.passage.source, "title": h.passage.title,
             "score": round(h.score, 2), "text": h.passage.text} for h in live_index.search(q)]
