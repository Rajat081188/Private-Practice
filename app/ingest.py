"""Load the three knowledge sources into uniform, citable passages.

Each passage has a stable id that the bot cites: FAQ-10, TK-004, GUIDE-4.2.
"""
from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from . import config
from .pii import mask


@dataclass
class Passage:
    id: str
    source: str  # faq | ticket | guide
    title: str
    text: str
    category: str = ""
    search_text: str = ""  # what BM25 indexes (may weight the title)
    meta: dict = field(default_factory=dict)


def load_faq(path: Path) -> list[Passage]:
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"id", "question", "answer", "category"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"FAQ file is missing columns: {sorted(missing)}")
        for row in reader:
            q, a = row["question"].strip(), row["answer"].strip()
            if not q or not a:
                continue
            out.append(Passage(
                id=f"FAQ-{row['id'].strip()}", source="faq", title=q, text=a,
                category=row["category"].strip(),
                # Question counted twice: it is the best match signal for a user question.
                search_text=f"{q} {q} {a} {row['category']}",
            ))
    return out


def load_tickets(path: Path) -> list[Passage]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT ticket_id, category, issue_type, description, resolution FROM tickets "
            "WHERE status = 'resolved'"
        ).fetchall()
    finally:
        con.close()
    out = []
    for tid, cat, issue, desc, res in rows:
        desc, _ = mask(desc)
        res, _ = mask(res)
        text = (f"Past resolved case. Problem: {desc} "
                f"What NovaCell staff did: {res}")
        out.append(Passage(
            id=tid, source="ticket", title=issue, text=text, category=cat,
            search_text=f"{issue} {issue} {desc} {res} {cat}",
        ))
    return out


_HEADING = re.compile(r"^(\d+)\.\s+(.+)$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _guide_sections(path: Path) -> list[tuple[int, str, str]]:
    reader = PdfReader(str(path))
    sections: list[tuple[int, str, str]] = []
    for page in reader.pages:
        lines = [l.strip() for l in (page.extract_text() or "").splitlines()]
        lines = [l for l in lines if l and "Internal Use Only" not in l and not re.fullmatch(r"Page \d+", l)]
        num, title, body = None, None, []
        for line in lines:
            m = _HEADING.match(line)
            if m and num is None:
                num, title = int(m.group(1)), m.group(2)
            elif num is not None:
                body.append(line)
        if num is not None:
            sections.append((num, title, " ".join(body)))
    return sections


def load_guide(path: Path, rules: dict) -> list[Passage]:
    internal = set(rules.get("internal_guide_sections", []))
    drop = [re.compile(p, re.I) for p in rules.get("drop_guide_sentences_matching", [])]
    out = []
    for num, title, body in _guide_sections(path):
        if num in internal:
            continue
        sentences = [s for s in _SENT_SPLIT.split(body) if not any(p.search(s) for p in drop)]
        # ~700-character chunks so a passage stays focused on one topic.
        chunks, cur = [], ""
        for s in sentences:
            if cur and len(cur) + len(s) > 700:
                chunks.append(cur.strip())
                cur = ""
            cur += " " + s
        if cur.strip():
            chunks.append(cur.strip())
        for i, chunk in enumerate(chunks, 1):
            out.append(Passage(
                id=f"GUIDE-{num}.{i}", source="guide", title=f"Guide section {num}: {title}",
                text=chunk, search_text=f"{title} {chunk}", meta={"section": num},
            ))
    return out


def load_all(rules: dict | None = None) -> tuple[list[Passage], list[str]]:
    """Load every source. A broken source is skipped and reported, never fatal."""
    rules = rules or config.load_rules()
    passages, errors = [], []
    for name, loader in (
        ("faq", lambda: load_faq(config.FAQ_PATH)),
        ("tickets", lambda: load_tickets(config.TICKETS_PATH)),
        ("guide", lambda: load_guide(config.GUIDE_PATH, rules)),
    ):
        try:
            passages.extend(loader())
        except Exception as e:  # noqa: BLE001 - surfaced to ops, bot keeps serving other sources
            errors.append(f"{name}: {e}")
    return passages, errors
