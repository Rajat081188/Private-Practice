"""BM25 search over all three sources, rebuilt automatically when ops change a source file."""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass

from . import config
from .ingest import Passage, load_all

STOP = set("""a an and are as at be by can do does for from has have how i if in is it its me my of on or
our so that the their them then there these this to was we what when where which who why will with you your
i'm im it's dont don't doesn't isnt isn't am been being get got just also into than too very please hi hello
thanks thank should would could""".split())

# Customer words -> vocabulary the sources use.
SYNONYMS = {
    "esim": ["sim", "esim"], "e-sim": ["sim", "esim"], "internet": ["data", "internet"],
    "slow": ["slow", "speed"], "speeds": ["speed"], "bill": ["bill", "billing", "charge"],
    "charged": ["charge", "bill"], "charges": ["charge", "bill"], "overcharged": ["charge", "bill", "higher"],
    "abroad": ["roaming", "travel"], "travel": ["roaming", "travel"], "travelling": ["roaming", "travel"],
    "traveling": ["roaming", "travel"], "trip": ["roaming", "travel"], "overseas": ["roaming", "international"],
    "dropping": ["drop"], "dropped": ["drop"], "drops": ["drop"], "calls": ["call", "voice"],
    "wifi": ["wi-fi", "wifi"], "signal": ["signal", "coverage"], "reception": ["signal", "coverage"],
    "app": ["mytelecom", "app"], "login": ["log", "password"], "password": ["password", "log"],
    "tethering": ["hotspot"], "unlock": ["unlock"], "port": ["porting", "port"], "switch": ["port", "switching"],
    "extra": ["higher", "additional", "charge"], "expensive": ["higher", "charge"], "more": ["higher"],
    "usage": ["usage", "balance", "data"], "left": ["balance", "remaining"],
    "holiday": ["roaming", "travel"], "vacation": ["roaming", "travel"], "country": ["roaming", "international"],
}
# Destinations: a customer naming a country almost always means roaming.
for _place in ("spain france italy germany portugal greece europe eu uk england ireland netherlands usa us america "
               "canada mexico japan singapore australia zealand thailand india china dubai turkey egypt").split():
    SYNONYMS[_place] = ["roaming", "travel", _place]
STOP |= {"going", "next", "week", "tomorrow", "today", "soon"}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str, expand: bool = False) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-']*", text.lower())
    out = []
    for w in words:
        w = w.strip("'-")
        if not w or w in STOP:
            continue
        if expand and w in SYNONYMS:
            out.extend(_stem(s) for s in SYNONYMS[w])
        else:
            out.append(_stem(w))
    return out


@dataclass
class Hit:
    passage: Passage
    score: float


class Index:
    def __init__(self, passages: list[Passage], k1: float = 1.4, b: float = 0.75):
        self.passages = passages
        self.docs = [Counter(tokenize(p.search_text)) for p in passages]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 6, priority: dict | None = None) -> list[Hit]:
        q = tokenize(query, expand=True)
        if not q:
            return []
        hits = []
        for p, d, dl in zip(self.passages, self.docs, self.lens):
            s = 0.0
            for t in set(q):
                tf = d.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg))
            if s > 0:
                # Small tie-breaker toward the policy of record (FAQ > guide > tickets).
                s *= 1 + 0.05 * (priority or {}).get(p.source, 0)
                hits.append(Hit(p, s))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


class LiveIndex:
    """Rebuilds when any source file or the rules file changes (content live on the next question)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sig = None
        self.index: Index | None = None
        self.errors: list[str] = []
        self.rules: dict = {}

    @staticmethod
    def _signature():
        sig = []
        for p in (config.FAQ_PATH, config.TICKETS_PATH, config.GUIDE_PATH, config.RULES_PATH):
            try:
                st = p.stat()
                sig.append((str(p), st.st_mtime_ns, st.st_size))
            except FileNotFoundError:
                sig.append((str(p), None, None))
        return tuple(sig)

    def get(self) -> Index:
        sig = self._signature()
        with self._lock:
            if sig != self._sig or self.index is None:
                self.rules = config.load_rules()
                passages, self.errors = load_all(self.rules)
                self.index = Index(passages)
                self._sig = sig
            return self.index

    def search(self, query: str) -> list[Hit]:
        idx = self.get()
        return idx.search(query, k=self.rules.get("max_passages", 6), priority=self.rules.get("source_priority"))


live_index = LiveIndex()
