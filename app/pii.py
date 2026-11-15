"""Mask personal data before it is logged or sent to the model (PRD NFR: no raw PII in logs)."""
import re

_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d\b")),
    ("ACCOUNT", re.compile(r"\b(?:acc(?:ount)?|acct)\s*(?:no\.?|number|#)?\s*[:#]?\s*[A-Z0-9-]{5,}\b", re.I)),
    ("SIM_SERIAL", re.compile(r"\b89\d{17,20}\b")),
]

# Customer-safe service numbers that look like digits but are not personal data.
_ALLOW = {"611", "*123#", "7726"}


def mask(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, kinds_found)."""
    found: list[str] = []
    # SIM serial before card, since ICCIDs also match the card pattern.
    ordered = [_PATTERNS[4]] + _PATTERNS[:4]
    for kind, pat in ordered:
        def _sub(m, kind=kind):
            if m.group(0).strip() in _ALLOW:
                return m.group(0)
            found.append(kind)
            return f"[{kind}]"
        text = pat.sub(_sub, text)
    return text, sorted(set(found))
