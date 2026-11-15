"""Paths and runtime settings. Content rules live in data/content_rules.json so ops can edit them."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=value lines from .env so the API key never has to be typed into a shell or chat.
    Variables already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


if not os.environ.get("NOVACELL_SKIP_DOTENV"):
    _load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.environ.get("NOVACELL_DATA_DIR", ROOT / "data"))
RUNTIME_DIR = Path(os.environ.get("NOVACELL_RUNTIME_DIR", ROOT / "runtime"))
VERSIONS_DIR = DATA_DIR / "versions"

FAQ_PATH = DATA_DIR / "faq.csv"
TICKETS_PATH = DATA_DIR / "tickets.db"
GUIDE_PATH = DATA_DIR / "telecom_guide.pdf"
RULES_PATH = DATA_DIR / "content_rules.json"

# Provider is chosen in app/llm.py: Groq when GROQ_API_KEY is set, else Anthropic. Override with NOVACELL_PROVIDER.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low")
MODEL = os.environ.get("NOVACELL_MODEL", "claude-opus-5-5")
# Chat answers are short and latency-sensitive (PRD: p95 first token <= 2 s), so default to low effort.
EFFORT = os.environ.get("NOVACELL_EFFORT", "low")


def load_rules() -> dict:
    with open(RULES_PATH, encoding="utf-8") as f:
        return json.load(f)
