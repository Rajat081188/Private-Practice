import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests run against a private copy of the data so uploads/rollbacks never touch real content.
_tmp = Path(tempfile.mkdtemp(prefix="novacell-test-"))
shutil.copytree(ROOT / "data", _tmp / "data", ignore=shutil.ignore_patterns("versions", ".staging"))
os.environ["NOVACELL_DATA_DIR"] = str(_tmp / "data")
os.environ["NOVACELL_RUNTIME_DIR"] = str(_tmp / "runtime")
os.environ["NOVACELL_SKIP_DOTENV"] = "1"  # tests never use a real key
for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "GROQ_API_KEY", "NOVACELL_PROVIDER"):
    os.environ.pop(var, None)
