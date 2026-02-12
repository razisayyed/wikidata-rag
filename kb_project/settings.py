"""Global settings for the KB Project."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Keep working if python-dotenv is not installed.
    pass

# ==========================================================================
# LLM Configuration (shared across agents)
# ==========================================================================

def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(default, minimum)
    try:
        value = int(raw)
    except ValueError:
        return max(default, minimum)
    return max(value, minimum)


WIKIDATA_RAG_MODEL = _env("WIKIDATA_RAG_MODEL", _env("LLM_MODEL", "qwen2.5:32b-instruct"))
PROMPT_ONLY_MODEL = _env("PROMPT_ONLY_MODEL", WIKIDATA_RAG_MODEL)
RAGTRUTH_MODEL = _env("RAGTRUTH_MODEL", WIKIDATA_RAG_MODEL)
OPENAI_JUDGE_MODEL = _env("OPENAI_JUDGE_MODEL", "gpt-4o")
VECTARA_DEVICE = _env("VECTARA_DEVICE", "auto").lower()
AIMON_DEVICE = _env("AIMON_DEVICE", "auto").lower()
RAG_RECURSION_LIMIT = _env_int("RAG_RECURSION_LIMIT", 40, minimum=1)

# Backward-compatible alias used across the codebase.
LLM_MODEL = WIKIDATA_RAG_MODEL
DEFAULT_TEMPERATURE = 0.1

# ==========================================================================
# Environment-based runtime options
# ==========================================================================

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "").strip()
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "").strip()
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()


def get_ollama_connection_kwargs() -> Dict[str, Any]:
    """
    Build ChatOllama connection kwargs from environment variables.

    Supported vars:
    - OLLAMA_HOST: full host URL (e.g., http://localhost:11434)
    - OLLAMA_PORT: fallback port when OLLAMA_HOST is not set
    - OLLAMA_API_KEY: adds Authorization: Bearer <token> header
    """
    kwargs: Dict[str, Any] = {}

    base_url = OLLAMA_HOST
    if not base_url and OLLAMA_PORT:
        base_url = f"http://localhost:{OLLAMA_PORT}"
    elif base_url and not (
        base_url.startswith("http://") or base_url.startswith("https://")
    ):
        # Normalize host values like "1.2.3.4:11434" to a valid URL.
        base_url = f"http://{base_url}"

    if base_url:
        kwargs["base_url"] = base_url

    if OLLAMA_API_KEY:
        kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        }

    return kwargs


def resolve_device(device_preference: str = "auto") -> str:
    """
    Resolve runtime device from preference.

    Supported values: auto, cpu, cuda, mps.
    """
    pref = (device_preference or "auto").strip().lower()
    if pref not in {"auto", "cpu", "cuda", "mps"}:
        pref = "auto"

    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"

    if pref == "cpu":
        return "cpu"

    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if pref == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            return "mps"
        return "cpu"

    # auto
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"
    return "cpu"


# ==========================================================================
# Wikidata / Wikipedia configuration
# ==========================================================================

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_USER_AGENT = "WikidataLangChainAgent/1.0 (research-demo)"

WIKIPEDIA_USER_AGENT = "WikidataLangChainAgent/1.0 (research-demo)"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1/page/html/"

MAX_SEARCH_RESULTS = 10  # Max candidates to return for LLM selection
MAX_ARTICLE_CHARS = 8000  # Max Wikipedia article chars to return
DEFAULT_SPARQL_LIMIT = 25  # Default max rows for SPARQL queries

# ==========================================================================
# Logging configuration
# ==========================================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = (
    LOG_DIR / f"wikidata_rag_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
