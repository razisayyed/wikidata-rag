"""Global settings for the KB Project."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# ==========================================================================
# LLM Configuration (shared across agents)
# ==========================================================================

LLM_MODEL = "gpt-oss:120b-cloud"
DEFAULT_TEMPERATURE = 0.1

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
LOG_FILE = LOG_DIR / f"wikidata_rag_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
