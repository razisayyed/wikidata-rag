"""Configuration for semantic equivalence evaluation."""

from __future__ import annotations

import os

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.92"))
NLI_CONFIDENCE_THRESHOLD = float(os.getenv("NLI_CONFIDENCE_THRESHOLD", "0.70"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
NLI_MODEL = os.getenv("NLI_MODEL", "microsoft/deberta-v3-large-mnli")
TRIPLE_LLM_MODEL = os.getenv("TRIPLE_LLM_MODEL", "gpt-4o-mini")
TRIPLE_OLLAMA_MODEL = os.getenv(
    "TRIPLE_OLLAMA_MODEL",
    os.getenv("WIKIDATA_RAG_MODEL", "qwen2.5:32b-instruct"),
)
SIMILARITY_FALLBACK_ST_MODEL = os.getenv(
    "SIMILARITY_FALLBACK_ST_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)
TEMPERATURE = 0
