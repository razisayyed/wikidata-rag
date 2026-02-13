"""Semantic similarity helpers with deterministic provider fallbacks."""

from __future__ import annotations

import math
import os
from typing import Any

from .config import EMBEDDING_MODEL, SIMILARITY_FALLBACK_ST_MODEL
from .utils import normalize_text

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    OpenAIEmbeddings = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

_EMBEDDING_CACHE: dict[tuple[str, str, str], list[float]] = {}
_OPENAI_CLIENT: Any = None
_ST_CLIENT: Any = None


def _get_openai_client() -> Any:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    if OpenAIEmbeddings is None:
        raise RuntimeError("langchain_openai not available for OpenAI embeddings")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    _OPENAI_CLIENT = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _OPENAI_CLIENT


def _get_sentence_transformer_client() -> Any:
    global _ST_CLIENT
    if _ST_CLIENT is not None:
        return _ST_CLIENT
    if SentenceTransformer is None:
        raise RuntimeError("sentence_transformers not available")
    _ST_CLIENT = SentenceTransformer(SIMILARITY_FALLBACK_ST_MODEL)
    return _ST_CLIENT


def _get_openai_embedding(text: str) -> list[float]:
    normalized = normalize_text(text)
    cache_key = ("openai", EMBEDDING_MODEL, normalized)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]
    client = _get_openai_client()
    embedding = client.embed_query(normalized)
    values = [float(value) for value in embedding]
    _EMBEDDING_CACHE[cache_key] = values
    return values


def _get_st_embedding(text: str) -> list[float]:
    normalized = normalize_text(text)
    cache_key = ("sentence_transformers", SIMILARITY_FALLBACK_ST_MODEL, normalized)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]
    model = _get_sentence_transformer_client()
    embedding = model.encode(normalized)
    values = [float(value) for value in embedding]
    _EMBEDDING_CACHE[cache_key] = values
    return values


def get_embedding_with_details(text: str) -> dict[str, Any]:
    """Resolve embedding with provider details and explicit failure reason."""
    normalized = normalize_text(text)
    if not normalized:
        return {
            "embedding": [0.0],
            "provider": "constant",
            "model": "constant-empty",
            "error": "",
        }

    openai_error = ""
    try:
        return {
            "embedding": _get_openai_embedding(normalized),
            "provider": "openai",
            "model": EMBEDDING_MODEL,
            "error": "",
        }
    except Exception as exc:
        openai_error = str(exc)

    st_error = ""
    try:
        return {
            "embedding": _get_st_embedding(normalized),
            "provider": "sentence_transformers",
            "model": SIMILARITY_FALLBACK_ST_MODEL,
            "error": "",
        }
    except Exception as exc:
        st_error = str(exc)

    return {
        "embedding": None,
        "provider": None,
        "model": None,
        "error": f"openai={openai_error}; sentence_transformers={st_error}",
    }


def get_embedding(text: str) -> list[float]:
    """Get an embedding for text using configured provider order."""
    result = get_embedding_with_details(text)
    embedding = result.get("embedding")
    if embedding is None:
        raise RuntimeError(result.get("error", "embedding provider unavailable"))
    return [float(value) for value in embedding]


def _cosine(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    if len(vec1) != len(vec2):
        limit = min(len(vec1), len(vec2))
        vec1 = vec1[:limit]
        vec2 = vec2[:limit]

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(norm1) * math.sqrt(norm2)))


def cosine_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity for two texts."""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return _cosine(emb1, emb2)


def cosine_similarity_with_details(text1: str, text2: str) -> dict[str, Any]:
    """Cosine similarity with provider/error details for diagnostics."""
    emb1 = get_embedding_with_details(text1)
    if emb1.get("embedding") is None:
        return {
            "score": 0.0,
            "provider": "",
            "model": "",
            "error": f"text1_embedding_error: {emb1.get('error', '')}",
        }

    emb2 = get_embedding_with_details(text2)
    if emb2.get("embedding") is None:
        return {
            "score": 0.0,
            "provider": "",
            "model": "",
            "error": f"text2_embedding_error: {emb2.get('error', '')}",
        }

    score = _cosine(
        [float(v) for v in emb1["embedding"]],
        [float(v) for v in emb2["embedding"]],
    )
    provider = emb1.get("provider") or emb2.get("provider") or ""
    model = emb1.get("model") or emb2.get("model") or ""
    return {
        "score": score,
        "provider": provider,
        "model": model,
        "error": "",
    }

