"""LangSmith tracing controls for benchmark execution scopes."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

try:
    from langsmith import tracing_context as _tracing_context
except Exception:
    try:
        from langsmith.run_helpers import tracing_context as _tracing_context
    except Exception:
        _tracing_context = None

TRACE_MODE_RAG = "rag"
TRACE_MODE_RAG_BASELINE = "rag_baseline"
TRACE_MODE_ALL = "all"

VALID_TRACE_MODES = {
    TRACE_MODE_RAG,
    TRACE_MODE_RAG_BASELINE,
    TRACE_MODE_ALL,
}


def normalize_trace_mode(value: str | None) -> str:
    """Normalize user/env trace mode to a supported value."""
    raw = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"rag+baseline", "rag_and_baseline", "baseline"}:
        raw = TRACE_MODE_RAG_BASELINE
    if raw not in VALID_TRACE_MODES:
        return TRACE_MODE_RAG
    return raw


def should_trace_component(mode: str, component: str) -> bool:
    """
    Decide whether a benchmark component should be traced.

    Components:
    - rag: RAG model calls only
    - baseline: prompt-only baseline model calls
    - evaluator: evaluator model calls (LLM judge, RAGTruth, etc.)
    """
    normalized_mode = normalize_trace_mode(mode)
    normalized_component = (component or "").strip().lower()

    if normalized_mode == TRACE_MODE_ALL:
        return True
    if normalized_mode == TRACE_MODE_RAG_BASELINE:
        return normalized_component in {"rag", "baseline"}
    return normalized_component == "rag"


@contextmanager
def langsmith_tracing(enabled: bool) -> Iterator[None]:
    """
    Temporarily toggle LangSmith/LangChain tracing for this call scope.

    Uses LangSmith runtime context when available, with env fallback for
    integrations that rely on process env flags.
    """
    keys = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
    previous = {key: os.environ.get(key) for key in keys}
    value = "true" if enabled else "false"
    for key in keys:
        os.environ[key] = value
    try:
        if _tracing_context is not None:
            with _tracing_context(enabled=enabled):
                yield
        else:
            yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
