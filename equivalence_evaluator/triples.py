"""Triple extraction and matching utilities for equivalence checks."""

from __future__ import annotations

import os
from typing import Any

from .config import TEMPERATURE, TRIPLE_LLM_MODEL, TRIPLE_OLLAMA_MODEL
from .utils import normalize_triple_token, safe_json_extract

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore[assignment]
    HumanMessage = None  # type: ignore[assignment]
    SystemMessage = None  # type: ignore[assignment]

from kb_project.settings import get_ollama_connection_kwargs
from kb_project.utils.imports import ChatOllama

TRIPLE_PROMPT_TEMPLATE = """Extract factual triples from the sentence.
Return JSON list of (subject, relation, object).
Only include factual relations.
No explanation.

Sentence:
{text}"""


def _normalize_triple(item: tuple[str, str, str]) -> tuple[str, str, str]:
    return (
        normalize_triple_token(item[0]),
        normalize_triple_token(item[1]),
        normalize_triple_token(item[2]),
    )


def _parse_triple_payload(parsed: Any) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    if not isinstance(parsed, list):
        return triples
    for row in parsed:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            triples.append((str(row[0]), str(row[1]), str(row[2])))
            continue
        if isinstance(row, dict):
            subject = row.get("subject")
            relation = row.get("relation")
            obj = row.get("object")
            if subject is None or relation is None or obj is None:
                continue
            triples.append((str(subject), str(relation), str(obj)))
    return [_normalize_triple(item) for item in triples if any(part.strip() for part in item)]


def _extract_with_openai(text: str) -> list[tuple[str, str, str]]:
    if ChatOpenAI is None or HumanMessage is None or SystemMessage is None:
        raise RuntimeError("langchain_openai not available")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    llm = ChatOpenAI(model=TRIPLE_LLM_MODEL, temperature=TEMPERATURE)
    response = llm.invoke(
        [
            SystemMessage(content="You extract factual triples and return strict JSON only."),
            HumanMessage(content=TRIPLE_PROMPT_TEMPLATE.format(text=text)),
        ]
    )
    parsed = safe_json_extract(str(response.content))
    triples = _parse_triple_payload(parsed)
    if not triples and str(text).strip():
        raise RuntimeError("OpenAI triple extraction returned empty/non-JSON")
    return triples


def _extract_with_ollama(text: str) -> list[tuple[str, str, str]]:
    if ChatOllama is None:
        raise RuntimeError("ChatOllama unavailable")
    llm = ChatOllama(
        model=TRIPLE_OLLAMA_MODEL,
        temperature=TEMPERATURE,
        **get_ollama_connection_kwargs(),
    )
    response = llm.invoke(TRIPLE_PROMPT_TEMPLATE.format(text=text))
    content = getattr(response, "content", response)
    parsed = safe_json_extract(str(content))
    triples = _parse_triple_payload(parsed)
    if not triples and str(text).strip():
        raise RuntimeError("Ollama triple extraction returned empty/non-JSON")
    return triples


def extract_triples_with_details(text: str, verbose: bool = False) -> dict[str, Any]:
    """Extract triples with backend/error details."""
    openai_error = ""
    try:
        triples = _extract_with_openai(text)
        return {
            "triples": triples,
            "backend": "openai",
            "error": "",
        }
    except Exception as exc:
        openai_error = str(exc)
        if verbose:
            print(f"[Equivalence] Triple extraction OpenAI failed: {openai_error}")

    ollama_error = ""
    try:
        triples = _extract_with_ollama(text)
        return {
            "triples": triples,
            "backend": "ollama",
            "error": "",
        }
    except Exception as exc:
        ollama_error = str(exc)
        if verbose:
            print(f"[Equivalence] Triple extraction Ollama failed: {ollama_error}")

    return {
        "triples": [],
        "backend": "none",
        "error": f"openai_error={openai_error}; ollama_error={ollama_error}",
    }


def extract_triples(text: str, verbose: bool = False) -> list[tuple[str, str, str]]:
    """Extract normalized triples from text."""
    return extract_triples_with_details(text=text, verbose=verbose).get("triples", [])


def compare_triples(ground_truth: str, rag_output: str, verbose: bool = False) -> dict[str, Any]:
    """Compare triple sets and return overlap diagnostics."""
    gt = extract_triples_with_details(ground_truth, verbose=verbose)
    rag = extract_triples_with_details(rag_output, verbose=verbose)

    gt_set = set(gt["triples"])
    rag_set = set(rag["triples"])
    matched = sorted(gt_set & rag_set)
    identical = gt_set == rag_set and len(gt_set) > 0
    has_overlap = len(matched) > 0
    overlap_ratio = 0.0
    if gt_set or rag_set:
        overlap_ratio = len(matched) / max(1, len(gt_set | rag_set))

    return {
        "equivalent": bool(identical or has_overlap),
        "gt_triples": sorted(gt_set),
        "rag_triples": sorted(rag_set),
        "matched": matched,
        "overlap_ratio": float(overlap_ratio),
        "identical": identical,
        "gt_backend": gt["backend"],
        "rag_backend": rag["backend"],
        "error": "; ".join([value for value in [gt.get("error", ""), rag.get("error", "")] if value]),
    }

