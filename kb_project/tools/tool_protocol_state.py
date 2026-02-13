"""Per-run tool protocol state for enforcing call ordering."""

from __future__ import annotations

import re
from threading import Lock
from typing import Dict, Iterable, List, Set

_STATE_LOCK = Lock()
_ALLOWED_QIDS: Set[str] = set()
_QID_TO_ENTITY: Dict[str, str] = {}
_SPARQL_ATTEMPTED = False
_QUESTION_CONTEXT = ""


def _normalize_qid(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    uri_match = re.search(r"/(Q\d+)$", text, flags=re.IGNORECASE)
    if uri_match:
        return uri_match.group(1).upper()
    if text.lower().startswith("wd:"):
        text = text.split(":", 1)[1]
    text = text.upper()
    if text.startswith("Q") and text[1:].isdigit():
        return text
    return ""


def reset_tool_protocol_state() -> None:
    """Reset candidate-derived QID state at the start of each question run."""
    with _STATE_LOCK:
        _ALLOWED_QIDS.clear()
        _QID_TO_ENTITY.clear()
        global _SPARQL_ATTEMPTED
        _SPARQL_ATTEMPTED = False
        global _QUESTION_CONTEXT
        _QUESTION_CONTEXT = ""


def register_search_candidates(
    entity_name: str,
    candidates: Iterable[Dict[str, str]],
) -> List[str]:
    """Register candidate QIDs returned by search_entity_candidates."""
    normalized_entity = (entity_name or "").strip()
    registered: List[str] = []

    with _STATE_LOCK:
        for candidate in candidates:
            qid = _normalize_qid(str(candidate.get("qid", "")))
            if not qid:
                continue
            _ALLOWED_QIDS.add(qid)
            if normalized_entity:
                _QID_TO_ENTITY[qid] = normalized_entity
            registered.append(qid)

    return registered


def is_qid_authorized(qid: str) -> bool:
    """Return whether a QID is authorized by prior search or inferred fetches."""
    normalized = _normalize_qid(qid)
    with _STATE_LOCK:
        return normalized in _ALLOWED_QIDS


def get_authorized_qids(limit: int = 15) -> List[str]:
    """Return a deterministic slice of currently authorized QIDs."""
    with _STATE_LOCK:
        return sorted(_ALLOWED_QIDS)[: max(1, limit)]


def register_inferred_qids(qids: Iterable[str]) -> List[str]:
    """Register additional QIDs inferred from prior property fetch results."""
    added: List[str] = []
    with _STATE_LOCK:
        for raw_qid in qids:
            qid = _normalize_qid(raw_qid)
            if not qid:
                continue
            if qid not in _ALLOWED_QIDS:
                added.append(qid)
            _ALLOWED_QIDS.add(qid)
    return added


def mark_sparql_attempt() -> None:
    """Mark that wikidata_sparql has been attempted in the current run."""
    with _STATE_LOCK:
        global _SPARQL_ATTEMPTED
        _SPARQL_ATTEMPTED = True


def has_sparql_attempt() -> bool:
    """Return whether wikidata_sparql was attempted in the current run."""
    with _STATE_LOCK:
        return _SPARQL_ATTEMPTED


def set_question_context(question: str) -> None:
    """Store the current user question for intent-aware tool behavior."""
    with _STATE_LOCK:
        global _QUESTION_CONTEXT
        _QUESTION_CONTEXT = (question or "").strip()


def get_question_context() -> str:
    """Return the current user question for this run."""
    with _STATE_LOCK:
        return _QUESTION_CONTEXT
