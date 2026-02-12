"""Per-run tool protocol state for enforcing call ordering."""

from __future__ import annotations

from threading import Lock
from typing import Dict, Iterable, List, Set

_STATE_LOCK = Lock()
_ALLOWED_QIDS: Set[str] = set()
_QID_TO_ENTITY: Dict[str, str] = {}
_SPARQL_ATTEMPTED = False


def reset_tool_protocol_state() -> None:
    """Reset candidate-derived QID state at the start of each question run."""
    with _STATE_LOCK:
        _ALLOWED_QIDS.clear()
        _QID_TO_ENTITY.clear()
        global _SPARQL_ATTEMPTED
        _SPARQL_ATTEMPTED = False


def register_search_candidates(
    entity_name: str,
    candidates: Iterable[Dict[str, str]],
) -> List[str]:
    """Register candidate QIDs returned by search_entity_candidates."""
    normalized_entity = (entity_name or "").strip()
    registered: List[str] = []

    with _STATE_LOCK:
        for candidate in candidates:
            qid = str(candidate.get("qid", "")).strip().upper()
            if not qid.startswith("Q") or len(qid) < 2 or not qid[1:].isdigit():
                continue
            _ALLOWED_QIDS.add(qid)
            if normalized_entity:
                _QID_TO_ENTITY[qid] = normalized_entity
            registered.append(qid)

    return registered


def is_qid_authorized(qid: str) -> bool:
    """Return whether a QID is authorized by prior candidate search."""
    normalized = (qid or "").strip().upper()
    with _STATE_LOCK:
        return normalized in _ALLOWED_QIDS


def get_authorized_qids(limit: int = 15) -> List[str]:
    """Return a deterministic slice of currently authorized QIDs."""
    with _STATE_LOCK:
        return sorted(_ALLOWED_QIDS)[: max(1, limit)]


def mark_sparql_attempt() -> None:
    """Mark that wikidata_sparql has been attempted in the current run."""
    with _STATE_LOCK:
        global _SPARQL_ATTEMPTED
        _SPARQL_ATTEMPTED = True


def has_sparql_attempt() -> bool:
    """Return whether wikidata_sparql was attempted in the current run."""
    with _STATE_LOCK:
        return _SPARQL_ATTEMPTED
