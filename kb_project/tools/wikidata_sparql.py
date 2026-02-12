"""LangChain tool implementations for Wikidata SPARQL queries."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain.tools import tool
from pydantic import BaseModel, Field

from ..utils.logging import (
    configure_logging,
    log_tool_usage,
)
from ..settings import DEFAULT_SPARQL_LIMIT
from .tool_protocol_state import mark_sparql_attempt
from ..wikidata.sparql import run_sparql as _run_sparql

logger = configure_logging()
MAX_SPARQL_ROWS = 100

_PREFIX_AND_SELECT_PATTERN = re.compile(
    r"^\s*(?:PREFIX\s+[^\n]+\s+)*SELECT\b",
    re.IGNORECASE | re.DOTALL,
)
_BLOCKED_KEYWORDS_PATTERN = re.compile(
    r"\b(INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|MOVE|COPY|ADD)\b",
    re.IGNORECASE,
)


def is_safe_read_only_select(sparql: str) -> tuple[bool, str]:
    """Validate that query is a read-only SELECT query."""
    query = (sparql or "").strip()
    if not query:
        return False, "Error: Empty SPARQL query."

    if _BLOCKED_KEYWORDS_PATTERN.search(query):
        return False, "Error: SPARQL update/mutation keywords are not allowed."

    if not _PREFIX_AND_SELECT_PATTERN.match(query):
        return False, "Error: Only read-only SELECT queries are allowed."

    return True, ""


class SparqlInput(BaseModel):
    sparql: str = Field(
        ...,
        description="A read-only SELECT SPARQL query as a SINGLE-LINE string.",
    )
    max_rows: int = Field(
        DEFAULT_SPARQL_LIMIT,
        ge=1,
        description=(
            "Requested max rows to return. Values above the safety cap are accepted "
            "and clipped internally."
        ),
    )


@tool("wikidata_sparql", args_schema=SparqlInput)
def wikidata_sparql(sparql: str, max_rows: int = DEFAULT_SPARQL_LIMIT) -> str:
    """
    Run a custom SPARQL query for complex questions.

    Prefer this before Wikipedia fallback for factual retrieval when
    fetch_entity_properties is insufficient.
    Use for aggregations, filtering, or multi-entity relationships.
    Only SELECT queries are allowed.
    """

    is_valid, validation_error = is_safe_read_only_select(sparql)
    if not is_valid:
        return validation_error
    mark_sparql_attempt()

    effective_max_rows = min(int(max_rows), MAX_SPARQL_ROWS)

    log_tool_usage(
        "wikidata_sparql",
        {
            "sparql": sparql,
            "max_rows_requested": max_rows,
            "max_rows_effective": effective_max_rows,
        },
        "Executing...",
    )

    try:
        results = _run_sparql(sparql)
    except Exception as exc:
        logger.error(f"SPARQL error: {exc}")
        return f"SPARQL error: {exc}"

    rows: List[Dict[str, Any]] = []
    for b in results.get("results", {}).get("bindings", [])[:effective_max_rows]:
        row = {k: v.get("value") for k, v in b.items()}
        rows.append(row)

    if not rows:
        return "Query returned no results."

    result = json.dumps({"rows": rows}, ensure_ascii=False, indent=2)
    log_tool_usage("wikidata_sparql", {"sparql": sparql}, result)
    return result
