"""LangChain tool implementations for Wikidata SPARQL queries."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain.tools import tool
from pydantic import BaseModel, Field

from ..utils.logging import (
    configure_logging,
    log_tool_usage,
)
from ..settings import DEFAULT_SPARQL_LIMIT
from ..wikidata.sparql import run_sparql as _run_sparql

logger = configure_logging()


class SparqlInput(BaseModel):
    sparql: str = Field(
        ...,
        description="A read-only SELECT SPARQL query as a SINGLE-LINE string.",
    )
    max_rows: int = Field(
        DEFAULT_SPARQL_LIMIT, ge=1, le=100, description="Max rows to return."
    )


@tool("wikidata_sparql", args_schema=SparqlInput)
def wikidata_sparql(sparql: str, max_rows: int = DEFAULT_SPARQL_LIMIT) -> str:
    """
    Run a custom SPARQL query for complex questions.

    Use for aggregations, filtering, or multi-entity relationships.
    Only SELECT queries are allowed.
    """

    normalized = sparql.strip().lower()
    if not normalized.startswith("select"):
        return "Error: Only SELECT queries are allowed."

    log_tool_usage(
        "wikidata_sparql", {"sparql": sparql, "max_rows": max_rows}, "Executing..."
    )

    try:
        results = _run_sparql(sparql)
    except Exception as exc:
        logger.error(f"SPARQL error: {exc}")
        return f"SPARQL error: {exc}"

    rows: List[Dict[str, Any]] = []
    for b in results.get("results", {}).get("bindings", [])[:max_rows]:
        row = {k: v.get("value") for k, v in b.items()}
        rows.append(row)

    if not rows:
        return "Query returned no results."

    result = json.dumps({"rows": rows}, ensure_ascii=False, indent=2)
    log_tool_usage("wikidata_sparql", {"sparql": sparql}, result)
    return result
