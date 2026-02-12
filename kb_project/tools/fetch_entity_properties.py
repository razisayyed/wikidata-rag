"""LangChain tool implementations for Wikidata entity property fetching."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from ..utils.logging import (
    configure_logging,
    log_tool_usage,
)
from ..wikidata.properties import WIKIDATA_PROPERTIES
from ..wikidata.sparql import run_sparql as _run_sparql

logger = configure_logging()


class FetchPropertiesInput(BaseModel):
    """Input for fetching properties by QID."""

    qid: str = Field(
        description="The Wikidata QID of the entity (e.g., 'Q142' for France)"
    )
    properties: List[str] = Field(
        description="List of Wikidata property IDs to fetch (e.g., ['P569', 'P106'])"
    )


@tool("fetch_entity_properties", args_schema=FetchPropertiesInput)
def fetch_entity_properties(qid: str, properties: List[str]) -> str:
    """
    Fetch specific properties for a Wikidata entity by its QID.

    Use this after selecting an entity from search_entity_candidates.

    Returns structured property data from Wikidata.
    """

    qid = qid.strip().upper()
    qid_pattern = re.compile(r"^Q\d+$")

    if not qid_pattern.match(qid):
        return f"Error: Invalid QID '{qid}'. Must be 'Q' followed by digits (e.g., 'Q142')."

    # Handle nested property lists
    processed_properties = []
    for p in properties:
        if isinstance(p, list):
            if len(p) > 0:
                processed_properties.append(str(p[0]))
        else:
            processed_properties.append(str(p))

    valid_props = [
        p.strip().upper()
        for p in processed_properties
        if p.strip().upper() in WIKIDATA_PROPERTIES
    ]

    if not valid_props:
        return "Error: No valid properties specified."

    query = build_dynamic_sparql_query(qid, valid_props)

    if not query:
        return f"Error: Could not build query for {qid}"

    try:
        result = _run_sparql(query)
        bindings = result.get("results", {}).get("bindings", [])

        if not bindings:
            return f"Error: Entity {qid} not found or has no data for requested properties."

        formatted_results, wikipedia_url = format_property_results(
            bindings, valid_props, qid
        )

        log_tool_usage(
            "fetch_entity_properties",
            {"qid": qid, "properties": properties},
            formatted_results,
        )

        return formatted_results

    except Exception as e:
        return f"Error fetching properties for {qid}: {e}"


def build_dynamic_sparql_query(qid: str, property_ids: List[str]) -> str:
    """Build a SPARQL query that fetches only the specified properties."""

    valid_props = [
        p.strip().upper()
        for p in property_ids
        if p.strip().upper() in WIKIDATA_PROPERTIES
    ]

    if not valid_props:
        return ""

    select_vars = ["?itemLabel", "?itemDescription", "?wikipediaUrl"]
    optional_clauses = []

    for prop in valid_props:
        var = prop.lower()
        select_vars.append(f"?{var}Label")
        optional_clauses.append(f"  OPTIONAL {{ ?item wdt:{prop} ?{var} . }}")

    query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>

SELECT {' '.join(select_vars)} WHERE {{
  BIND(wd:{qid} AS ?item)

{chr(10).join(optional_clauses)}

  OPTIONAL {{
    ?wikipediaUrl schema:about ?item ;
                  schema:isPartOf <https://en.wikipedia.org/> .
  }}

  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en".
  }}
}}
LIMIT 100"""

    return query


def format_property_results(
    bindings: List[Dict[str, Any]], valid_props: List[str], qid: str = ""
) -> tuple[str, Optional[str]]:
    """Format SPARQL results into a readable string."""

    if not bindings:
        return "No data found.", None

    collected: Dict[str, List[str]] = {p: [] for p in valid_props}
    entity_label = None
    entity_desc = None
    wikipedia_url = None

    for b in bindings:
        if not entity_label:
            entity_label = b.get("itemLabel", {}).get("value")
        if not entity_desc:
            entity_desc = b.get("itemDescription", {}).get("value")
        if not wikipedia_url:
            wikipedia_url = b.get("wikipediaUrl", {}).get("value")

        for prop in valid_props:
            var = prop.lower()
            val_label = b.get(f"{var}Label", {}).get("value")
            if val_label and val_label not in collected[prop]:
                collected[prop].append(val_label)

    lines = []
    if entity_label and qid:
        lines.append(f"Entity: {entity_label}")
        lines.append(f"QID: {qid}")
    elif entity_label:
        lines.append(f"Entity: {entity_label}")
    if entity_desc:
        lines.append(f"Description: {entity_desc}")
    lines.append("")

    for prop in valid_props:
        prop_name = WIKIDATA_PROPERTIES.get(prop, prop)
        label = f"{prop}: {prop_name}"
        values = collected[prop][:5]
        if values:
            if len(values) == 1:
                lines.append(f"{label} — {values[0]}")
            else:
                lines.append(f"{label}:")
                for v in values:
                    lines.append(f"  - {v}")
        else:
            lines.append(f"{label}: (not available)")

    return "\n".join(lines), wikipedia_url
