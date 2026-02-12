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


def _format_time_value(value: str) -> str:
    """Normalize Wikidata datetime-like values for readability."""
    if not value:
        return value
    if "T" in value and value.endswith("Z"):
        return value.split("T", 1)[0]
    return value


class FetchPropertiesInput(BaseModel):
    """Input for fetching properties by QID."""

    qid: str = Field(
        description="The Wikidata QID of the entity (e.g., 'Q142' for France)"
    )
    properties: List[str] = Field(
        description="List of Wikidata property IDs to fetch (e.g., ['P569', 'P106'])"
    )
    include_qualifiers: bool = Field(
        default=True,
        description=(
            "Whether to include statement qualifiers (for example start/end/point-in-time) "
            "when available."
        ),
    )


@tool("fetch_entity_properties", args_schema=FetchPropertiesInput)
def fetch_entity_properties(
    qid: str,
    properties: List[str],
    include_qualifiers: bool = True,
) -> str:
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

    query = build_dynamic_sparql_query(
        qid=qid,
        property_ids=valid_props,
        include_qualifiers=include_qualifiers,
    )

    if not query:
        return f"Error: Could not build query for {qid}"

    try:
        result = _run_sparql(query)
        bindings = result.get("results", {}).get("bindings", [])

        if not bindings:
            return f"Error: Entity {qid} not found or has no data for requested properties."

        formatted_results, wikipedia_url = format_property_results(
            bindings=bindings,
            valid_props=valid_props,
            qid=qid,
            include_qualifiers=include_qualifiers,
        )

        log_tool_usage(
            "fetch_entity_properties",
            {
                "qid": qid,
                "properties": properties,
                "include_qualifiers": include_qualifiers,
            },
            formatted_results,
        )

        return formatted_results

    except Exception as e:
        return f"Error fetching properties for {qid}: {e}"


def build_dynamic_sparql_query(
    qid: str,
    property_ids: List[str],
    include_qualifiers: bool = True,
) -> str:
    """Build a statement-level SPARQL query for the requested properties."""

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
        statement_var = f"?{var}Statement"
        value_var = f"?{var}Value"
        value_label_var = f"?{var}ValueLabel"

        select_vars.append(value_var)
        select_vars.append(value_label_var)

        qualifier_selects: List[str] = []
        qualifier_optionals: List[str] = []
        if include_qualifiers:
            for qualifier_prop in ("P580", "P582", "P585"):
                qualifier_var = f"?{var}{qualifier_prop}"
                qualifier_selects.append(qualifier_var)
                qualifier_optionals.append(
                    f"    OPTIONAL {{ {statement_var} pq:{qualifier_prop} {qualifier_var} . }}"
                )
        select_vars.extend(qualifier_selects)

        clause = [
            "  OPTIONAL {",
            f"    ?item p:{prop} {statement_var} .",
            f"    {statement_var} ps:{prop} {value_var} .",
            *qualifier_optionals,
            "  }",
        ]
        optional_clauses.append("\n".join(clause))

    query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
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
LIMIT 200"""

    return query


def format_property_results(
    bindings: List[Dict[str, Any]],
    valid_props: List[str],
    qid: str = "",
    include_qualifiers: bool = True,
) -> tuple[str, Optional[str]]:
    """Format SPARQL results into a readable string."""

    if not bindings:
        return "No data found.", None

    collected: Dict[str, List[Dict[str, Any]]] = {p: [] for p in valid_props}
    dedupe_keys: Dict[str, set[tuple[str, str, str, str]]] = {
        p: set() for p in valid_props
    }
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
            value = (
                b.get(f"{var}ValueLabel", {}).get("value")
                or b.get(f"{var}Value", {}).get("value")
            )
            if not value:
                continue
            value = _format_time_value(value)

            start_time = b.get(f"{var}P580", {}).get("value", "")
            end_time = b.get(f"{var}P582", {}).get("value", "")
            point_in_time = b.get(f"{var}P585", {}).get("value", "")

            dedupe_key = (value, start_time, end_time, point_in_time)
            if dedupe_key in dedupe_keys[prop]:
                continue
            dedupe_keys[prop].add(dedupe_key)

            qualifiers: Dict[str, str] = {}
            if include_qualifiers:
                if start_time:
                    qualifiers["P580"] = _format_time_value(start_time)
                if end_time:
                    qualifiers["P582"] = _format_time_value(end_time)
                if point_in_time:
                    qualifiers["P585"] = _format_time_value(point_in_time)

            collected[prop].append({"value": value, "qualifiers": qualifiers})

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
        entries = collected[prop][:5]
        if entries:
            if len(entries) == 1 and not entries[0]["qualifiers"]:
                lines.append(f"{label} — {entries[0]['value']}")
            else:
                lines.append(f"{label}:")
                for entry in entries:
                    value = entry["value"]
                    qualifiers = entry["qualifiers"]
                    if qualifiers:
                        qualifier_parts = []
                        for qid_, qvalue in qualifiers.items():
                            if qid_ == "P580":
                                qualifier_parts.append(f"start: {qvalue}")
                            elif qid_ == "P582":
                                qualifier_parts.append(f"end: {qvalue}")
                            elif qid_ == "P585":
                                qualifier_parts.append(f"time: {qvalue}")
                        lines.append(f"  - {value} ({', '.join(qualifier_parts)})")
                    else:
                        lines.append(f"  - {value}")
        else:
            lines.append(f"{label}: (not available)")

    return "\n".join(lines), wikipedia_url
