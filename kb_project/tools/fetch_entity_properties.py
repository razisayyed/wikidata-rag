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
from .tool_protocol_state import (
    get_authorized_qids,
    get_question_context,
    is_qid_authorized,
    register_inferred_qids,
)
from ..wikidata.properties import WIKIDATA_PROPERTIES
from ..wikidata.sparql import run_sparql as _run_sparql

logger = configure_logging()
_PROPERTY_ID_PATTERN = re.compile(r"^P\d+$")
_VALUE_QID_PATTERN = re.compile(r"\[(Q\d+)\]")

_INTENT_PROPERTY_RULES: List[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("discover", "discovered", "discoverer", "invent", "invented", "inventor"), ("P61", "P575")),
    (("born", "birth", "date of birth"), ("P569",)),
    (("born in", "birthplace", "place of birth"), ("P19", "P131", "P17")),
    (("died", "death", "date of death"), ("P570",)),
    (("citizenship", "nationality"), ("P27",)),
    (
        (
            "author's country",
            "authors country",
            "author country",
            "author's citizenship",
            "author citizenship",
            "author of",
            "country of the author",
        ),
        ("P27", "P17", "P495"),
    ),
    (("founder", "founded by"), ("P112", "P571")),
    (("head of state",), ("P35", "P1906")),
    (("head of government",), ("P6", "P1313")),
    (("ceo", "chief executive officer"), ("P169", "P488")),
    (
        ("when", "year", "date", "during", "at the time", "currently", "current", "former", "historical"),
        ("P580", "P582", "P585", "P1319", "P1326", "P2031", "P2032", "P2669"),
    ),
    (("continent",), ("P30",)),
    (("country",), ("P17", "P131")),
    (("award", "medal", "order", "knight", "honor"), ("P166", "P1027", "P17", "P495")),
    (
        ("major achievements", "achievements", "accomplishments", "contribution", "contributions", "known for"),
        ("P101", "P800", "P166", "P39", "P108", "P61"),
    ),
    (("capital",), ("P36", "P1376")),
    (("wrote", "author", "novel", "book"), ("P50", "P577")),
    (("painted", "painter", "painted by"), ("P170",)),
    (("formula", "chemical symbol"), ("P274",)),
    (("boiling point", "boils"), ("P2102",)),
    (("speed of light", "speed"), ("P2052",)),
    (("work for", "worked for", "employer", "organization"), ("P108", "P39")),
]
_BASELINE_PROPERTY_GUARDRAIL: tuple[str, ...] = ("P31",)


def _normalize_property_id(raw: Any) -> str:
    prop_id = str(raw or "").strip().upper()
    if not _PROPERTY_ID_PATTERN.match(prop_id):
        return ""
    return prop_id


def _format_time_value(value: str) -> str:
    """Normalize Wikidata datetime-like values for readability."""
    if not value:
        return value
    if "T" in value and value.endswith("Z"):
        return value.split("T", 1)[0]
    return value


def _extract_year(value: str) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"\b(\d{4})\b", value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _entry_sort_key(entry: Dict[str, Any]) -> tuple[int, int]:
    qualifiers = entry.get("qualifiers", {}) or {}
    end_year = _extract_year(str(qualifiers.get("P582", "")))
    start_year = _extract_year(str(qualifiers.get("P580", "")))
    current_flag = 1 if end_year is None else 0
    recency = start_year if start_year is not None else -1
    return (current_flag, recency)


def _normalize_temporal_alias(
    prop: str,
    value: str,
    qualifiers: Dict[str, str],
) -> tuple[str, Optional[str]]:
    """
    Normalize known historical naming aliases using qualifier time windows.

    Example:
      P108 employer "Government Communications Headquarters" with WWII qualifiers
      is normalized to the wartime label "Government Code and Cypher School (GC&CS)".
    """
    lowered = (value or "").strip().lower()
    if prop != "P108":
        return value, None

    if lowered not in {
        "government communications headquarters",
        "gchq",
    }:
        return value, None

    start_year = _extract_year(qualifiers.get("P580", ""))
    end_year = _extract_year(qualifiers.get("P582", ""))
    point_year = _extract_year(qualifiers.get("P585", ""))

    is_wwii_window = False
    if start_year is not None and start_year <= 1945:
        is_wwii_window = True
    if end_year is not None and 1939 <= end_year <= 1946:
        is_wwii_window = True
    if point_year is not None and 1939 <= point_year <= 1945:
        is_wwii_window = True

    if not is_wwii_window:
        return value, None

    normalized = "Government Code and Cypher School (GC&CS)"
    note = (
        "historical alias normalization: "
        "Government Communications Headquarters/GCHQ -> Government Code and Cypher School (GC&CS)"
    )
    return normalized, note


def _qid_from_entity_value(value: str) -> str:
    raw = (value or "").strip()
    if raw.lower().startswith("http://www.wikidata.org/entity/q") or raw.lower().startswith(
        "https://www.wikidata.org/entity/q"
    ):
        qid = raw.rsplit("/", 1)[-1].upper()
        if qid.startswith("Q") and qid[1:].isdigit():
            return qid
    return ""


def _augment_properties_for_question(
    properties: List[str],
    question: str,
) -> tuple[List[str], List[str]]:
    """
    Add deterministic intent-critical properties from question keywords.

    This reduces failures when the LLM under-selects required properties.
    """
    normalized_question = (question or "").strip().lower()
    merged: List[str] = []
    seen = set()

    def _add(prop: str) -> None:
        prop_id = _normalize_property_id(prop)
        if not prop_id or prop_id in seen:
            return
        seen.add(prop_id)
        merged.append(prop_id)

    for prop in properties:
        _add(prop)

    for prop in _BASELINE_PROPERTY_GUARDRAIL:
        _add(prop)

    auto_added: List[str] = []
    if normalized_question:
        for keywords, required_props in _INTENT_PROPERTY_RULES:
            if not any(keyword in normalized_question for keyword in keywords):
                continue
            for prop in required_props:
                before = len(merged)
                _add(prop)
                if len(merged) > before:
                    auto_added.append(prop)

    return merged, auto_added


def _extract_qids_from_bindings(
    bindings: List[Dict[str, Any]],
    source_properties: List[str],
) -> List[str]:
    """
    Extract Wikidata entity QIDs from raw SPARQL binding values.

    This is used to authorize follow-up fetches for related entities discovered
    during an earlier fetch_entity_properties call.
    """
    found: set[str] = set()
    # Follow-up fetches should come from relation-valued properties, not
    # ontology/type guardrails (P31/P279), which often cause class drift.
    allowed_value_keys = {
        f"{prop.lower()}Value"
        for prop in source_properties
        if prop not in {"P31", "P279"}
    }
    for row in bindings:
        for key, binding in row.items():
            if key not in allowed_value_keys:
                continue
            if not isinstance(binding, dict):
                continue
            value = str(binding.get("value", "")).strip()
            if not value:
                continue
            if value.lower().startswith("http://www.wikidata.org/entity/q"):
                qid = value.rsplit("/", 1)[-1].upper()
            elif value.lower().startswith("https://www.wikidata.org/entity/q"):
                qid = value.rsplit("/", 1)[-1].upper()
            elif value.lower().startswith("wd:q"):
                qid = value.split(":", 1)[-1].upper()
            else:
                continue
            if qid.startswith("Q") and qid[1:].isdigit():
                found.add(qid)
    return sorted(found)


def _extract_qids_from_property_bindings(bindings: List[Dict[str, Any]]) -> List[str]:
    """Extract QIDs from per-property query bindings."""
    found: set[str] = set()
    for row in bindings:
        value_binding = row.get("value")
        if not isinstance(value_binding, dict):
            continue
        value = str(value_binding.get("value", "")).strip()
        if not value:
            continue
        qid = _qid_from_entity_value(value)
        if qid:
            found.add(qid)
    return sorted(found)


class FetchPropertiesInput(BaseModel):
    """Input for fetching properties by QID."""

    qid: str = Field(
        description=(
            "The Wikidata QID of the entity as a literal string "
            "(e.g., 'Q142' for France). Do not pass code/expression syntax."
        ),
        pattern=r"^Q\d+$",
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
    expression_like_qid_pattern = re.compile(
        r"SEARCH_ENTITY_CANDIDATES|\[|\]|\(|\)|\{|\}|\"QID\"|'QID'",
        re.IGNORECASE,
    )

    allowed = get_authorized_qids()
    if not allowed:
        return (
            "Error: Tool-order protocol violation. "
            "Call search_entity_candidates(entity_name, entity_type) first for each entity, "
            "select one returned candidate, then pass its literal QID to fetch_entity_properties "
            "(for example: qid='Q142'). No candidate QIDs are registered for this run."
        )

    if not qid_pattern.match(qid):
        if expression_like_qid_pattern.search(qid):
            return (
                "Error: Invalid QID argument. "
                "Call search_entity_candidates(entity_name, entity_type) first, "
                "select a returned candidate, then pass only its literal QID string "
                "to fetch_entity_properties (for example: 'Q142')."
            )
        return f"Error: Invalid QID '{qid}'. Must be 'Q' followed by digits (e.g., 'Q142')."

    if not is_qid_authorized(qid):
        allowed_hint = (
            f" Authorized candidate QIDs in this run: {', '.join(allowed)}."
            if allowed
            else " No candidate QIDs are registered for this run."
        )
        return (
            "Error: Tool-order protocol violation. "
            "Call search_entity_candidates(entity_name, entity_type) first for each entity, "
            "select a returned QID, then call fetch_entity_properties."
            f"{allowed_hint}"
        )

    # Handle nested property lists
    processed_properties = []
    for p in properties:
        if isinstance(p, list):
            if len(p) > 0:
                processed_properties.append(str(p[0]))
        else:
            processed_properties.append(str(p))

    valid_props = []
    for raw_prop in processed_properties:
        prop_id = _normalize_property_id(raw_prop)
        if prop_id:
            valid_props.append(prop_id)
    valid_props, auto_added_props = _augment_properties_for_question(
        valid_props,
        question=get_question_context(),
    )

    if not valid_props:
        return "Error: No valid properties specified."

    try:
        property_bindings: Dict[str, List[Dict[str, Any]]] = {}
        any_data = False
        inferred_qids_all: set[str] = set()

        for prop in valid_props:
            query = build_property_sparql_query(
                qid=qid,
                prop=prop,
                include_qualifiers=include_qualifiers,
            )
            result = _run_sparql(query)
            bindings = result.get("results", {}).get("bindings", [])
            property_bindings[prop] = bindings
            if bindings:
                any_data = True

            if prop not in {"P31", "P279"}:
                inferred = _extract_qids_from_property_bindings(bindings)
                for q in inferred:
                    inferred_qids_all.add(q)

        if not any_data:
            return f"Error: Entity {qid} not found or has no data for requested properties."

        newly_authorized_qids = register_inferred_qids(sorted(inferred_qids_all))

        formatted_results, wikipedia_url = format_property_results(
            property_bindings=property_bindings,
            valid_props=valid_props,
            qid=qid,
            include_qualifiers=include_qualifiers,
            question=get_question_context(),
        )
        if newly_authorized_qids:
            formatted_results = (
                "Tool note: inferred QIDs now authorized for follow-up fetches: "
                + ", ".join(newly_authorized_qids[:8])
                + ("\n" if len(newly_authorized_qids) <= 8 else ", ...\n")
                + formatted_results
            )
        if auto_added_props:
            formatted_results = (
                "Tool note: auto-added intent properties: "
                + ", ".join(sorted(set(auto_added_props)))
                + "\n"
                + formatted_results
            )

        log_tool_usage(
            "fetch_entity_properties",
            {
                "qid": qid,
                "properties": properties,
                "auto_added_properties": auto_added_props,
                "newly_authorized_qids": newly_authorized_qids,
                "include_qualifiers": include_qualifiers,
            },
            formatted_results,
        )

        return formatted_results

    except Exception as e:
        return f"Error fetching properties for {qid}: {e}"


def build_property_sparql_query(
    qid: str,
    prop: str,
    include_qualifiers: bool = True,
) -> str:
    """Build a statement-level SPARQL query for a single property."""
    qualifier_selects: List[str] = []
    qualifier_optionals: List[str] = []
    if include_qualifiers:
        for qualifier_prop in ("P580", "P582", "P585"):
            qualifier_var = f"?{qualifier_prop.lower()}"
            qualifier_selects.append(qualifier_var)
            qualifier_optionals.append(
                f"  OPTIONAL {{ ?statement pq:{qualifier_prop} {qualifier_var} . }}"
            )

    query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>

SELECT ?itemLabel ?itemDescription ?wikipediaUrl ?value ?valueLabel {' '.join(qualifier_selects)} WHERE {{
  BIND(wd:{qid} AS ?item)

  OPTIONAL {{
    ?item p:{prop} ?statement .
    ?statement ps:{prop} ?value .
{chr(10).join(qualifier_optionals)}
  }}

  OPTIONAL {{
    ?wikipediaUrl schema:about ?item ;
                  schema:isPartOf <https://en.wikipedia.org/> .
  }}

  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en".
  }}
}}
LIMIT 500"""

    return query


def format_property_results(
    property_bindings: Dict[str, List[Dict[str, Any]]],
    valid_props: List[str],
    qid: str = "",
    include_qualifiers: bool = True,
    question: str = "",
) -> tuple[str, Optional[str]]:
    """Format SPARQL results into a readable string."""
    if not property_bindings:
        return "No data found.", None

    collected: Dict[str, List[Dict[str, Any]]] = {p: [] for p in valid_props}
    dedupe_keys: Dict[str, set[tuple[str, str, str, str]]] = {
        p: set() for p in valid_props
    }
    entity_label = None
    entity_desc = None
    wikipedia_url = None

    for prop in valid_props:
        bindings = property_bindings.get(prop, [])
        for b in bindings:
            if not entity_label:
                entity_label = b.get("itemLabel", {}).get("value")
            if not entity_desc:
                entity_desc = b.get("itemDescription", {}).get("value")
            if not wikipedia_url:
                wikipedia_url = b.get("wikipediaUrl", {}).get("value")

            raw_value = b.get("value", {}).get("value", "")
            value = b.get("valueLabel", {}).get("value") or raw_value
            if not value:
                continue
            value = _format_time_value(value)
            linked_qid = _qid_from_entity_value(raw_value)
            if linked_qid:
                value = f"{value} [{linked_qid}]"

            start_time = b.get("p580", {}).get("value", "")
            end_time = b.get("p582", {}).get("value", "")
            point_in_time = b.get("p585", {}).get("value", "")

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

            normalized_value, alias_note = _normalize_temporal_alias(
                prop=prop,
                value=value,
                qualifiers=qualifiers,
            )

            collected[prop].append(
                {
                    "value": normalized_value,
                    "qualifiers": qualifiers,
                    "alias_note": alias_note,
                }
            )

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
        prop_name = WIKIDATA_PROPERTIES.get(
            prop,
            "property (not in suggestion catalog)",
        )
        label = f"{prop}: {prop_name}"
        all_entries = list(collected[prop])
        entries = list(all_entries)
        if prop in {"P35", "P6", "P39", "P169"}:
            entries.sort(key=_entry_sort_key, reverse=True)
            display_limit = 12
            entries = entries[:display_limit]
        else:
            display_limit = 5
            entries = entries[:display_limit]
        if entries:
            if len(entries) == 1 and not entries[0]["qualifiers"]:
                lines.append(f"{label} — {entries[0]['value']}")
            else:
                lines.append(f"{label}:")
                for entry in entries:
                    value = entry["value"]
                    qualifiers = entry["qualifiers"]
                    alias_note = entry.get("alias_note")
                    if qualifiers:
                        qualifier_parts = []
                        for qid_, qvalue in qualifiers.items():
                            if qid_ == "P580":
                                qualifier_parts.append(f"start: {qvalue}")
                            elif qid_ == "P582":
                                qualifier_parts.append(f"end: {qvalue}")
                            elif qid_ == "P585":
                                qualifier_parts.append(f"time: {qvalue}")
                        if alias_note:
                            qualifier_parts.append(alias_note)
                        lines.append(f"  - {value} ({', '.join(qualifier_parts)})")
                    else:
                        if alias_note:
                            lines.append(f"  - {value} ({alias_note})")
                        else:
                            lines.append(f"  - {value}")
                remaining_count = max(0, len(all_entries) - len(entries))
                if remaining_count > 0:
                    lines.append(f"  - +{remaining_count} more")
        else:
            lines.append(f"{label}: (not available)")

    # Add deterministic next-hop guidance for multi-hop questions.
    question_lower = (question or "").strip().lower()
    needs_capital = "capital" in question_lower
    needs_continent = "continent" in question_lower
    guidance: List[str] = []

    def _first_linked_qid(prop_id: str) -> str:
        for entry in collected.get(prop_id, []):
            match = _VALUE_QID_PATTERN.search(str(entry.get("value", "")))
            if match:
                return match.group(1)
        return ""

    if needs_capital and not collected.get("P36"):
        bridge_qid = _first_linked_qid("P27") or _first_linked_qid("P17")
        if bridge_qid:
            guidance.append(
                f"NEXT STEP hint: fetch_entity_properties(qid='{bridge_qid}', properties=['P36'], include_qualifiers=true)"
            )

    if needs_continent and not collected.get("P30"):
        bridge_qid = _first_linked_qid("P27") or _first_linked_qid("P17")
        if bridge_qid:
            guidance.append(
                f"NEXT STEP hint: fetch_entity_properties(qid='{bridge_qid}', properties=['P30'], include_qualifiers=true)"
            )

    if guidance:
        lines.append("")
        lines.extend(guidance)

    return "\n".join(lines), wikipedia_url
