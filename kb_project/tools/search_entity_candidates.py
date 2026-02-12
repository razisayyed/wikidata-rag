"""LangChain tool implementations for Wikidata entity candidate search."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain.tools import tool
from pydantic import BaseModel, Field

from ..utils.logging import (
    configure_logging,
    log_tool,
    log_tool_usage,
)
from ..settings import MAX_SEARCH_RESULTS
from .tool_protocol_state import register_search_candidates
from ..wikidata.sparql import run_sparql as _run_sparql

logger = configure_logging()

_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "person": ["person", "human", "scientist", "politician", "artist", "author"],
    "scientist": ["scientist", "researcher", "physicist", "biologist", "chemist"],
    "politician": ["politician", "president", "prime minister", "senator", "governor"],
    "athlete": ["athlete", "footballer", "basketball player", "runner", "swimmer"],
    "country": ["country", "sovereign state", "nation"],
    "city": ["city", "town", "municipality", "metropolis"],
    "organization": ["organization", "company", "university", "institution"],
    "mountain": ["mountain", "peak", "summit"],
    "lake": ["lake", "body of water"],
    "island": ["island", "archipelago"],
    "film": ["film", "movie", "motion picture"],
    "book": ["book", "novel", "literary work", "publication"],
    "album": ["album", "studio album", "music album"],
    "song": ["song", "single", "musical composition"],
    "painting": ["painting", "artwork", "oil painting"],
    "software": ["software", "computer program", "application"],
    "game": ["video game", "game", "computer game"],
    "company": ["company", "corporation", "business", "enterprise"],
    "band": ["band", "musical group", "rock band"],
    "sports_team": ["sports team", "football club", "basketball team"],
    "political_party": ["political party", "party"],
    "ngo": ["non-governmental organization", "ngo", "nonprofit"],
    "species": ["species", "taxon", "organism"],
    "chemical": ["chemical compound", "chemical element", "molecule"],
    "disease": ["disease", "medical condition", "illness"],
    "event": ["event", "occurrence", "historical event", "war"],
    "award": ["award", "prize", "honor"],
}
_PERSON_TITLE_PREFIXES = ("dr ", "prof ", "mr ", "mrs ", "ms ")
_NON_PERSON_HINTS = (
    "street",
    "painting",
    "album",
    "film",
    "award",
    "building",
    "timeline",
    "radio",
    "asteroid",
    "memorial",
    "episode",
    "year",
)


def _normalize_for_match(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_for_match(text)
    if not normalized:
        return []
    return normalized.split()


def _is_person_like_query(entity_name: str, entity_type: str) -> bool:
    if (entity_type or "").strip().lower() in {"person", "scientist", "politician", "athlete"}:
        return True

    normalized = _normalize_for_match(entity_name)
    if not normalized:
        return False

    stripped = normalized
    for prefix in _PERSON_TITLE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break
    tokens = stripped.split()
    return len(tokens) >= 2


def _type_score(entity: Dict[str, Any], entity_type: str) -> int:
    if not entity_type:
        return 0
    keywords = _TYPE_KEYWORDS.get(entity_type.lower(), [entity_type.lower()])
    desc_lower = str(entity.get("description", "")).lower()
    types_lower = [t.lower() for t in entity.get("instance_of", [])]
    score = 0
    for kw in keywords:
        if kw in desc_lower:
            score += 2
        if any(kw in t for t in types_lower):
            score += 3
    return score


def _score_entity(entity_name: str, entity: Dict[str, Any], entity_type: str) -> int:
    query_norm = _normalize_for_match(entity_name)
    label_norm = _normalize_for_match(str(entity.get("label", "")))
    desc_lower = str(entity.get("description", "")).lower()
    types_lower = [t.lower() for t in entity.get("instance_of", [])]

    score = 0

    if label_norm == query_norm:
        score += 30
    elif label_norm.startswith(query_norm):
        score += 18
    elif query_norm and query_norm in label_norm:
        score += 10

    query_tokens = set(_tokenize(query_norm))
    label_tokens = set(_tokenize(label_norm))
    token_overlap = len(query_tokens & label_tokens)
    score += token_overlap * 2
    if query_tokens and query_tokens.issubset(label_tokens):
        score += 8

    if _is_person_like_query(entity_name, entity_type):
        if any("human" in t or "person" in t for t in types_lower):
            score += 8
        else:
            score -= 10
        if any(h in desc_lower for h in _NON_PERSON_HINTS):
            score -= 7
        if re.search(r"\b\d{4}[–-]\d{4}\b", desc_lower):
            score += 3

    # Penalize likely unwanted fictional matches for real-world factual queries.
    if "fictional" in desc_lower and "fictional" not in query_norm:
        score -= 6

    score += _type_score(entity, entity_type)
    return score


def _confidence_label(score: int, gap_to_next: int) -> str:
    if score >= 24 and gap_to_next >= 4:
        return "high"
    if score >= 14:
        return "medium"
    return "low"


class SearchCandidatesInput(BaseModel):
    """Input for entity search."""

    entity_name: str = Field(description="The name of the entity to search for")
    entity_type: str = Field(
        default="",
        description="OPTIONAL type hint for disambiguation (e.g., 'person', 'country', 'organization'). "
        "Use only for ambiguous names.",
    )


@tool("search_entity_candidates", args_schema=SearchCandidatesInput)
def search_entity_candidates(entity_name: str, entity_type: str = "") -> str:
    """
    Search for entity candidates in Wikidata using SPARQL EntitySearch.

    Returns a numbered list of candidates with QID, label, description, and type.
    YOU must analyze the candidates and select the best match based on:
    - The question context
    - The entity type expected
    - The description relevance

    After selecting a candidate, use its QID with fetch_entity_properties tool.
    If no candidate matches, conclude the entity cannot be verified.
    """

    entity_name = entity_name.replace("\u00a0", " ").strip()

    candidates = search_entity_sparql(entity_name, entity_type=entity_type)

    if not candidates:
        return f"NO CANDIDATES FOUND for '{entity_name}'. Entity cannot be verified in Wikidata."

    register_search_candidates(entity_name=entity_name, candidates=candidates)

    # Format candidates for LLM analysis
    lines = [f"CANDIDATES for '{entity_name}' ({len(candidates)} found):"]
    lines.append("")

    for i, c in enumerate(candidates, 1):
        desc = c.get("description", "")
        instance_of = c.get("instance_of", "")
        confidence = str(c.get("confidence", "low")).upper()
        score = c.get("disambiguation_score", 0)

        # Build display string
        if desc and instance_of:
            info = f"{desc} (Type: {instance_of})"
        elif desc:
            info = desc
        elif instance_of:
            info = f"Type: {instance_of}"
        else:
            info = "(no description)"

        lines.append(
            f"{i}. [{c['qid']}] {c['label']} - {info} "
            f"(disambiguation: {confidence}, score: {score})"
        )

    top_score = int(candidates[0].get("disambiguation_score", 0))
    second_score = int(candidates[1].get("disambiguation_score", -999)) if len(candidates) > 1 else -999
    top_conf = str(candidates[0].get("confidence", "low")).lower()
    ambiguous = (top_score - second_score) <= 2 and len(candidates) > 1

    lines.append("")
    if top_conf == "low" or ambiguous:
        lines.append(
            "DISAMBIGUATION WARNING: no high-confidence unique match was found."
        )
        lines.append(
            "NO-ANSWER GATING: If you cannot disambiguate confidently from question context, "
            "do NOT select a QID and refuse verification for that entity."
        )
    else:
        lines.append(
            f"Top candidate confidence: {top_conf.upper()} (score gap to next: {top_score - second_score})."
        )

    lines.append("")
    lines.append(
        "NEXT STEP: Call fetch_entity_properties with one literal QID from above "
        "(example: qid='Q142'). Do not use code/expression syntax."
    )

    log_tool_usage(
        "search_entity_candidates",
        {"entity_name": entity_name, "entity_type": entity_type},
        f"{len(candidates)} candidates",
    )

    return "\n".join(lines)


def search_entity_sparql(
    label: str, limit: int = MAX_SEARCH_RESULTS, entity_type: str = ""
) -> List[Dict[str, str]]:
    """
    Search Wikidata for entities matching *label* using the mwapi service.

    Args:
        label: The entity name to search for
        limit: Maximum number of results to return
        entity_type: Optional type hint (e.g., 'person', 'country', 'city', 'organization')
                    Used to filter and prioritize results

    Returns:
        List of entity dictionaries with qid, label, description, and instance_of
    """
    safe_label = label.replace('"', '\\"')

    # Build query with optional instance_of information for better context
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>

SELECT ?item ?itemLabel ?itemDescription ?instanceOfLabel WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:api "EntitySearch" ;
                    wikibase:endpoint "www.wikidata.org" ;
                    mwapi:search "{safe_label}" ;
                    mwapi:language "en" .
    ?item wikibase:apiOutputItem mwapi:item .
  }}
  OPTIONAL {{ ?item wdt:P31 ?instanceOf . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {limit * 2}
"""
    try:
        results = _run_sparql(query)
    except Exception as exc:
        log_tool("SPARQL Search", f"❌ Error: {exc}", "🔍")
        return []

    # Collect entities and deduplicate (same entity may appear multiple times with different instance_of)
    entities_dict: Dict[str, Dict[str, Any]] = {}
    for b in results.get("results", {}).get("bindings", []):
        uri = b["item"]["value"]
        qid = uri.rsplit("/", 1)[-1]
        lbl = b.get("itemLabel", {}).get("value", qid)
        desc = b.get("itemDescription", {}).get("value", "")
        instance_of = b.get("instanceOfLabel", {}).get("value", "")

        if qid not in entities_dict:
            entities_dict[qid] = {
                "qid": qid,
                "label": lbl,
                "description": desc,
                "instance_of": [instance_of] if instance_of else [],
            }
        elif instance_of and instance_of not in entities_dict[qid]["instance_of"]:
            entities_dict[qid]["instance_of"].append(instance_of)

    # Convert to list
    entities = list(entities_dict.values())

    # Filter out Wikimedia internal pages
    wikimedia_types = {
        "Wikimedia category",
        "Wikimedia disambiguation page",
        "Wikimedia template",
        "Wikimedia project page",
        "Wikimedia list article",
        "Wikimedia internal item",
    }

    filtered_entities = []
    for e in entities:
        if e["description"] in wikimedia_types:
            continue
        filtered_entities.append(e)

    # Composite ranking for stronger disambiguation and stable candidate quality.
    for e in filtered_entities:
        e["disambiguation_score"] = _score_entity(label, e, entity_type)

    filtered_entities.sort(
        key=lambda e: int(e.get("disambiguation_score", 0)),
        reverse=True,
    )

    # Add confidence label using top-2 margin.
    top_score = int(filtered_entities[0].get("disambiguation_score", 0)) if filtered_entities else 0
    second_score = int(filtered_entities[1].get("disambiguation_score", -999)) if len(filtered_entities) > 1 else -999
    gap = top_score - second_score
    for idx, e in enumerate(filtered_entities):
        score = int(e.get("disambiguation_score", 0))
        local_gap = gap if idx == 0 else score - int(filtered_entities[idx + 1].get("disambiguation_score", -999)) if idx + 1 < len(filtered_entities) else score
        e["confidence"] = _confidence_label(score, local_gap)

    # Format instance_of as string for output
    for e in filtered_entities:
        if e["instance_of"]:
            e["instance_of"] = ", ".join(e["instance_of"][:3])  # Limit to first 3 types
        else:
            e["instance_of"] = ""

    log_tool(
        "SPARQL Search", f"Found {len(filtered_entities)} entities for '{label}'", "🔍"
    )
    return filtered_entities[:limit]
