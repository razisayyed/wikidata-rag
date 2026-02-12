"""LangChain tool implementations for Wikidata entity candidate search."""

from __future__ import annotations

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

        # Build display string
        if desc and instance_of:
            info = f"{desc} (Type: {instance_of})"
        elif desc:
            info = desc
        elif instance_of:
            info = f"Type: {instance_of}"
        else:
            info = "(no description)"

        lines.append(f"{i}. [{c['qid']}] {c['label']} - {info}")

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

    # Additional filtering based on entity_type hint
    if entity_type:
        # Common type mappings for better filtering
        type_keywords = {
            # People-related types
            "person": [
                "person",
                "human",
                "politician",
                "scientist",
                "artist",
                "author",
            ],
            "scientist": [
                "scientist",
                "researcher",
                "physicist",
                "biologist",
                "chemist",
            ],
            "politician": [
                "politician",
                "president",
                "prime minister",
                "senator",
                "governor",
            ],
            "athlete": [
                "athlete",
                "footballer",
                "basketball player",
                "runner",
                "swimmer",
            ],
            # Place-related types
            "country": ["country", "sovereign state", "nation"],
            "city": ["city", "town", "municipality", "metropolis"],
            "organization": ["organization", "company", "university", "institution"],
            # Geographic features
            "mountain": ["mountain", "peak", "summit"],
            "lake": ["lake", "body of water"],
            "island": ["island", "archipelago"],
            # Work types
            "film": ["film", "movie", "motion picture"],
            "book": ["book", "novel", "literary work", "publication"],
            "album": ["album", "studio album", "music album"],
            "song": ["song", "single", "musical composition"],
            "painting": ["painting", "artwork", "oil painting"],
            "software": ["software", "computer program", "application"],
            "game": ["video game", "game", "computer game"],
            # Organization types
            "company": ["company", "corporation", "business", "enterprise"],
            "band": ["band", "musical group", "rock band"],
            "sports_team": ["sports team", "football club", "basketball team"],
            "political_party": ["political party", "party"],
            "ngo": ["non-governmental organization", "NGO", "nonprofit"],
            # Other types
            "species": ["species", "taxon", "organism"],
            "chemical": ["chemical compound", "chemical element", "molecule"],
            "disease": ["disease", "medical condition", "illness"],
            "event": ["event", "occurrence", "historical event"],
            "award": ["award", "prize", "honor"],
        }
        keywords = type_keywords.get(entity_type.lower(), [entity_type.lower()])

        def score_entity(e: Dict[str, Any]) -> int:
            """Score entity based on type match."""

            score = 0
            desc_lower = e["description"].lower()
            types_lower = [t.lower() for t in e["instance_of"]]

            for kw in keywords:
                if kw in desc_lower:
                    score += 2
                if any(kw in t for t in types_lower):
                    score += 3
            return score

        # Sort by type match score (higher first), keeping original order for ties
        filtered_entities.sort(key=lambda e: score_entity(e), reverse=True)

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
