from __future__ import annotations

from typing import Any, Dict

from SPARQLWrapper import JSON, SPARQLWrapper

from ..settings import WIKIDATA_ENDPOINT, WIKIDATA_USER_AGENT


def get_sparql_client() -> SPARQLWrapper:
    client = SPARQLWrapper(WIKIDATA_ENDPOINT)
    client.setReturnFormat(JSON)
    client.addCustomHttpHeader("User-Agent", WIKIDATA_USER_AGENT)
    return client


def run_sparql(query: str) -> Dict[str, Any]:
    client = get_sparql_client()
    client.setQuery(query)
    result = client.query().convert()
    return result  # type: ignore[return-value]
