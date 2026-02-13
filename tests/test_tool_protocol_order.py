from __future__ import annotations

import importlib

import pytest

from kb_project.tools.tool_protocol_state import (
    register_search_candidates,
    reset_tool_protocol_state,
)


def test_fetch_entity_properties_rejects_call_before_search():
    reset_tool_protocol_state()
    fetch_module = importlib.import_module("kb_project.tools.fetch_entity_properties")

    payload = fetch_module.fetch_entity_properties.invoke(
        {"qid": "Q142", "properties": ["P36"]}
    )

    assert "Tool-order protocol violation" in payload
    assert "search_entity_candidates" in payload


def test_fetch_entity_properties_rejects_expression_style_qid(monkeypatch):
    reset_tool_protocol_state()
    search_module = importlib.import_module("kb_project.tools.search_entity_candidates")
    fetch_module = importlib.import_module("kb_project.tools.fetch_entity_properties")

    monkeypatch.setattr(
        search_module,
        "search_entity_sparql",
        lambda _entity_name, limit=10, entity_type="": [
            {
                "qid": "Q142",
                "label": "France",
                "description": "country in Western Europe",
                "instance_of": "country",
            }
        ],
    )
    search_module.search_entity_candidates.invoke({"entity_name": "France"})

    with pytest.raises(Exception) as exc_info:
        fetch_module.fetch_entity_properties.invoke(
            {
                "qid": 'search_entity_candidates(entity_name="France", entity_type="country")[0]["qid"]',
                "properties": ["P36"],
            }
        )

    message = str(exc_info.value)
    assert "String should match pattern '^Q\\d+$'" in message or "string_pattern_mismatch" in message


def test_fetch_entity_properties_allows_qid_from_search(monkeypatch):
    reset_tool_protocol_state()
    search_module = importlib.import_module("kb_project.tools.search_entity_candidates")
    fetch_module = importlib.import_module("kb_project.tools.fetch_entity_properties")

    monkeypatch.setattr(
        search_module,
        "search_entity_sparql",
        lambda _entity_name, limit=10, entity_type="": [
            {
                "qid": "Q142",
                "label": "France",
                "description": "country in Western Europe",
                "instance_of": "country",
            }
        ],
    )

    search_module.search_entity_candidates.invoke({"entity_name": "France"})

    monkeypatch.setattr(
        fetch_module,
        "_run_sparql",
        lambda _query: {
            "results": {
                "bindings": [
                    {
                        "itemLabel": {"value": "France"},
                        "itemDescription": {"value": "country in Western Europe"},
                        "p36ValueLabel": {"value": "Paris"},
                    }
                ]
            }
        },
    )

    payload = fetch_module.fetch_entity_properties.invoke(
        {"qid": "Q142", "properties": ["P36"]}
    )

    assert "Tool-order protocol violation" not in payload
    assert "Entity: France" in payload
    assert "P36: capital" in payload
    assert "Paris" in payload


def test_fetch_entity_properties_allows_qid_inferred_from_previous_fetch(monkeypatch):
    reset_tool_protocol_state()
    fetch_module = importlib.import_module("kb_project.tools.fetch_entity_properties")

    # Seed protocol state with one explicit candidate.
    register_search_candidates(
        entity_name="penicillin",
        candidates=[{"qid": "Q12190", "label": "penicillin"}],
    )

    def fake_run_sparql(query):
        if "wd:Q12190" in query:
            return {
                "results": {
                    "bindings": [
                        {
                            "itemLabel": {"value": "penicillin"},
                            "itemDescription": {"value": "group of antibiotics"},
                            "p61Value": {"value": "http://www.wikidata.org/entity/Q203"},
                            "p61ValueLabel": {"value": "Alexander Fleming"},
                        }
                    ]
                }
            }
        if "wd:Q203" in query:
            return {
                "results": {
                    "bindings": [
                        {
                            "itemLabel": {"value": "Alexander Fleming"},
                            "itemDescription": {"value": "Scottish physician"},
                            "p31ValueLabel": {"value": "human"},
                        }
                    ]
                }
            }
        return {"results": {"bindings": []}}

    monkeypatch.setattr(fetch_module, "_run_sparql", fake_run_sparql)

    first_payload = fetch_module.fetch_entity_properties.invoke(
        {"qid": "Q12190", "properties": ["P61"]}
    )
    assert "inferred QIDs now authorized" in first_payload
    assert "Q203" in first_payload

    # No intermediate search call for Q203: must still be accepted.
    second_payload = fetch_module.fetch_entity_properties.invoke(
        {"qid": "Q203", "properties": ["P31"]}
    )
    assert "Tool-order protocol violation" not in second_payload
    assert "Entity: Alexander Fleming" in second_payload
    assert "P31: instance of" in second_payload


def test_protocol_state_reset_disables_previous_qids():
    reset_tool_protocol_state()
    fetch_module = importlib.import_module("kb_project.tools.fetch_entity_properties")

    # Register candidate first without network calls.
    register_search_candidates(
        entity_name="France",
        candidates=[{"qid": "Q142", "label": "France"}],
    )
    # Reset should clear authorization.
    reset_tool_protocol_state()

    payload = fetch_module.fetch_entity_properties.invoke(
        {"qid": "Q142", "properties": ["P36"]}
    )
    assert "Tool-order protocol violation" in payload


def test_fetch_wikipedia_requires_prior_sparql_attempt():
    reset_tool_protocol_state()
    wiki_module = importlib.import_module("kb_project.tools.fetch_wikipedia_article")

    payload = wiki_module.fetch_wikipedia_article_tool.invoke(
        {"qid": "Q937", "entity_name": "Albert Einstein"}
    )

    assert "Tool-order protocol violation" in payload
    assert "wikidata_sparql" in payload


def test_fetch_wikipedia_allowed_after_sparql_attempt(monkeypatch):
    reset_tool_protocol_state()
    sparql_module = importlib.import_module("kb_project.tools.wikidata_sparql")
    wiki_module = importlib.import_module("kb_project.tools.fetch_wikipedia_article")

    monkeypatch.setattr(
        sparql_module,
        "_run_sparql",
        lambda _query: {"results": {"bindings": []}},
    )
    sparql_module.wikidata_sparql.invoke({"sparql": "SELECT * WHERE { ?s ?p ?o } LIMIT 1"})

    monkeypatch.setattr(
        wiki_module,
        "get_wikipedia_title_from_qid",
        lambda _qid: "Albert_Einstein",
    )
    monkeypatch.setattr(
        wiki_module,
        "fetch_wikipedia_article",
        lambda _title: "Albert Einstein was a physicist.",
    )

    payload = wiki_module.fetch_wikipedia_article_tool.invoke(
        {"qid": "Q937", "entity_name": "Albert Einstein"}
    )

    assert "Tool-order protocol violation" not in payload
    assert "Wikipedia Article: Albert Einstein (Q937)" in payload
