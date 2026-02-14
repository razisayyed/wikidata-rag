from __future__ import annotations

import importlib

from kb_project.tools.fetch_entity_properties import format_property_results
from kb_project.tools.tool_protocol_state import (
    register_search_candidates,
    reset_tool_protocol_state,
    set_question_context,
)
from kb_project.tools.wikidata_sparql import (
    is_safe_read_only_select,
    MAX_SPARQL_ROWS,
)
from kb_project.wikidata.properties import WIKIDATA_PROPERTIES


def test_wikidata_sparql_accepts_prefix_select():
    query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?item WHERE { VALUES ?item { wd:Q142 } }
    """
    is_valid, error = is_safe_read_only_select(query)
    assert is_valid is True
    assert error == ""


def test_wikidata_sparql_rejects_mutation_keywords():
    query = "INSERT DATA { <a:b> <c:d> <e:f> }"
    is_valid, error = is_safe_read_only_select(query)
    assert is_valid is False
    assert "not allowed" in error


def test_wikidata_sparql_rejects_non_select_read_queries():
    query = "ASK WHERE { ?s ?p ?o }"
    is_valid, error = is_safe_read_only_select(query)
    assert is_valid is False
    assert "SELECT" in error


def test_wikidata_sparql_clips_oversized_max_rows(monkeypatch):
    module = importlib.import_module("kb_project.tools.wikidata_sparql")

    def fake_run_sparql(_):
        return {
            "results": {
                "bindings": [
                    {"item": {"value": f"Q{i}"}} for i in range(MAX_SPARQL_ROWS + 50)
                ]
            }
        }

    monkeypatch.setattr(module, "_run_sparql", fake_run_sparql)

    payload = module.wikidata_sparql.invoke(
        {
            "sparql": "SELECT ?item WHERE { VALUES ?item { wd:Q1 wd:Q2 } }",
            "max_rows": 1000,
        }
    )

    assert '"rows"' in payload
    # The last kept row should correspond to the safety cap boundary.
    assert f'"Q{MAX_SPARQL_ROWS - 1}"' in payload
    assert f'"Q{MAX_SPARQL_ROWS}"' not in payload


def test_fetch_entity_properties_formats_qualifiers():
    bindings = [
        {
            "itemLabel": {"value": "Alan Turing"},
            "itemDescription": {"value": "English computer scientist"},
            "p108ValueLabel": {"value": "Government Code and Cypher School"},
            "p108P580": {"value": "1938-09-04T00:00:00Z"},
            "p108P582": {"value": "1945-09-02T00:00:00Z"},
        }
    ]
    output, _ = format_property_results(
        bindings=bindings,
        valid_props=["P108"],
        qid="Q7251",
        include_qualifiers=True,
    )
    assert "P108: employer" in output
    assert "start: 1938-09-04" in output
    assert "end: 1945-09-02" in output


def test_fetch_entity_properties_temporal_alias_normalization_for_wwii():
    bindings = [
        {
            "itemLabel": {"value": "Alan Turing"},
            "itemDescription": {"value": "English computer scientist"},
            "p108ValueLabel": {"value": "Government Communications Headquarters"},
            "p108P580": {"value": "1938-01-01T00:00:00Z"},
            "p108P582": {"value": "1945-01-01T00:00:00Z"},
        }
    ]
    output, _ = format_property_results(
        bindings=bindings,
        valid_props=["P108"],
        qid="Q7251",
        include_qualifiers=True,
    )
    assert "Government Code and Cypher School (GC&CS)" in output
    assert "historical alias normalization" in output


def test_property_catalog_includes_core_physical_quantity_properties():
    # Needed for current benchmark coverage (boiling point and speed questions).
    assert "P2102" in WIKIDATA_PROPERTIES  # boiling point
    assert "P2052" in WIKIDATA_PROPERTIES  # speed


def test_property_catalog_includes_additional_disambiguation_and_name_properties():
    assert "P1889" in WIKIDATA_PROPERTIES  # different from
    assert "P460" in WIKIDATA_PROPERTIES  # said to be the same as
    assert "P1705" in WIKIDATA_PROPERTIES  # native label
    assert "P1448" in WIKIDATA_PROPERTIES  # official name


def test_fetch_entity_properties_auto_adds_discoverer_properties_from_question(monkeypatch):
    reset_tool_protocol_state()
    set_question_context("Who discovered penicillin?")
    register_search_candidates(
        entity_name="penicillin",
        candidates=[{"qid": "Q12190", "label": "penicillin"}],
    )

    module = importlib.import_module("kb_project.tools.fetch_entity_properties")
    captured = {"query": ""}

    def fake_run_sparql(query):
        captured["query"] = query
        return {
            "results": {
                "bindings": [
                    {
                        "itemLabel": {"value": "penicillin"},
                        "itemDescription": {
                            "value": "group of antibiotics derived from Penicillium fungi"
                        },
                        "p31ValueLabel": {"value": "structural class of chemical entities"},
                        "p61ValueLabel": {"value": "Alexander Fleming"},
                    }
                ]
            }
        }

    monkeypatch.setattr(module, "_run_sparql", fake_run_sparql)

    payload = module.fetch_entity_properties.invoke(
        {"qid": "Q12190", "properties": ["P31"]}
    )

    assert "p:P61" in captured["query"]
    assert "Tool note: auto-added intent properties:" in payload
    assert "P61" in payload
    assert "P575" in payload
    assert "P61: discoverer or inventor" in payload
    assert "Alexander Fleming" in payload


def test_fetch_entity_properties_auto_adds_broad_achievement_bundle(monkeypatch):
    reset_tool_protocol_state()
    set_question_context("When was this scientist born and what were their major achievements?")
    register_search_candidates(
        entity_name="Example Scientist",
        candidates=[{"qid": "Q999", "label": "Example Scientist"}],
    )

    module = importlib.import_module("kb_project.tools.fetch_entity_properties")
    captured = {"query": ""}

    def fake_run_sparql(query):
        captured["query"] = query
        return {
            "results": {
                "bindings": [
                    {
                        "itemLabel": {"value": "Example Scientist"},
                        "itemDescription": {"value": "scientist"},
                        "p106ValueLabel": {"value": "physicist"},
                        "p101ValueLabel": {"value": "quantum physics"},
                        "p800ValueLabel": {"value": "Example Landmark Work"},
                        "p166ValueLabel": {"value": "Nobel Prize in Physics"},
                    }
                ]
            }
        }

    monkeypatch.setattr(module, "_run_sparql", fake_run_sparql)

    payload = module.fetch_entity_properties.invoke(
        {"qid": "Q999", "properties": ["P106"]}
    )

    # Broader coverage beyond the single originally requested property.
    assert "p:P101" in captured["query"]
    assert "p:P800" in captured["query"]
    assert "p:P166" in captured["query"]
    assert "Tool note: auto-added intent properties:" in payload
    assert "P101" in payload
    assert "P800" in payload
    assert "P166" in payload


def test_fetch_entity_properties_accepts_non_catalog_property_id(monkeypatch):
    reset_tool_protocol_state()
    set_question_context("Give me a broad profile.")
    register_search_candidates(
        entity_name="Example Entity",
        candidates=[{"qid": "Q4242", "label": "Example Entity"}],
    )

    module = importlib.import_module("kb_project.tools.fetch_entity_properties")
    captured = {"query": ""}

    def fake_run_sparql(query):
        captured["query"] = query
        return {
            "results": {
                "bindings": [
                    {
                        "itemLabel": {"value": "Example Entity"},
                        "itemDescription": {"value": "example"},
                        "p31ValueLabel": {"value": "human"},
                    }
                ]
            }
        }

    monkeypatch.setattr(module, "_run_sparql", fake_run_sparql)

    payload = module.fetch_entity_properties.invoke(
        {"qid": "Q4242", "properties": ["P999999"]}
    )

    # Unknown-but-valid property IDs should be passed through (suggestions, not allowlist).
    assert "p:P999999" in captured["query"]
    assert "P999999: property (not in suggestion catalog)" in payload
