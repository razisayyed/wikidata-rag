from __future__ import annotations

import importlib

from kb_project.tools.fetch_entity_properties import format_property_results
from kb_project.tools.wikidata_sparql import (
    is_safe_read_only_select,
    MAX_SPARQL_ROWS,
)


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
