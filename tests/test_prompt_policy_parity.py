from __future__ import annotations

from kb_project.prompts import (
    PROMPT_ONLY_SYSTEM_PROMPT,
    SHARED_FACTUAL_ANSWER_POLICY,
    WIKIDATA_SYSTEM_PROMPT,
)


def test_shared_answer_policy_is_applied_to_both_models():
    assert SHARED_FACTUAL_ANSWER_POLICY.strip() in PROMPT_ONLY_SYSTEM_PROMPT
    assert SHARED_FACTUAL_ANSWER_POLICY.strip() in WIKIDATA_SYSTEM_PROMPT


def test_rag_prompt_contains_tool_protocol_but_prompt_only_does_not():
    assert "search_entity_candidates" in WIKIDATA_SYSTEM_PROMPT
    assert "search_entity_candidates" not in PROMPT_ONLY_SYSTEM_PROMPT


def test_rag_prompt_enforces_entity_search_property_and_conditional_sparql_flow():
    assert "MANDATORY for every entity found in the question" in WIKIDATA_SYSTEM_PROMPT
    assert "MANDATORY for each candidate selected in step 2" in WIKIDATA_SYSTEM_PROMPT
    assert "If no candidate was selected in step 2" in WIKIDATA_SYSTEM_PROMPT
    assert "`wikidata_sparql` is REQUIRED before producing the final answer" in WIKIDATA_SYSTEM_PROMPT
    assert "Use it after structured properties and SPARQL attempts, not before." in WIKIDATA_SYSTEM_PROMPT
