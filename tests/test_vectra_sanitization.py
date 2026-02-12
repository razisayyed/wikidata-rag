from __future__ import annotations

from kb_project.benchmark.vectra import AgentRun, ToolCall, sanitize_tool_output


def test_search_candidate_chatter_is_removed():
    output = """CANDIDATES for 'Ada Lovelace' (2 found):

1. [Q7259] Ada Lovelace - English mathematician (Type: human)
2. [Q114068528] Ada Lovelace - GPU microarchitecture (Type: microarchitecture)
INSTRUCTIONS: Analyze these candidates and select the best one.
"""
    cleaned = sanitize_tool_output("search_entity_candidates", output)
    assert cleaned == ""


def test_no_candidate_signal_is_preserved():
    output = "NO CANDIDATES FOUND for 'Dr. Liora Anstrum'. Entity cannot be verified in Wikidata."
    cleaned = sanitize_tool_output("search_entity_candidates", output)
    assert "NO CANDIDATES FOUND" in cleaned


def test_agent_run_sanitized_context_keeps_factual_tool_outputs():
    run = AgentRun(
        question="Where did Alan Turing work during World War II?",
        tool_calls=[
            ToolCall(
                name="search_entity_candidates",
                args={"entity_name": "Alan Turing", "entity_type": "person"},
                output="CANDIDATES for 'Alan Turing' (3 found):\n1. [Q7251] Alan Turing - ...",
            ),
            ToolCall(
                name="fetch_entity_properties",
                args={"qid": "Q7251", "properties": ["P108"]},
                output="Entity: Alan Turing\nP108: employer - Government Code and Cypher School",
            ),
        ],
    )
    sanitized = run.sanitized_retrieved_context
    assert "CANDIDATES for" not in sanitized
    assert "Government Code and Cypher School" in sanitized
