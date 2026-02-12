from __future__ import annotations

from kb_project.benchmark.vectra import ToolCall, _apply_no_answer_gating


def test_no_answer_gating_refuses_when_entity_not_found():
    answer = _apply_no_answer_gating(
        answer="They collaborated in 1944.",
        tool_calls=[
            ToolCall(
                name="search_entity_candidates",
                args={"entity_name": "Dr. Liora Anstrum"},
                output="NO CANDIDATES FOUND for 'Dr. Liora Anstrum'.",
            )
        ],
    )
    assert "cannot verify" in answer.lower()


def test_no_answer_gating_refuses_on_disambiguation_warning():
    answer = _apply_no_answer_gating(
        answer="It was led by that person.",
        tool_calls=[
            ToolCall(
                name="search_entity_candidates",
                args={"entity_name": "John Smith"},
                output="DISAMBIGUATION WARNING: multiple similarly ranked entities.",
            )
        ],
    )
    assert "cannot determine" in answer.lower()
