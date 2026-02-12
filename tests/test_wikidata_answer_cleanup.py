from __future__ import annotations

from kb_project.wikidata_rag_agent import finalize_agent_answer, is_process_message


def test_finalize_answer_removes_wikidata_process_references():
    raw = (
        "I cannot verify a real-world collaboration between Dr. Liora Anstrum and "
        "Prof. Armin Delacroix in Wikidata, based on the search results."
    )
    cleaned = finalize_agent_answer(
        raw,
        "Tell me about the collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
    )

    assert "wikidata" not in cleaned.lower()
    assert "search results" not in cleaned.lower()
    assert ",." not in cleaned
    assert "I cannot verify a real-world collaboration" in cleaned


def test_finalize_answer_removes_wikipedia_reference_phrase():
    raw = "Paris is the capital of France from Wikipedia."
    cleaned = finalize_agent_answer(raw, "What is the capital of France?")

    assert "wikipedia" not in cleaned.lower()
    assert "Paris is the capital of France" in cleaned


def test_finalize_answer_removes_wikidata_id_clause():
    raw = (
        "George Orwell, whose Wikidata ID is Q3335, is the British writer and journalist "
        "who wrote the novel '1984'."
    )
    cleaned = finalize_agent_answer(raw, "Who wrote the novel '1984'?")

    assert "wikidata" not in cleaned.lower()
    assert "qid" not in cleaned.lower()
    assert "Q3335" not in cleaned
    assert "George Orwell" in cleaned
    assert "wrote the novel '1984'" in cleaned


def test_python_tag_tool_payload_is_treated_as_process_message():
    raw = (
        '<|python_tag|>{"name":"search_entity_candidates","parameters":'
        '{"entity_name":"Niels Bohr","entity_type":"person"}}'
    )
    assert is_process_message(raw) is True


def test_finalize_answer_drops_python_tag_tool_payload():
    raw = (
        '<|python_tag|>{"name":"search_entity_candidates","parameters":'
        '{"entity_name":"Niels Bohr","entity_type":"person"}}'
    )
    cleaned = finalize_agent_answer(
        raw,
        "When was Niels Bohr born and what were his major achievements?",
    )
    assert cleaned == ""
