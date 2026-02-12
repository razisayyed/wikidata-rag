from __future__ import annotations

from kb_project.benchmark.vectra import GROUND_TRUTH_TEST_CASES


def test_turing_case_contains_gc_cs_and_gchq_aliases():
    turing_case = next(
        case
        for case in GROUND_TRUTH_TEST_CASES
        if case.question == "What organization did Alan Turing work for during World War II?"
    )

    flattened = [value for group in turing_case.accepted_aliases for value in group]
    assert "GC&CS" in flattened
    assert "GCHQ" in flattened
    assert "Government Code and Cypher School" in flattened
