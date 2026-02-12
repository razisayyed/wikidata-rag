from __future__ import annotations

from kb_project.benchmark.vectra import GROUND_TRUTH_TEST_CASES


def test_case_2_replaced_with_niels_bohr():
    case = GROUND_TRUTH_TEST_CASES[1]

    assert case.question == "When was Niels Bohr born and what were his major achievements?"
    assert "Marie Curie" not in case.question
    assert "7 October 1885" in case.ground_truth
    assert "1922 Nobel Prize in Physics" in case.ground_truth


def test_refusal_cases_use_real_world_wording():
    refusal_cases = [case for case in GROUND_TRUTH_TEST_CASES if case.refusal_expected]
    assert refusal_cases

    for case in refusal_cases:
        assert "real-world" in case.ground_truth.lower()
        assert any("real-world" in fact.lower() for fact in case.key_facts)
