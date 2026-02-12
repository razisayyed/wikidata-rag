from __future__ import annotations

from kb_project.benchmark.vectra import GROUND_TRUTH_TEST_CASES


def test_dataset_size_and_refusal_balance():
    assert len(GROUND_TRUTH_TEST_CASES) == 24

    refusal_cases = [case for case in GROUND_TRUTH_TEST_CASES if case.refusal_expected]
    non_refusal_cases = [case for case in GROUND_TRUTH_TEST_CASES if not case.refusal_expected]

    assert len(refusal_cases) == 4
    assert len(non_refusal_cases) == 20


def test_required_case_topics_exist():
    questions = {case.question for case in GROUND_TRUTH_TEST_CASES}

    assert "Who is Albert Einstein?" in questions
    assert "When was Niels Bohr born and what were his major achievements?" in questions
    assert "What is the capital of the fictional country Eldoria Prime?" in questions
