from __future__ import annotations

from kb_project.benchmark.models import (
    CaseResult,
    EvaluatorResult,
    ModelOutput,
    TestCase,
    winner_from_labels,
)
from kb_project.benchmark.runner import _compute_evaluator_summary


def test_winner_resolution_is_local_to_single_evaluator():
    assert winner_from_labels("factual", "hallucinated") == "RAG"
    assert winner_from_labels("hallucinated", "factual") == "BASELINE"
    assert winner_from_labels("factual", "factual") == "Tie"


def test_summary_keeps_evaluators_independent():
    case = CaseResult(
        test_case=TestCase(
            id="case_x",
            question="Q",
            ground_truth="A",
            category="test",
        ),
        rag_output=ModelOutput(response="A"),
        baseline_output=ModelOutput(response="B"),
        evaluations={
            "vectara": EvaluatorResult(
                name="vectara",
                status="completed",
                rag_label="factual",
                baseline_label="hallucinated",
                winner="RAG",
            ),
            "aimon": EvaluatorResult(
                name="aimon",
                status="completed",
                rag_label="hallucinated",
                baseline_label="factual",
                winner="BASELINE",
            ),
            "llm_judge": EvaluatorResult(
                name="llm_judge",
                status="completed",
                rag_label="factual",
                baseline_label="factual",
                winner="Tie",
            ),
            "ragtruth": EvaluatorResult(
                name="ragtruth",
                status="skipped",
                rag_label="skipped",
                baseline_label="skipped",
                winner="N/A",
            ),
        },
    )

    summary = _compute_evaluator_summary([case])

    assert summary["vectara"]["rag_wins"] == 1
    assert summary["vectara"]["baseline_wins"] == 0

    assert summary["aimon"]["baseline_wins"] == 1
    assert summary["aimon"]["rag_wins"] == 0

    assert summary["llm_judge"]["ties"] == 1
    assert summary["ragtruth"]["skipped"] == 1


def test_runner_source_no_longer_contains_dual_track_fields():
    with open("kb_project/benchmark/runner.py", "r", encoding="utf-8") as source_file:
        source = source_file.read()

    assert "dual_track" not in source
    assert "consensus" not in source
    assert "rag_completeness" not in source
