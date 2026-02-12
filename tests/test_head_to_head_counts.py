from __future__ import annotations

from kb_project.benchmark.models import (
    CaseResult,
    EvaluatorResult,
    ModelOutput,
    TestCase,
)
from kb_project.benchmark.runner import _compute_evaluator_summary


def _build_case(case_id: str, winner: str) -> CaseResult:
    return CaseResult(
        test_case=TestCase(
            id=case_id,
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
                baseline_label="factual",
                winner=winner,
            ),
            "aimon": EvaluatorResult(
                name="aimon",
                status="completed",
                rag_label="factual",
                baseline_label="factual",
                winner="Tie",
            ),
            "llm_judge": EvaluatorResult(
                name="llm_judge",
                status="skipped",
                rag_label="skipped",
                baseline_label="skipped",
                winner="N/A",
            ),
            "ragtruth": EvaluatorResult(
                name="ragtruth",
                status="completed",
                rag_label="hallucinated",
                baseline_label="hallucinated",
                winner="Tie",
            ),
        },
    )


def test_head_to_head_summary_counts_wins_and_ties():
    summary = _compute_evaluator_summary([
        _build_case("c1", "RAG"),
        _build_case("c2", "BASELINE"),
        _build_case("c3", "Tie"),
    ])

    assert summary["vectara"]["rag_wins"] == 1
    assert summary["vectara"]["baseline_wins"] == 1
    assert summary["vectara"]["ties"] == 1

    assert summary["llm_judge"]["skipped"] == 3
