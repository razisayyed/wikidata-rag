from __future__ import annotations

from kb_project.benchmark.models import ModelOutput, TestCase
from kb_project.benchmark.runner import _evaluate_llm_judge


def test_llm_judge_missing_api_key_is_skipped(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = _evaluate_llm_judge(
        rag_output=ModelOutput(response="Paris is the capital of France."),
        baseline_output=ModelOutput(response="Paris is the capital of France."),
        test_case=TestCase(
            id="case",
            question="What is the capital of France?",
            ground_truth="Paris is the capital of France.",
            category="geography",
        ),
        reference_ground_truth="Paris is the capital of France.",
    )

    assert result.status == "skipped"
    assert result.rag_label == "skipped"
    assert "OPENAI_API_KEY" in result.notes
