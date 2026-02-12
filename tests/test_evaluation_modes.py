from __future__ import annotations

from pathlib import Path

from kb_project.benchmark.evaluation import (
    evaluate_response,
    evaluate_rag_faithfulness,
)


class SpyModel:
    def __init__(self, score: float = 0.9):
        self.score = score
        self.last_pair = None

    def predict(self, pairs):
        self.last_pair = pairs[0]
        return [self.score]


class PhraseContextModel:
    def predict(self, pairs):
        scores = []
        for context, response in pairs:
            if (
                "Government Communications Headquarters" in context
                and "Government Communications Headquarters" in response
            ):
                scores.append(0.9)
            else:
                scores.append(0.1)
        return scores


def test_ground_truth_mode_ignores_retrieved_context():
    model = SpyModel()
    evaluate_response(
        response="The capital of France is Paris.",
        ground_truth="Paris is the capital of France.",
        retrieved_context="Noise that should not be used in ground_truth mode.",
        model=model,
        eval_context_mode="ground_truth",
    )
    assert model.last_pair[0] == "Paris is the capital of France."


def test_combined_mode_includes_retrieved_context():
    model = SpyModel()
    evaluate_response(
        response="The capital of France is Paris.",
        ground_truth="Paris is the capital of France.",
        retrieved_context="France has capital Paris.",
        model=model,
        eval_context_mode="combined",
    )
    assert "=== RETRIEVED FACTS ===" in model.last_pair[0]
    assert "France has capital Paris." in model.last_pair[0]


def test_turing_regression_wrong_temporal_org_not_rewarded_in_ground_truth_mode():
    model = PhraseContextModel()
    response = "Alan Turing worked for Government Communications Headquarters during World War II."
    ground_truth = (
        "During World War II, Alan Turing worked for the Government Code and Cypher School (GC&CS) at Bletchley Park."
    )
    retrieved = "P108 employer: Government Communications Headquarters"

    gt_result = evaluate_response(
        response=response,
        ground_truth=ground_truth,
        retrieved_context=retrieved,
        model=model,
        eval_context_mode="ground_truth",
        threshold=0.5,
    )
    combined_result = evaluate_response(
        response=response,
        ground_truth=ground_truth,
        retrieved_context=retrieved,
        model=model,
        eval_context_mode="combined",
        threshold=0.5,
    )

    assert gt_result["is_hallucination"] is True
    assert combined_result["is_hallucination"] is False


def test_rag_faithfulness_returns_none_without_context():
    model = SpyModel()
    result = evaluate_rag_faithfulness(
        response="Any response.",
        retrieved_context="",
        model=model,
    )
    assert result is None


def test_benchmark_files_define_deterministic_default_temperature():
    repo_root = Path(__file__).resolve().parents[1]
    runner_source = (repo_root / "kb_project/benchmark/runner.py").read_text(
        encoding="utf-8"
    )
    cli_source = (repo_root / "run_benchmark.py").read_text(encoding="utf-8")

    assert "benchmark_temperature: float = 0.0" in runner_source
    assert "default=0.0" in cli_source
