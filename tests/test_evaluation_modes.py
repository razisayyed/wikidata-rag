from __future__ import annotations

from pathlib import Path

from kb_project.benchmark.evaluation import build_primary_context, evaluate_response


class SpyModel:
    def __init__(self, score: float = 0.9):
        self.score = score
        self.last_pair = None

    def predict(self, pairs):
        self.last_pair = pairs[0]
        return [self.score]


def test_ground_truth_mode_ignores_retrieved_context():
    model = SpyModel()
    evaluate_response(
        response="The capital of France is Paris.",
        ground_truth="Paris is the capital of France.",
        retrieved_context="noise",
        model=model,
        eval_context_mode="ground_truth",
    )
    assert model.last_pair[0] == "Paris is the capital of France."


def test_combined_mode_includes_retrieved_context():
    context = build_primary_context(
        ground_truth="Paris is the capital of France.",
        retrieved_context="France has capital Paris.",
        eval_context_mode="combined",
    )
    assert "=== GROUND TRUTH ===" in context
    assert "=== RETRIEVED FACTS ===" in context
    assert "France has capital Paris." in context


def test_retrieved_only_mode_uses_only_retrieved_context():
    context = build_primary_context(
        ground_truth="Paris is the capital of France.",
        retrieved_context="France has capital Paris.",
        eval_context_mode="retrieved_only",
    )
    assert context == "France has capital Paris."


def test_benchmark_defaults_remain_deterministic():
    repo_root = Path(__file__).resolve().parents[1]
    runner_source = (repo_root / "kb_project/benchmark/runner.py").read_text(
        encoding="utf-8"
    )
    cli_source = (repo_root / "run_benchmark.py").read_text(encoding="utf-8")

    assert "temperature: float = 0.0" in runner_source
    assert "default=0.0" in cli_source


def test_legacy_runner_does_not_expose_dual_track_switches():
    repo_root = Path(__file__).resolve().parents[1]
    runner_source = (repo_root / "kb_project/benchmark/runner.py").read_text(
        encoding="utf-8"
    )
    assert "benchmark_axis" not in runner_source
    assert "factual_mode" not in runner_source
    assert "diagnostic_mode" not in runner_source
