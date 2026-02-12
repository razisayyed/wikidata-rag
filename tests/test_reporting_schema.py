from __future__ import annotations

import json
from pathlib import Path

from kb_project.benchmark.models import (
    ANALYSIS_VERSION,
    CaseResult,
    EvaluatorResult,
    ModelOutput,
    SuiteResult,
    TestCase,
)
from kb_project.benchmark.reporting import save_benchmark_report


def _fake_suite() -> SuiteResult:
    case = CaseResult(
        test_case=TestCase(
            id="case_01",
            question="What is the capital of France?",
            ground_truth="Paris is the capital of France.",
            category="geography",
            refusal_expected=False,
        ),
        rag_output=ModelOutput(
            response="Paris is the capital of France.",
            retrieved_context="[Tool: fetch_entity_properties]\nP36: capital - Paris",
            tool_calls=[{"name": "fetch_entity_properties", "args": {}, "output": "P36"}],
        ),
        baseline_output=ModelOutput(response="Paris is the capital of France."),
        evaluations={
            "vectara": EvaluatorResult(
                name="vectara",
                status="completed",
                rag_label="factual",
                baseline_label="factual",
                rag_score=0.9,
                baseline_score=0.9,
                winner="Tie",
            ),
            "aimon": EvaluatorResult(
                name="aimon",
                status="completed",
                rag_label="factual",
                baseline_label="factual",
                rag_score=0.1,
                baseline_score=0.1,
                winner="Tie",
            ),
            "llm_judge": EvaluatorResult(
                name="llm_judge",
                status="skipped",
                rag_label="skipped",
                baseline_label="skipped",
                winner="N/A",
                notes="OPENAI_API_KEY is not set.",
            ),
            "ragtruth": EvaluatorResult(
                name="ragtruth",
                status="completed",
                rag_label="factual",
                baseline_label="factual",
                rag_score=0.0,
                baseline_score=0.0,
                winner="Tie",
            ),
        },
    )

    return SuiteResult(
        analysis_version=ANALYSIS_VERSION,
        threshold=0.5,
        temperature=0.0,
        cases=[case],
        evaluator_summary={
            "vectara": {
                "rag_wins": 0,
                "baseline_wins": 0,
                "ties": 1,
                "rag_factual": 1,
                "rag_hallucinated": 0,
                "baseline_factual": 1,
                "baseline_hallucinated": 0,
                "skipped": 0,
                "errors": 0,
            },
            "aimon": {
                "rag_wins": 0,
                "baseline_wins": 0,
                "ties": 1,
                "rag_factual": 1,
                "rag_hallucinated": 0,
                "baseline_factual": 1,
                "baseline_hallucinated": 0,
                "skipped": 0,
                "errors": 0,
            },
            "llm_judge": {
                "rag_wins": 0,
                "baseline_wins": 0,
                "ties": 0,
                "rag_factual": 0,
                "rag_hallucinated": 0,
                "baseline_factual": 0,
                "baseline_hallucinated": 0,
                "skipped": 1,
                "errors": 0,
            },
            "ragtruth": {
                "rag_wins": 0,
                "baseline_wins": 0,
                "ties": 1,
                "rag_factual": 1,
                "rag_hallucinated": 0,
                "baseline_factual": 1,
                "baseline_hallucinated": 0,
                "skipped": 0,
                "errors": 0,
            },
        },
    )


def test_save_benchmark_report_uses_simple_top_level_schema(tmp_path: Path):
    suite = _fake_suite()
    json_path = tmp_path / "benchmark_results.json"
    md_path = tmp_path / "benchmark_report.md"

    save_benchmark_report(suite, json_path=str(json_path), md_path=str(md_path))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["analysis_version"] == ANALYSIS_VERSION
    assert payload["config"]["threshold"] == 0.5
    assert payload["config"]["temperature"] == 0.0
    assert payload["config"]["total_cases"] == 1

    assert "evaluator_summary" in payload
    assert "head_to_head" in payload["evaluator_summary"]["vectara"]
    assert "factual_vs_hallucination" in payload["evaluator_summary"]["vectara"]

    assert len(payload["cases"]) == 1
    case_entry = payload["cases"][0]
    assert case_entry["id"] == "case_01"
    assert "evaluations" in case_entry
    assert set(case_entry["evaluations"].keys()) == {
        "vectara",
        "aimon",
        "llm_judge",
        "ragtruth",
    }

    report_text = md_path.read_text(encoding="utf-8")
    assert "## Head-to-Head by Evaluator" in report_text
    assert "## vectara Results" in report_text
    assert "## aimon Results" in report_text
    assert "## llm_judge Results" in report_text
    assert "## ragtruth Results" in report_text
