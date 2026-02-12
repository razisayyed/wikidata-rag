"""Benchmark module exports with lazy loading."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    # Models
    "Colors": ("kb_project.benchmark.models", "Colors"),
    "TestCase": ("kb_project.benchmark.models", "TestCase"),
    "ModelOutput": ("kb_project.benchmark.models", "ModelOutput"),
    "EvaluatorResult": ("kb_project.benchmark.models", "EvaluatorResult"),
    "CaseResult": ("kb_project.benchmark.models", "CaseResult"),
    "SuiteResult": ("kb_project.benchmark.models", "SuiteResult"),
    "ComparisonResult": ("kb_project.benchmark.models", "ComparisonResult"),
    # Runner
    "build_reference_ground_truth": (
        "kb_project.benchmark.runner",
        "build_reference_ground_truth",
    ),
    "run_comparison_suite": ("kb_project.benchmark.runner", "run_comparison_suite"),
    # Reporting
    "generate_comparison_table": (
        "kb_project.benchmark.reporting",
        "generate_comparison_table",
    ),
    "generate_markdown_table": (
        "kb_project.benchmark.reporting",
        "generate_markdown_table",
    ),
    "generate_summary_stats": (
        "kb_project.benchmark.reporting",
        "generate_summary_stats",
    ),
    "generate_full_report": ("kb_project.benchmark.reporting", "generate_full_report"),
    "generate_json_payload": (
        "kb_project.benchmark.reporting",
        "generate_json_payload",
    ),
    "save_benchmark_report": (
        "kb_project.benchmark.reporting",
        "save_benchmark_report",
    ),
    # Vectara/cases
    "load_hallucination_model": (
        "kb_project.benchmark.vectra",
        "load_hallucination_model",
    ),
    "run_agent_with_capture": ("kb_project.benchmark.vectra", "run_agent_with_capture"),
    "GROUND_TRUTH_TEST_CASES": (
        "kb_project.benchmark.vectra",
        "GROUND_TRUTH_TEST_CASES",
    ),
    # Optional evaluators
    "RAGTruthEvaluator": ("kb_project.benchmark.ragtruth", "RAGTruthEvaluator"),
    "AimonEvaluator": ("kb_project.benchmark.aimon", "AimonEvaluator"),
    "judge_responses": ("kb_project.benchmark.llm_judge", "judge_responses"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'kb_project.benchmark' has no attribute '{name}'")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MAP.keys())
