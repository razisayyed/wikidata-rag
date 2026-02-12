"""
Benchmark Module
================
Comparison Test Suite: RAG vs Prompt-Only

Exports benchmark APIs via lazy attribute loading to avoid importing optional
runtime dependencies unless they are actually used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    # Models
    "Colors": ("kb_project.benchmark.models", "Colors"),
    "ComparisonResult": ("kb_project.benchmark.models", "ComparisonResult"),
    "TestCase": ("kb_project.benchmark.vectra", "TestCase"),
    "JudgeResult": ("kb_project.benchmark.llm_judge", "JudgeResult"),
    "RAGTruthResult": ("kb_project.benchmark.ragtruth", "RAGTruthResult"),
    "HallucinatedSpan": ("kb_project.benchmark.ragtruth", "HallucinatedSpan"),
    "load_ragtruth_qa_cases": (
        "kb_project.benchmark.ragtruth_dataset",
        "load_ragtruth_qa_cases",
    ),
    "ensure_ragtruth_files": (
        "kb_project.benchmark.ragtruth_dataset",
        "ensure_ragtruth_files",
    ),
    # Evaluation
    "evaluate_response": ("kb_project.benchmark.evaluation", "evaluate_response"),
    "evaluate_rag_faithfulness": (
        "kb_project.benchmark.evaluation",
        "evaluate_rag_faithfulness",
    ),
    "build_primary_context": ("kb_project.benchmark.evaluation", "build_primary_context"),
    "load_hallucination_model": (
        "kb_project.benchmark.vectra",
        "load_hallucination_model",
    ),
    # Runner
    "test_rag_model": ("kb_project.benchmark.runner", "test_rag_model"),
    "test_prompt_only_model": ("kb_project.benchmark.runner", "test_prompt_only_model"),
    "test_both_models": ("kb_project.benchmark.runner", "test_both_models"),
    "run_comparison_suite": ("kb_project.benchmark.runner", "run_comparison_suite"),
    "run_agent_with_capture": ("kb_project.benchmark.vectra", "run_agent_with_capture"),
    # Reporting
    "generate_comparison_table": (
        "kb_project.benchmark.reporting",
        "generate_comparison_table",
    ),
    "generate_markdown_table": ("kb_project.benchmark.reporting", "generate_markdown_table"),
    "generate_summary_stats": ("kb_project.benchmark.reporting", "generate_summary_stats"),
    "generate_full_report": ("kb_project.benchmark.reporting", "generate_full_report"),
    "save_benchmark_report": ("kb_project.benchmark.reporting", "save_benchmark_report"),
    # RAGTruth
    "RAGTruthEvaluator": ("kb_project.benchmark.ragtruth", "RAGTruthEvaluator"),
    # LLM Judge
    "judge_responses": ("kb_project.benchmark.llm_judge", "judge_responses"),
    "format_judge_result_detailed": (
        "kb_project.benchmark.llm_judge",
        "format_judge_result_detailed",
    ),
    # AIMon
    "AimonEvaluator": ("kb_project.benchmark.aimon", "AimonEvaluator"),
    "AimonResult": ("kb_project.benchmark.aimon", "AimonResult"),
    "HallucinatedSentence": ("kb_project.benchmark.aimon", "HallucinatedSentence"),
    "load_aimon_model": ("kb_project.benchmark.aimon", "load_aimon_model"),
    "evaluate_with_aimon": ("kb_project.benchmark.aimon", "evaluate_with_aimon"),
    "format_aimon_result": ("kb_project.benchmark.aimon", "format_aimon_result"),
    # Test Cases
    "GROUND_TRUTH_TEST_CASES": ("kb_project.benchmark.vectra", "GROUND_TRUTH_TEST_CASES"),
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
