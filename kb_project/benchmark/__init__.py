"""
Benchmark Module
================
Comparison Test Suite: RAG vs Prompt-Only

Compares hallucination rates between:
  1. Wikidata RAG Agent (with knowledge retrieval)
  2. Prompt-Only Agent (no RAG, just anti-hallucination prompt)

Uses multiple evaluation methods:
  - Vectara hallucination evaluation model
  - LLM-as-a-Judge (GPT-4o)
  - RAGTruth-style span-level hallucination detection
"""

# Import public API from submodules
from .models import Colors, ComparisonResult
from .evaluation import evaluate_response
from .reporting import (
    generate_comparison_table,
    generate_markdown_table,
    generate_summary_stats,
    generate_full_report,
    save_benchmark_report,
)
from .runner import (
    test_rag_model,
    test_prompt_only_model,
    test_both_models,
    run_comparison_suite,
)
from .vectra import (
    GROUND_TRUTH_TEST_CASES,
    TestCase,
    load_hallucination_model,
    run_agent_with_capture,
)
from .ragtruth import (
    RAGTruthEvaluator,
    RAGTruthResult,
    HallucinatedSpan,
)
from .ragtruth_dataset import (
    load_ragtruth_qa_cases,
    ensure_ragtruth_files,
)
from .llm_judge import (
    JudgeResult,
    judge_responses,
    format_judge_result_detailed,
)
from .aimon import (
    AimonEvaluator,
    AimonResult,
    HallucinatedSentence,
    load_aimon_model,
    evaluate_with_aimon,
    format_aimon_result,
)

__all__ = [
    # Models
    "Colors",
    "ComparisonResult",
    "TestCase",
    "JudgeResult",
    "RAGTruthResult",
    "HallucinatedSpan",
    "load_ragtruth_qa_cases",
    "ensure_ragtruth_files",
    # Evaluation
    "evaluate_response",
    "load_hallucination_model",
    # Runner
    "test_rag_model",
    "test_prompt_only_model",
    "test_both_models",
    "run_comparison_suite",
    "run_agent_with_capture",
    # Reporting
    "generate_comparison_table",
    "generate_markdown_table",
    "generate_summary_stats",
    "generate_full_report",
    "save_benchmark_report",
    # RAGTruth
    "RAGTruthEvaluator",
    # LLM Judge
    "judge_responses",
    "format_judge_result_detailed",
    # AIMon
    "AimonEvaluator",
    "AimonResult",
    "HallucinatedSentence",
    "load_aimon_model",
    "evaluate_with_aimon",
    "format_aimon_result",
    # Test Cases
    "GROUND_TRUTH_TEST_CASES",
]
