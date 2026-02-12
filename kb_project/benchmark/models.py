"""Minimal data models for the legacy-simple benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

AIMON_WINNER_EPSILON = 0.05
ANALYSIS_VERSION = "v1_legacy_simple"


class Colors:
    """ANSI color codes for terminal output."""

    MAGENTA = "\033[95m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


@dataclass
class TestCase:
    """Single benchmark test case."""

    __test__ = False

    id: str
    question: str
    ground_truth: str
    category: str
    refusal_expected: bool = False
    accepted_aliases: List[List[str]] = field(default_factory=list)


@dataclass
class ModelOutput:
    """Model output captured for a case."""

    response: str
    retrieved_context: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluatorResult:
    """Normalized output for one evaluator."""

    name: str
    status: str  # completed | skipped | error
    rag_label: str = "error"  # factual | hallucinated | skipped | error
    baseline_label: str = "error"
    rag_score: Optional[float] = None
    baseline_score: Optional[float] = None
    winner: str = "N/A"  # RAG | BASELINE | Tie | N/A
    notes: str = ""


@dataclass
class CaseResult:
    """Complete benchmark result for one case."""

    test_case: TestCase
    rag_output: ModelOutput
    baseline_output: ModelOutput
    evaluations: Dict[str, EvaluatorResult]


@dataclass
class SuiteResult:
    """Complete benchmark suite result."""

    analysis_version: str
    threshold: float
    temperature: float
    cases: List[CaseResult]
    evaluator_summary: Dict[str, Dict[str, int]]


# Backward-compat alias for old code paths that still refer to ComparisonResult.
ComparisonResult = CaseResult


def label_from_hallucination_flag(is_hallucination: bool) -> str:
    return "hallucinated" if is_hallucination else "factual"


def winner_from_labels(
    rag_label: str,
    baseline_label: str,
    rag_score: Optional[float] = None,
    baseline_score: Optional[float] = None,
    lower_is_better: bool = False,
    epsilon: float = 0.0,
) -> str:
    """Resolve per-evaluator winner from labels and optional scores."""
    if rag_label not in {"factual", "hallucinated"}:
        return "N/A"
    if baseline_label not in {"factual", "hallucinated"}:
        return "N/A"

    if rag_label == "factual" and baseline_label == "hallucinated":
        return "RAG"
    if rag_label == "hallucinated" and baseline_label == "factual":
        return "BASELINE"

    if rag_score is None or baseline_score is None:
        return "Tie"

    diff = abs(rag_score - baseline_score)
    if diff <= epsilon:
        return "Tie"

    if lower_is_better:
        return "RAG" if rag_score < baseline_score else "BASELINE"
    return "RAG" if rag_score > baseline_score else "BASELINE"
