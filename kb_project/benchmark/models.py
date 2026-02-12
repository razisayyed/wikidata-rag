"""
Data Models for Benchmark Module
================================
Contains data structures used across the benchmark system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .aimon import AimonResult
    from .llm_judge import JudgeResult
    from .ragtruth import RAGTruthResult
else:
    AimonResult = Any
    JudgeResult = Any
    RAGTruthResult = Any

AIMON_WINNER_EPSILON = 0.05


# ==========================================================================
# Color Constants for Terminal Output
# ==========================================================================


class Colors:
    """ANSI color codes for terminal output"""

    MAGENTA = "\033[95m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ==========================================================================
# Data Structures
# ==========================================================================


@dataclass
class ComparisonResult:
    """Holds results from testing both models on the same question."""

    # Benchmark meta
    question: str
    description: str
    ground_truth: str
    analysis_version: str = "v2_dual_track"
    benchmark_axis: str = "dual_track"
    evaluation_mode: str = "ground_truth"
    factual_mode: str = "ground_truth"
    diagnostic_mode: str = "combined"

    # RAG model results
    rag_response: str = ""
    rag_retrieved_context: str = ""
    rag_score: float = 0.0
    rag_is_hallucination: bool = False

    # Prompt-only model results
    prompt_only_response: str = ""
    prompt_only_score: float = 0.0
    prompt_only_is_hallucination: bool = False

    # Secondary RAG-only grounding signal
    rag_faithfulness_score: Optional[float] = None
    rag_faithfulness_is_hallucination: Optional[bool] = None
    rag_grounding_status: str = "unavailable"  # faithful|non_faithful|unavailable
    rag_grounding_score: Optional[float] = None

    # Per-evaluator factual labels
    rag_factual_vectara: Optional[bool] = None
    rag_factual_llm: Optional[bool] = None
    rag_factual_ragtruth: Optional[bool] = None
    prompt_factual_vectara: Optional[bool] = None
    prompt_factual_llm: Optional[bool] = None
    prompt_factual_ragtruth: Optional[bool] = None

    # Consensus factual labels
    rag_factual_consensus: Optional[bool] = None
    prompt_factual_consensus: Optional[bool] = None

    # Completeness labels
    rag_completeness: str = "insufficient"
    prompt_completeness: str = "insufficient"

    # Disagreement rates
    rag_factual_disagreement_rate: Optional[float] = None
    prompt_factual_disagreement_rate: Optional[float] = None

    # LLM Judge results (optional, may be None if not run)
    llm_judge_result: Optional[JudgeResult] = None

    # RAGTruth results (optional, may be None if not run)
    rag_ragtruth_result: Optional[RAGTruthResult] = None
    prompt_only_ragtruth_result: Optional[RAGTruthResult] = None

    # AIMon results (optional, may be None if not run)
    rag_aimon_result: Optional[AimonResult] = None
    prompt_only_aimon_result: Optional[AimonResult] = None

    @property
    def factual_winner(self) -> str:
        """Winner according to factual-consensus track."""
        rag = self.rag_factual_consensus
        prompt = self.prompt_factual_consensus
        if rag is None or prompt is None:
            return "N/A"
        if rag and not prompt:
            return "RAG"
        if prompt and not rag:
            return "Prompt-Only"
        return "Tie"

    @property
    def winner(self) -> str:
        """Backward-compatible winner property (factual consensus first)."""
        if self.factual_winner != "N/A":
            if self.factual_winner != "Tie":
                return self.factual_winner
            if self.rag_score > self.prompt_only_score:
                return "RAG"
            if self.prompt_only_score > self.rag_score:
                return "Prompt-Only"
            return "Tie"

        # Legacy fallback
        if self.rag_is_hallucination and not self.prompt_only_is_hallucination:
            return "Prompt-Only"
        if not self.rag_is_hallucination and self.prompt_only_is_hallucination:
            return "RAG"
        if self.rag_score > self.prompt_only_score:
            return "RAG"
        if self.prompt_only_score > self.rag_score:
            return "Prompt-Only"
        return "Tie"

    @property
    def llm_judge_winner(self) -> str:
        """Winner according to LLM judge."""
        if self.llm_judge_result is None:
            return "N/A"
        if self.llm_judge_result.error:
            return "Error"
        return self.llm_judge_result.winner

    @property
    def ragtruth_winner(self) -> str:
        """Winner according to RAGTruth evaluation."""
        if self.rag_ragtruth_result is None or self.prompt_only_ragtruth_result is None:
            return "N/A"

        rag_halluc = self.rag_ragtruth_result.has_hallucination
        prompt_halluc = self.prompt_only_ragtruth_result.has_hallucination

        if rag_halluc and not prompt_halluc:
            return "Prompt-Only"
        if not rag_halluc and prompt_halluc:
            return "RAG"
        if not rag_halluc and not prompt_halluc:
            if (
                self.rag_ragtruth_result.hallucination_score
                < self.prompt_only_ragtruth_result.hallucination_score
            ):
                return "RAG"
            if (
                self.prompt_only_ragtruth_result.hallucination_score
                < self.rag_ragtruth_result.hallucination_score
            ):
                return "Prompt-Only"
            return "Tie"

        if (
            self.rag_ragtruth_result.hallucination_score
            < self.prompt_only_ragtruth_result.hallucination_score
        ):
            return "RAG"
        if (
            self.prompt_only_ragtruth_result.hallucination_score
            < self.rag_ragtruth_result.hallucination_score
        ):
            return "Prompt-Only"
        return "Tie"

    @property
    def aimon_winner(self) -> str:
        """Winner according to AIMon evaluation."""
        if self.rag_aimon_result is None or self.prompt_only_aimon_result is None:
            return "N/A"

        rag_halluc = self.rag_aimon_result.has_hallucination
        prompt_halluc = self.prompt_only_aimon_result.has_hallucination
        rag_severity = self.rag_aimon_result.hallucination_severity
        prompt_severity = self.prompt_only_aimon_result.hallucination_severity

        if rag_halluc and not prompt_halluc:
            return "Prompt-Only"
        if not rag_halluc and prompt_halluc:
            return "RAG"
        if not rag_halluc and not prompt_halluc:
            if abs(rag_severity - prompt_severity) <= AIMON_WINNER_EPSILON:
                return "Tie"
            if rag_severity < prompt_severity:
                return "RAG"
            if prompt_severity < rag_severity:
                return "Prompt-Only"
            return "Tie"

        if abs(rag_severity - prompt_severity) <= AIMON_WINNER_EPSILON:
            return "Tie"
        if rag_severity < prompt_severity:
            return "RAG"
        if prompt_severity < rag_severity:
            return "Prompt-Only"
        return "Tie"
