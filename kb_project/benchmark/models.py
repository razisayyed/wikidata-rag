"""
Data Models for Benchmark Module
================================
Contains data structures used across the benchmark system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .aimon import AimonResult
from .llm_judge import JudgeResult
from .ragtruth import RAGTruthResult


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

    question: str
    description: str
    ground_truth: str

    # RAG model results
    rag_response: str
    rag_retrieved_context: str
    rag_score: float
    rag_is_hallucination: bool

    # Prompt-only model results
    prompt_only_response: str
    prompt_only_score: float
    prompt_only_is_hallucination: bool

    # LLM Judge results (optional, may be None if not run)
    llm_judge_result: Optional[JudgeResult] = None

    # RAGTruth results (optional, may be None if not run)
    rag_ragtruth_result: Optional[RAGTruthResult] = None
    prompt_only_ragtruth_result: Optional[RAGTruthResult] = None

    # AIMon results (optional, may be None if not run)
    rag_aimon_result: Optional[AimonResult] = None
    prompt_only_aimon_result: Optional[AimonResult] = None

    @property
    def winner(self) -> str:
        """Determine which model performed better (Vectara-based)."""
        if self.rag_is_hallucination and not self.prompt_only_is_hallucination:
            return "Prompt-Only"
        elif not self.rag_is_hallucination and self.prompt_only_is_hallucination:
            return "RAG"
        elif self.rag_score > self.prompt_only_score:
            return "RAG"
        elif self.prompt_only_score > self.rag_score:
            return "Prompt-Only"
        else:
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
        elif not rag_halluc and prompt_halluc:
            return "RAG"
        elif not rag_halluc and not prompt_halluc:
            # Both factual - compare by score (lower is better for hallucination score)
            if (
                self.rag_ragtruth_result.hallucination_score
                < self.prompt_only_ragtruth_result.hallucination_score
            ):
                return "RAG"
            elif (
                self.prompt_only_ragtruth_result.hallucination_score
                < self.rag_ragtruth_result.hallucination_score
            ):
                return "Prompt-Only"
            else:
                return "Tie"
        else:
            # Both hallucinated - compare by score (lower is better)
            if (
                self.rag_ragtruth_result.hallucination_score
                < self.prompt_only_ragtruth_result.hallucination_score
            ):
                return "RAG"
            elif (
                self.prompt_only_ragtruth_result.hallucination_score
                < self.rag_ragtruth_result.hallucination_score
            ):
                return "Prompt-Only"
            else:
                return "Tie"

    @property
    def aimon_winner(self) -> str:
        """Winner according to AIMon evaluation."""
        if self.rag_aimon_result is None or self.prompt_only_aimon_result is None:
            return "N/A"

        rag_halluc = self.rag_aimon_result.has_hallucination
        prompt_halluc = self.prompt_only_aimon_result.has_hallucination

        if rag_halluc and not prompt_halluc:
            return "Prompt-Only"
        elif not rag_halluc and prompt_halluc:
            return "RAG"
        elif not rag_halluc and not prompt_halluc:
            # Both factual - compare by score (lower is better for hallucination severity)
            if (
                self.rag_aimon_result.hallucination_severity
                < self.prompt_only_aimon_result.hallucination_severity
            ):
                return "RAG"
            elif (
                self.prompt_only_aimon_result.hallucination_severity
                < self.rag_aimon_result.hallucination_severity
            ):
                return "Prompt-Only"
            else:
                return "Tie"
        else:
            # Both hallucinated - compare by score (lower is better)
            if (
                self.rag_aimon_result.hallucination_severity
                < self.prompt_only_aimon_result.hallucination_severity
            ):
                return "RAG"
            elif (
                self.prompt_only_aimon_result.hallucination_severity
                < self.rag_aimon_result.hallucination_severity
            ):
                return "Prompt-Only"
            else:
                return "Tie"
