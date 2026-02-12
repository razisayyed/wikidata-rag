"""
AIMon Hallucination Detection Model (HDM-2-3B)
==============================================

Implements hallucination detection using AIMon Labs' HDM-2 model.
HDM-2 is a state-of-the-art hallucination detection model that provides:
- Context-based hallucination evaluations
- Common knowledge contradiction detection
- Phrase, token, and sentence-level hallucination identification
- Token-level probability scores

References:
- Model: https://huggingface.co/AimonLabs/hallucination-detection-model
- Paper: https://arxiv.org/abs/2504.07069
- GitHub: https://github.com/aimonlabs/hallucination-detection-model
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..settings import AIMON_DEVICE, resolve_device
from .evaluation import build_primary_context

# ==========================================================================
# Data Structures
# ==========================================================================


@dataclass
class HallucinatedSentence:
    """Represents a detected hallucinated sentence."""

    text: str
    probability: float
    is_common_knowledge: bool = False


@dataclass
class AimonResult:
    """Result from AIMon hallucination evaluation."""

    # Core results
    has_hallucination: bool
    hallucination_severity: float  # 0.0 = no hallucination, 1.0 = fully hallucinated
    hallucinated_sentences: List[HallucinatedSentence] = field(default_factory=list)

    # Additional info
    sentence_count: int = 0
    raw_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def case_label(self) -> str:
        """Binary label: hallucinated or not."""
        return "hallucinated" if self.has_hallucination else "factual"

    @property
    def hallucination_score(self) -> float:
        """Alias for severity to match other evaluator interfaces."""
        return self.hallucination_severity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_hallucination": self.has_hallucination,
            "hallucination_severity": self.hallucination_severity,
            "hallucination_score": self.hallucination_score,
            "sentence_count": self.sentence_count,
            "hallucinated_sentences": [
                {
                    "text": s.text,
                    "probability": s.probability,
                    "is_common_knowledge": s.is_common_knowledge,
                }
                for s in self.hallucinated_sentences
            ],
            "case_label": self.case_label,
            "error": self.error,
        }


# ==========================================================================
# AIMon Hallucination Evaluator
# ==========================================================================


class AimonEvaluator:
    """
    Evaluator using AIMon Labs' HDM-2-3B model.

    HDM-2 is designed for enterprise-grade hallucination detection with:
    - High accuracy on RagTruth, TruthfulQA, and HDM-Bench
    - Low latency (~200ms on L4 GPU)
    - Sentence-level hallucination identification
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize the AIMon evaluator.

        Args:
            threshold: Hallucination severity above this is flagged as hallucination.
                      Default 0.5 means if severity >= 0.5, it's considered hallucinated.
        """
        self.threshold = threshold
        self.model: Any = None
        self._loaded = False
        self.device = "cpu"

    def load_model(self) -> None:
        """Load the HDM-2 model from HuggingFace."""
        if self._loaded:
            return

        try:
            from hdm2 import HallucinationDetectionModel

            self.device = resolve_device(AIMON_DEVICE)
            print("Loading AIMon HDM-2-3B hallucination detection model...")
            kwargs: Dict[str, Any] = {}

            # Best-effort device argument mapping across hdm2 versions.
            sig = inspect.signature(HallucinationDetectionModel.__init__)
            if "device" in sig.parameters:
                kwargs["device"] = self.device
            elif "torch_device" in sig.parameters:
                kwargs["torch_device"] = self.device
            elif "model_device" in sig.parameters:
                kwargs["model_device"] = self.device

            self.model = HallucinationDetectionModel(**kwargs)

            # Secondary fallback if wrapper exposes a .to() method.
            if hasattr(self.model, "to"):
                try:
                    self.model = self.model.to(self.device)
                except Exception:
                    pass

            self._loaded = True
            print(f"AIMon model loaded (device: {self.device}).\n")
        except ImportError as e:
            raise ImportError(
                "hdm2 package not installed. Install with: pip install hdm2"
            ) from e

    def evaluate(
        self,
        prompt: str,
        context: str,
        response: str,
    ) -> AimonResult:
        """
        Evaluate a response for hallucinations.

        Args:
            prompt: The original question/prompt.
            context: Source context (ground truth + retrieved facts).
            response: The model's response to evaluate.

        Returns:
            AimonResult with hallucination detection results.
        """
        if not self._loaded:
            self.load_model()

        try:
            # Call HDM-2 model
            results = self.model.apply(prompt, context, response)

            # Extract hallucination severity
            severity = results.get("adjusted_hallucination_severity", 0.0)

            # Extract hallucinated sentences
            hallucinated_sentences: List[HallucinatedSentence] = []
            candidate_sentences = results.get("candidate_sentences", [])
            ck_results = results.get("ck_results", [])

            # Process common knowledge results to find hallucinations
            for sentence_result in ck_results:
                if sentence_result.get("prediction") == 1:  # 1 indicates hallucination
                    hallucinated_sentences.append(
                        HallucinatedSentence(
                            text=sentence_result.get("text", ""),
                            probability=sentence_result.get(
                                "hallucination_probability", 0.0
                            ),
                            is_common_knowledge=False,
                        )
                    )

            # Determine if response is hallucinated based on threshold
            has_hallucination = severity >= self.threshold

            return AimonResult(
                has_hallucination=has_hallucination,
                hallucination_severity=float(severity),
                hallucinated_sentences=hallucinated_sentences,
                sentence_count=len(candidate_sentences),
                raw_results=results,
            )

        except Exception as e:
            return AimonResult(
                has_hallucination=False,
                hallucination_severity=0.0,
                error=str(e),
            )

    def evaluate_response(
        self,
        question: str,
        ground_truth: str,
        retrieved_context: str,
        response: str,
        eval_context_mode: str = "ground_truth",
    ) -> AimonResult:
        """
        Evaluate a response using ground truth and retrieved context.

        This method matches the interface used by other evaluators in the benchmark.

        Args:
            question: The original question.
            ground_truth: Known correct answer.
            retrieved_context: Facts retrieved by RAG (empty for prompt-only).
            response: Model's response to evaluate.
            eval_context_mode: "ground_truth" (default) or "combined".

        Returns:
            AimonResult with hallucination detection results.
        """
        # Build combined context using standard utility
        combined_context = build_primary_context(
            ground_truth=ground_truth,
            retrieved_context=retrieved_context,
            eval_context_mode=eval_context_mode,
        )

        return self.evaluate(
            prompt=question,
            context=combined_context,
            response=response,
        )


# ==========================================================================
# Module-level convenience functions
# ==========================================================================

_evaluator: Optional[AimonEvaluator] = None


def load_aimon_model(threshold: float = 0.5) -> AimonEvaluator:
    """
    Load and return the AIMon evaluator singleton.

    Args:
        threshold: Hallucination severity threshold.

    Returns:
        Loaded AimonEvaluator instance.
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = AimonEvaluator(threshold=threshold)
        _evaluator.load_model()
    return _evaluator


def evaluate_with_aimon(
    question: str,
    ground_truth: str,
    retrieved_context: str,
    response: str,
    threshold: float = 0.5,
) -> AimonResult:
    """
    Convenience function to evaluate a response with AIMon.

    Args:
        question: The original question.
        ground_truth: Known correct answer.
        retrieved_context: Facts retrieved by RAG.
        response: Model's response to evaluate.
        threshold: Hallucination severity threshold.

    Returns:
        AimonResult with hallucination detection results.
    """
    evaluator = load_aimon_model(threshold=threshold)
    return evaluator.evaluate_response(
        question=question,
        ground_truth=ground_truth,
        retrieved_context=retrieved_context,
        response=response,
    )


def format_aimon_result(result: AimonResult) -> str:
    """
    Format AIMon result for display.

    Args:
        result: AimonResult to format.

    Returns:
        Formatted string representation.
    """
    lines = [
        "=" * 60,
        "AIMon HDM-2 Hallucination Detection Results",
        "=" * 60,
        f"Hallucination Severity: {result.hallucination_severity:.4f}",
        f"Is Hallucinated: {'Yes' if result.has_hallucination else 'No'}",
        f"Label: {result.case_label.upper()}",
    ]

    if result.error:
        lines.append(f"Error: {result.error}")

    if result.hallucinated_sentences:
        lines.append("")
        lines.append("Hallucinated Sentences:")
        for sent in result.hallucinated_sentences:
            ck_marker = " [CK]" if sent.is_common_knowledge else ""
            lines.append(f"  - {sent.text} (prob: {sent.probability:.4f}){ck_marker}")

    lines.append("=" * 60)
    return "\n".join(lines)
