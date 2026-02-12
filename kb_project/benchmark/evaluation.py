"""
Evaluation Functions for Benchmark Module
==========================================
Contains functions for evaluating model responses.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

VALID_EVAL_CONTEXT_MODES = {"ground_truth", "combined"}


def build_primary_context(
    ground_truth: str,
    retrieved_context: str,
    eval_context_mode: str = "ground_truth",
) -> str:
    """
    Build the context used for primary factual-correctness scoring.

    Modes:
    - ground_truth: use curated reference text only.
    - combined: use ground truth + retrieved context (legacy behaviour).
    """
    mode = (eval_context_mode or "ground_truth").strip().lower()
    if mode not in VALID_EVAL_CONTEXT_MODES:
        mode = "ground_truth"

    if mode == "combined" and retrieved_context.strip():
        return f"""=== GROUND TRUTH ===
{ground_truth.strip()}

=== RETRIEVED FACTS ===
{retrieved_context.strip()}
"""

    return ground_truth.strip()


def evaluate_response(
    response: str,
    ground_truth: str,
    retrieved_context: str,
    model,
    threshold: float = 0.5,
    eval_context_mode: str = "ground_truth",
) -> Dict[str, Any]:
    """
    Evaluate factual correctness against the configured primary context.

    Args:
        response: Model's response.
        ground_truth: Known correct answer.
        retrieved_context: Facts retrieved (used only in combined mode).
        model: Vectara hallucination model.
        threshold: Score below this = hallucination.
        eval_context_mode: "ground_truth" (default) or "combined".

    Returns:
        Dict with score, is_hallucination, and interpretation.
    """
    primary_context = build_primary_context(
        ground_truth=ground_truth,
        retrieved_context=retrieved_context,
        eval_context_mode=eval_context_mode,
    )

    score = model.predict([[primary_context, response]])[0]
    # Convert to Python float (in case it's a tensor or numpy type)
    score_float = float(score.item() if hasattr(score, "item") else score)
    is_hallucination = score_float < threshold

    return {
        "score": score_float,
        "is_hallucination": is_hallucination,
        "context_mode": (eval_context_mode or "ground_truth").strip().lower(),
    }


def evaluate_rag_faithfulness(
    response: str,
    retrieved_context: str,
    model,
    threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate whether a RAG response is faithful to retrieved evidence only.

    Returns None when no retrieved context is available.
    """
    context = (retrieved_context or "").strip()
    if not context:
        return None

    score = model.predict([[context, response]])[0]
    score_float = float(score.item() if hasattr(score, "item") else score)
    is_hallucination = score_float < threshold

    return {
        "score": score_float,
        "is_hallucination": is_hallucination,
    }
