"""Simple evaluation helpers shared by benchmark evaluators."""

from __future__ import annotations

from typing import Any, Dict, Optional

VALID_EVAL_CONTEXT_MODES = {"ground_truth", "combined"}


def build_primary_context(
    ground_truth: str,
    retrieved_context: str,
    eval_context_mode: str = "ground_truth",
) -> str:
    """Build evaluation context. Ground truth is default."""
    mode = (eval_context_mode or "ground_truth").strip().lower()
    if mode not in VALID_EVAL_CONTEXT_MODES:
        mode = "ground_truth"

    if mode == "combined" and (retrieved_context or "").strip():
        return (
            "=== GROUND TRUTH ===\n"
            f"{ground_truth.strip()}\n\n"
            "=== RETRIEVED FACTS ===\n"
            f"{retrieved_context.strip()}"
        )

    return ground_truth.strip()


def evaluate_response(
    response: str,
    ground_truth: str,
    retrieved_context: str,
    model,
    threshold: float = 0.5,
    eval_context_mode: str = "ground_truth",
) -> Dict[str, Any]:
    """Evaluate a response against the selected reference context."""
    primary_context = build_primary_context(
        ground_truth=ground_truth,
        retrieved_context=retrieved_context,
        eval_context_mode=eval_context_mode,
    )

    score = model.predict([[primary_context, response]])[0]
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
    """Evaluate RAG answer faithfulness to retrieved context only."""
    context = (retrieved_context or "").strip()
    if not context:
        return None

    score = model.predict([[context, response]])[0]
    score_float = float(score.item() if hasattr(score, "item") else score)
    return {
        "score": score_float,
        "is_hallucination": score_float < threshold,
    }
