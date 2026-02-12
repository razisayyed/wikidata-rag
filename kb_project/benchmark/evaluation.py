"""
Evaluation Functions for Benchmark Module
==========================================
Contains functions for evaluating model responses.
"""

from __future__ import annotations

from typing import Any, Dict


def evaluate_response(
    response: str,
    ground_truth: str,
    retrieved_context: str,
    model,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate if response is grounded in the combined context.

    Args:
        response: Model's response.
        ground_truth: Known correct answer.
        retrieved_context: Facts retrieved (empty for prompt-only).
        model: Vectara hallucination model.
        threshold: Score below this = hallucination.

    Returns:
        Dict with score, is_hallucination, and interpretation.
    """
    # Build combined context
    if retrieved_context:
        combined_context = f"""=== GROUND TRUTH ===
{ground_truth.strip()}

=== RETRIEVED FACTS ===
{retrieved_context.strip()}
"""
    else:
        # For prompt-only, just use ground truth
        combined_context = ground_truth.strip()

    score = model.predict([[combined_context, response]])[0]
    # Convert to Python float (in case it's a tensor or numpy type)
    score_float = float(score.item() if hasattr(score, "item") else score)
    is_hallucination = score_float < threshold

    return {
        "score": score_float,
        "is_hallucination": is_hallucination,
    }
