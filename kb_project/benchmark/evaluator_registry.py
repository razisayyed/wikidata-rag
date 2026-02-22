"""Shared evaluator identifiers and display metadata for benchmark reporting."""

from __future__ import annotations

from typing import Dict, List, Set

HEAD_TO_HEAD_EVALUATOR_ORDER: List[str] = [
    "vectara",
    "vectara_hhem",
    "aimon",
    "llm_judge",
    "ragtruth",
]
RAG_ONLY_EVALUATOR_ORDER: List[str] = ["rag_retrieval_faithfulness"]
ALL_EVALUATOR_ORDER: List[str] = [
    *HEAD_TO_HEAD_EVALUATOR_ORDER,
    *RAG_ONLY_EVALUATOR_ORDER,
]

RAG_ONLY_EVALUATORS: Set[str] = set(RAG_ONLY_EVALUATOR_ORDER)
DISABLED_BY_DEFAULT_EVALUATORS: Set[str] = {"vectara", "ragtruth"}
DEFAULT_ENABLED_EVALUATOR_ORDER: List[str] = [
    evaluator for evaluator in ALL_EVALUATOR_ORDER if evaluator not in DISABLED_BY_DEFAULT_EVALUATORS
]

EVALUATOR_DISPLAY_NAMES: Dict[str, str] = {
    "vectara": "Ground-Truth Equivalence",
    "vectara_hhem": "Vectara HHEM (Ground-Truth Context)",
    "aimon": "Ground-Truth Hallucination Severity (AIMon)",
    "llm_judge": "LLM Judge (Ground-Truth Reference)",
    "ragtruth": "Ground-Truth Grounding (RAGTruth-style)",
    "rag_retrieval_faithfulness": "Retrieved-Context Faithfulness (RAG Only)",
}


def normalize_enabled_evaluators(enabled_evaluators: List[str] | None) -> List[str]:
    """Normalize evaluator ids to known order with duplicates removed."""
    if not enabled_evaluators:
        requested = set(DEFAULT_ENABLED_EVALUATOR_ORDER)
    else:
        requested = {str(item).strip() for item in enabled_evaluators if str(item).strip()}

    return [evaluator for evaluator in ALL_EVALUATOR_ORDER if evaluator in requested]
