"""Main semantic equivalence orchestrator."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .config import NLI_CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD
from .nli import evaluate_bidirectional_entailment
from .similarity import cosine_similarity_with_details
from .triples import compare_triples
from .utils import factual_judge_adapter, normalize_text

_EQUIVALENCE_METHODS = {
    "exact_match",
    "semantic_similarity",
    "nli",
    "triple_match",
}


class EquivalenceEvaluator:
    """Stepwise semantic equivalence evaluator with factual-judge fallback."""

    def __init__(
        self,
        factual_judge_callable: Optional[Callable[[str, str], dict[str, Any]]] = None,
        verbose: bool = False,
    ) -> None:
        self.factual_judge_callable = factual_judge_callable or factual_judge_adapter
        self.verbose = bool(verbose)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def evaluate(
        self,
        ground_truth: str,
        rag_output: str,
        baseline_output: str | None = None,
    ) -> dict[str, Any]:
        gt = ground_truth or ""
        rag = rag_output or ""

        base_details: dict[str, Any] = {}
        if baseline_output is not None:
            base_details["baseline_output"] = baseline_output

        # Step 1 — exact match
        if normalize_text(gt) == normalize_text(rag):
            self._log("[Equivalence] Step1 exact match -> PASS")
            return {
                "equivalent": True,
                "method": "exact_match",
                "score": 1.0,
                "details": {
                    **base_details,
                    "normalized_ground_truth": normalize_text(gt),
                    "normalized_rag_output": normalize_text(rag),
                },
            }
        self._log("[Equivalence] Step1 exact match -> FAIL")

        # Step 2 — semantic similarity
        sim = cosine_similarity_with_details(gt, rag)
        sim_score = float(sim.get("score", 0.0) or 0.0)
        sim_pass = sim_score >= SIMILARITY_THRESHOLD
        self._log(
            f"[Equivalence] Step2 similarity score: {sim_score:.4f} "
            f"(threshold>={SIMILARITY_THRESHOLD:.2f}) -> {'PASS' if sim_pass else 'FAIL'}"
        )
        if sim_pass:
            return {
                "equivalent": True,
                "method": "semantic_similarity",
                "score": 1.0,
                "details": {
                    **base_details,
                    "provider": sim.get("provider", ""),
                    "model": sim.get("model", ""),
                    "raw_similarity_score": sim_score,
                    "threshold": SIMILARITY_THRESHOLD,
                },
            }

        # Step 3 — bidirectional entailment
        nli = evaluate_bidirectional_entailment(gt, rag)
        nli_confidence = float(nli.get("confidence", 0.0) or 0.0)
        nli_pass = (
            bool(nli.get("entailment_forward"))
            and bool(nli.get("entailment_backward"))
            and nli_confidence >= NLI_CONFIDENCE_THRESHOLD
        )
        self._log(
            f"[Equivalence] Step3 NLI -> {'PASS' if nli_pass else 'FAIL'} "
            f"(backend={nli.get('backend', 'unknown')}, "
            f"confidence={nli_confidence:.4f}, "
            f"threshold>={NLI_CONFIDENCE_THRESHOLD:.2f})"
        )
        if nli_pass:
            return {
                "equivalent": True,
                "method": "nli",
                "score": 1.0,
                "details": {
                    **base_details,
                    **nli,
                    "raw_nli_confidence": nli_confidence,
                    "confidence_threshold": NLI_CONFIDENCE_THRESHOLD,
                },
            }

        # Step 4 — triple overlap
        triple = compare_triples(gt, rag, verbose=self.verbose)
        triple_pass = bool(triple.get("equivalent"))
        self._log(
            f"[Equivalence] Step4 triple_match -> {'PASS' if triple_pass else 'FAIL'}"
        )
        if triple_pass:
            return {
                "equivalent": True,
                "method": "triple_match",
                "score": (
                    1.0
                    if bool(triple.get("identical"))
                    else float(triple.get("overlap_ratio", 0.0) or 0.0)
                ),
                "details": {**base_details, **triple},
            }

        # Step 5 — factual judge fallback only
        factual = self.factual_judge_callable(gt, rag)
        equivalent = bool(factual.get("equivalent", False))
        score = float(factual.get("score", 1.0 if equivalent else 0.0) or 0.0)
        self._log(
            f"[Equivalence] Step5 factual_judge -> {'PASS' if equivalent else 'FAIL'}"
        )
        return {
            "equivalent": equivalent,
            "method": "factual_judge",
            "score": score,
            "details": {
                **base_details,
                "factual_judge_details": factual.get("details", {}),
                "prior_failures": {
                    "step2_similarity": sim,
                    "step3_nli": nli,
                    "step4_triples": triple,
                },
            },
        }


def is_equivalence_method(method: str) -> bool:
    """Return whether method indicates pre-judge equivalence pass."""
    return (method or "").strip() in _EQUIVALENCE_METHODS
