from __future__ import annotations

from typing import Any

import equivalence_evaluator.nli as nli_module
from equivalence_evaluator.evaluator import EquivalenceEvaluator
from equivalence_evaluator.nli import evaluate_bidirectional_entailment
from equivalence_evaluator.triples import _parse_triple_payload
from kb_project.benchmark.models import ModelOutput
from kb_project.benchmark.runner import _evaluate_vectara


def _false_factual_judge(_ground_truth: str, _answer: str) -> dict[str, Any]:
    return {"equivalent": False, "score": 0.0, "details": {"reason": "stub"}}


def test_case_1_paraphrase_equivalent(monkeypatch):
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.cosine_similarity_with_details",
        lambda _a, _b: {
            "score": 0.95,
            "provider": "stub",
            "model": "stub",
            "error": "",
        },
    )
    evaluator = EquivalenceEvaluator(factual_judge_callable=_false_factual_judge)
    result = evaluator.evaluate(
        "Mars is known as the Red Planet.",
        "The Red Planet is known as Mars.",
    )
    assert result["equivalent"] is True
    assert result["method"] == "semantic_similarity"


def test_case_2_reorder_equivalent_via_nli(monkeypatch):
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.cosine_similarity_with_details",
        lambda _a, _b: {"score": 0.4, "provider": "stub", "model": "stub", "error": ""},
    )
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.evaluate_bidirectional_entailment",
        lambda _a, _b: {
            "entailment_forward": True,
            "entailment_backward": True,
            "confidence": 0.9,
            "backend": "stub",
            "error": "",
        },
    )
    evaluator = EquivalenceEvaluator(factual_judge_callable=_false_factual_judge)
    result = evaluator.evaluate(
        "Einstein was born in Germany.",
        "Germany is where Einstein was born.",
    )
    assert result["equivalent"] is True
    assert result["method"] == "nli"


def test_case_3_incorrect_not_equivalent(monkeypatch):
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.cosine_similarity_with_details",
        lambda _a, _b: {"score": 0.2, "provider": "stub", "model": "stub", "error": ""},
    )
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.evaluate_bidirectional_entailment",
        lambda _a, _b: {
            "entailment_forward": False,
            "entailment_backward": False,
            "confidence": 0.1,
            "backend": "stub",
            "error": "",
        },
    )
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.compare_triples",
        lambda _a, _b, verbose=False: {  # noqa: ARG005
            "equivalent": False,
            "gt_triples": [],
            "rag_triples": [],
            "matched": [],
            "overlap_ratio": 0.0,
            "identical": False,
            "error": "",
        },
    )
    evaluator = EquivalenceEvaluator(factual_judge_callable=_false_factual_judge)
    result = evaluator.evaluate(
        "Mars is the fourth planet.",
        "Mars is the largest planet.",
    )
    assert result["equivalent"] is False
    assert result["method"] == "factual_judge"


def test_step_order_short_circuit_exact_match(monkeypatch):
    called = {"similarity": 0, "nli": 0, "triples": 0}

    def _similarity(_a, _b):
        called["similarity"] += 1
        return {"score": 0.99, "provider": "stub", "model": "stub", "error": ""}

    def _nli(_a, _b):
        called["nli"] += 1
        return {"entailment_forward": True, "entailment_backward": True, "confidence": 1.0}

    def _triples(_a, _b, verbose=False):  # noqa: ARG001, ARG005
        called["triples"] += 1
        return {"equivalent": True, "overlap_ratio": 1.0}

    monkeypatch.setattr("equivalence_evaluator.evaluator.cosine_similarity_with_details", _similarity)
    monkeypatch.setattr("equivalence_evaluator.evaluator.evaluate_bidirectional_entailment", _nli)
    monkeypatch.setattr("equivalence_evaluator.evaluator.compare_triples", _triples)

    evaluator = EquivalenceEvaluator(factual_judge_callable=_false_factual_judge)
    result = evaluator.evaluate("Mars is known as the Red Planet.", "mars is known as the red planet")

    assert result["method"] == "exact_match"
    assert called["similarity"] == 0
    assert called["nli"] == 0
    assert called["triples"] == 0


def test_similarity_threshold_boundary_is_strict(monkeypatch):
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.cosine_similarity_with_details",
        lambda _a, _b: {"score": 0.92, "provider": "stub", "model": "stub", "error": ""},
    )
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.evaluate_bidirectional_entailment",
        lambda _a, _b: {
            "entailment_forward": False,
            "entailment_backward": False,
            "confidence": 0.0,
            "backend": "stub",
            "error": "",
        },
    )
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.compare_triples",
        lambda _a, _b, verbose=False: {  # noqa: ARG005
            "equivalent": False,
            "gt_triples": [],
            "rag_triples": [],
            "matched": [],
            "overlap_ratio": 0.0,
            "identical": False,
            "error": "",
        },
    )

    evaluator = EquivalenceEvaluator(factual_judge_callable=_false_factual_judge)
    result = evaluator.evaluate("A", "B")
    assert result["method"] == "factual_judge"
    assert result["equivalent"] is False


def test_nli_one_way_entailment_fails_bidirectional_rule(monkeypatch):
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.cosine_similarity_with_details",
        lambda _a, _b: {"score": 0.2, "provider": "stub", "model": "stub", "error": ""},
    )
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.evaluate_bidirectional_entailment",
        lambda _a, _b: {
            "entailment_forward": True,
            "entailment_backward": False,
            "confidence": 0.6,
            "backend": "stub",
            "error": "",
        },
    )
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.compare_triples",
        lambda _a, _b, verbose=False: {"equivalent": False, "overlap_ratio": 0.0, "error": ""},
    )

    evaluator = EquivalenceEvaluator(factual_judge_callable=_false_factual_judge)
    result = evaluator.evaluate("Einstein was born in Germany.", "Germany is where Einstein was born.")
    assert result["method"] == "factual_judge"
    assert result["equivalent"] is False


def test_triple_normalization_is_deterministic():
    parsed = _parse_triple_payload([["Mars,", "Nickname", "Red Planet!"]])
    assert ("mars", "nickname", "red planet") in parsed


def test_factual_judge_called_only_after_steps_1_to_4_fail(monkeypatch):
    calls = {"count": 0}

    def _judge(_gt, _answer):
        calls["count"] += 1
        return {"equivalent": False, "score": 0.0, "details": {"reason": "stub"}}

    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.cosine_similarity_with_details",
        lambda _a, _b: {"score": 0.1, "provider": "stub", "model": "stub", "error": ""},
    )
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.evaluate_bidirectional_entailment",
        lambda _a, _b: {
            "entailment_forward": False,
            "entailment_backward": False,
            "confidence": 0.0,
            "backend": "stub",
            "error": "",
        },
    )
    monkeypatch.setattr(
        "equivalence_evaluator.evaluator.compare_triples",
        lambda _a, _b, verbose=False: {"equivalent": False, "overlap_ratio": 0.0, "error": ""},
    )

    evaluator = EquivalenceEvaluator(factual_judge_callable=_judge)
    evaluator.evaluate("A", "B")
    assert calls["count"] == 1


def test_nli_backend_fallback_returns_deterministic_fail(monkeypatch):
    monkeypatch.setattr(nli_module, "AutoTokenizer", None)
    monkeypatch.setattr(nli_module, "AutoModelForSequenceClassification", None)
    monkeypatch.setattr(nli_module, "torch", None)
    monkeypatch.setattr(nli_module, "ChatOpenAI", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = evaluate_bidirectional_entailment("A", "B")
    assert result["entailment_forward"] is False
    assert result["entailment_backward"] is False
    assert result["confidence"] == 0.0
    assert result["backend"] == "none"
    assert "error" in result and result["error"]


class _FakeEquivalenceEvaluator:
    def evaluate(self, ground_truth: str, rag_output: str, baseline_output: str | None = None) -> dict[str, Any]:
        if "Red Planet" in rag_output:
            return {"equivalent": True, "method": "semantic_similarity", "score": 0.97, "details": {}}
        if "largest planet" in rag_output:
            return {"equivalent": False, "method": "factual_judge", "score": 0.0, "details": {"reason": "incorrect"}}
        return {"equivalent": False, "method": "factual_judge", "score": 0.0, "details": {}}


def test_runner_uses_equivalence_first_and_writes_method_notes():
    result = _evaluate_vectara(
        rag_output=ModelOutput(response="The Red Planet is known as Mars."),
        baseline_output=ModelOutput(response="Mars is the largest planet."),
        reference_ground_truth="Mars is known as the Red Planet.",
        threshold=0.5,
        hallucination_model=None,
        equivalence_evaluator=_FakeEquivalenceEvaluator(),
    )
    assert result.status == "completed"
    assert result.rag_label == "factual"
    assert result.baseline_label == "hallucinated"
    assert "RAG method=semantic_similarity" in result.notes
    assert "BASELINE method=factual_judge" in result.notes


def test_runner_preserves_factual_judge_fallback_labels():
    result = _evaluate_vectara(
        rag_output=ModelOutput(response="Mars is the largest planet."),
        baseline_output=ModelOutput(response="Mars is the largest planet."),
        reference_ground_truth="Mars is the fourth planet.",
        threshold=0.5,
        hallucination_model=None,
        equivalence_evaluator=_FakeEquivalenceEvaluator(),
    )
    assert result.status == "completed"
    assert result.rag_label == "hallucinated"
    assert result.baseline_label == "hallucinated"
    assert result.winner == "Tie"

