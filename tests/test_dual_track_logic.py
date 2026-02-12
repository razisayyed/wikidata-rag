from __future__ import annotations

import pytest

from kb_project.benchmark.ragtruth import (
    RAGTruthResult,
    interpret_ragtruth_for_dual_track,
)
from kb_project.benchmark.runner import (
    _consensus_factual,
    _has_grounding_evidence,
    _is_omission_only_signal,
)


def test_consensus_factual_majority_vote():
    consensus, disagreement = _consensus_factual(
        vectara_vote=True,
        llm_vote=False,
        ragtruth_vote=True,
    )
    assert consensus is True
    assert disagreement == pytest.approx(1 / 3)


def test_consensus_factual_tie_resolves_via_llm_vote():
    consensus, disagreement = _consensus_factual(
        vectara_vote=True,
        llm_vote=False,
        ragtruth_vote=None,
    )
    assert consensus is False
    assert disagreement == pytest.approx(0.5)


def test_consensus_factual_no_votes_returns_none():
    consensus, disagreement = _consensus_factual(
        vectara_vote=None,
        llm_vote=None,
        ragtruth_vote=None,
    )
    assert consensus is None
    assert disagreement is None


def test_omission_signal_detected_without_marking_contradiction():
    assert _is_omission_only_signal("The answer omits the requested atomic counts.")
    assert not _is_omission_only_signal("The answer is false and contradicts the source.")


def test_ragtruth_omission_maps_to_partial_not_factual_error():
    result = RAGTruthResult(
        has_hallucination=True,
        hallucination_score=0.4,
        analysis="The response omits the required birth date.",
    )
    interpreted = interpret_ragtruth_for_dual_track(result)
    assert interpreted["has_factual_error"] is False
    assert interpreted["completeness"] == "partial"


def test_ragtruth_contradiction_maps_to_factual_error():
    result = RAGTruthResult(
        has_hallucination=True,
        hallucination_score=0.9,
        analysis="The response contains a false claim that contradicts the source.",
    )
    interpreted = interpret_ragtruth_for_dual_track(result)
    assert interpreted["has_factual_error"] is True
    assert interpreted["completeness"] == "insufficient"


def test_grounding_evidence_detection_unavailable_for_no_candidates():
    assert not _has_grounding_evidence(
        "[Tool: search_entity_candidates]\nNO CANDIDATES FOUND for Unknown Entity"
    )
    assert _has_grounding_evidence(
        "[Tool: fetch_entity_properties]\nP31: instance of - human"
    )
