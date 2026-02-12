from __future__ import annotations

from kb_project.benchmark.models import TestCase
from kb_project.benchmark.runner import build_reference_ground_truth


def test_build_reference_ground_truth_returns_canonical_by_default():
    case = TestCase(
        id="t1",
        question="Q",
        ground_truth="Canonical answer.",
        category="test",
        accepted_aliases=[["Alpha", "A"]],
    )
    assert build_reference_ground_truth(case) == "Canonical answer."


def test_build_reference_ground_truth_can_include_aliases():
    case = TestCase(
        id="t2",
        question="Q",
        ground_truth="Canonical answer.",
        category="test",
        accepted_aliases=[["GC&CS", "GCHQ"], ["Alpha", "A", "A1"]],
    )
    reference = build_reference_ground_truth(case, include_aliases=True)

    assert "Canonical answer." in reference
    assert "Accepted equivalent wording:" in reference
    assert "GC&CS ~= GCHQ" in reference
    assert "Alpha ~= A, A1" in reference
