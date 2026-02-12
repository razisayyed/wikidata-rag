from __future__ import annotations

from kb_project.benchmark.runner import build_reference_ground_truth
from kb_project.benchmark.vectra import TestCase


def test_build_reference_ground_truth_includes_key_facts():
    case = TestCase(
        question="Q",
        ground_truth="Canonical answer.",
        key_facts=["Fact A", "Fact B"],
    )
    ref = build_reference_ground_truth(case)
    assert "Canonical answer." in ref
    assert "Key facts:" in ref
    assert "- Fact A" in ref
    assert "- Fact B" in ref
