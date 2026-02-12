from __future__ import annotations

from kb_project.benchmark.runner import build_reference_ground_truth
from kb_project.benchmark.vectra import TestCase


def test_build_reference_ground_truth_defaults_to_concise():
    case = TestCase(
        question="Q",
        ground_truth="Canonical answer.",
        key_facts=["Fact A", "Fact B"],
    )
    ref = build_reference_ground_truth(case)
    assert ref == "Canonical answer."


def test_build_reference_ground_truth_rich_includes_key_facts():
    case = TestCase(
        question="Q",
        ground_truth="Canonical answer.",
        key_facts=["Fact A", "Fact B"],
    )
    ref = build_reference_ground_truth(case, ground_truth_style="rich")
    assert "Canonical answer." in ref
    assert "Key facts:" in ref
    assert "- Fact A" in ref
    assert "- Fact B" in ref


def test_build_reference_ground_truth_rich_respects_fact_cap():
    case = TestCase(
        question="Q",
        ground_truth="Canonical answer.",
        key_facts=["Fact A", "Fact B", "Fact C"],
    )
    ref = build_reference_ground_truth(
        case,
        ground_truth_style="rich",
        max_ground_truth_facts=1,
    )
    assert "- Fact A" in ref
    assert "- Fact B" not in ref
    assert "- Fact C" not in ref
