from __future__ import annotations

import json
from pathlib import Path

from kb_project.benchmark.reporting import save_benchmark_report


class FakeResult:
    def __init__(self):
        self.question = "What is the capital of France?"
        self.description = "Simple geographic fact"
        self.ground_truth = "Paris is the capital of France."
        self.rag_response = "Paris is the capital of France."
        self.rag_retrieved_context = "[Tool: fetch_entity_properties]\nP36: capital - Paris"
        self.rag_score = 0.95
        self.rag_is_hallucination = False
        self.prompt_only_response = "Paris is the capital of France."
        self.prompt_only_score = 0.88
        self.prompt_only_is_hallucination = False
        self.evaluation_mode = "ground_truth"
        self.rag_faithfulness_score = 0.93
        self.rag_faithfulness_is_hallucination = False
        self.llm_judge_result = None
        self.rag_ragtruth_result = None
        self.prompt_only_ragtruth_result = None
        self.rag_aimon_result = None
        self.prompt_only_aimon_result = None

    @property
    def winner(self):
        return "RAG"

    @property
    def ragtruth_winner(self):
        return "N/A"

    @property
    def aimon_winner(self):
        return "N/A"


def test_save_benchmark_report_keeps_backward_compat_and_adds_new_fields(tmp_path: Path):
    json_path = tmp_path / "benchmark_results.json"
    md_path = tmp_path / "benchmark_report.md"

    save_benchmark_report([FakeResult()], json_path=str(json_path), md_path=str(md_path))

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert len(payload) == 1
    entry = payload[0]

    # Backward-compatible existing fields
    assert "rag" in entry
    assert "prompt_only" in entry
    assert "score" in entry["rag"]
    assert "score" in entry["prompt_only"]

    # Newly added additive fields
    assert entry["evaluation_mode"] == "ground_truth"
    assert "faithfulness_score" in entry["rag"]
    assert "faithfulness_is_hallucination" in entry["rag"]
