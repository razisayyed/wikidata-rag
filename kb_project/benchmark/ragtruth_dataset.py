"""
RAGTruth Dataset Integration
============================
Utilities to download and load the public RAGTruth dataset for benchmarking.

We use the QA subset so our existing evaluation pipeline (question + ground
truth context) can run against more than the small hand-written set of cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

import requests

from .vectra import TestCase

RAGTRUTH_URLS: Dict[str, str] = {
    "response.jsonl": "https://raw.githubusercontent.com/CodingLL/RAGTruth/main/dataset/response.jsonl",
    "source_info.jsonl": "https://raw.githubusercontent.com/CodingLL/RAGTruth/main/dataset/source_info.jsonl",
}

DEFAULT_CACHE_DIR = Path("data/ragtruth")


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def ensure_ragtruth_files(cache_dir: Path = DEFAULT_CACHE_DIR) -> Dict[str, Path]:
    """Download RAGTruth dataset files if they are not already cached."""
    paths: Dict[str, Path] = {}
    for filename, url in RAGTRUTH_URLS.items():
        path = cache_dir / filename
        if not path.exists():
            _download_file(url, path)
        paths[filename] = path
    return paths


def _load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _source_ids_for_split(
    responses_path: Path, split: str, quality_filter: Set[str] | None = None
) -> Set[str]:
    """Collect source_ids for the desired split (train/test) and quality."""
    if quality_filter is None:
        quality_filter = {"good", "ok", "excellent"}
    ids: Set[str] = set()
    for row in _load_jsonl(responses_path):
        if row.get("split") != split:
            continue
        quality = row.get("quality", "").lower()
        if quality_filter and quality and quality not in quality_filter:
            continue
        source_id = row.get("source_id")
        if source_id:
            ids.add(str(source_id))
    return ids


def _extract_question(source_info: Dict) -> str:
    if isinstance(source_info, dict):
        return str(source_info.get("question", "")).strip()
    return ""


def _extract_context(source_info: Dict) -> str:
    """Flatten the QA passages/context into a single string."""
    if isinstance(source_info, dict):
        if "passages" in source_info:
            return str(source_info["passages"]).strip()
        if "context" in source_info:
            return str(source_info["context"]).strip()
        return json.dumps(source_info, ensure_ascii=False)
    return str(source_info)


def load_ragtruth_qa_cases(
    split: str = "test",
    limit: int | None = 50,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> List[TestCase]:
    """
    Load QA-style RAGTruth cases as TestCase objects.

    Args:
        split: "train" or "test" split from the dataset.
        limit: Maximum number of cases to return (None = all).
        cache_dir: Where to cache downloaded dataset files.
    """
    paths = ensure_ragtruth_files(cache_dir)
    allowed_ids = _source_ids_for_split(paths["response.jsonl"], split)

    cases: List[TestCase] = []
    for row in _load_jsonl(paths["source_info.jsonl"]):
        if row.get("task_type", "").lower() != "qa":
            continue

        source_id = str(row.get("source_id"))
        if allowed_ids and source_id not in allowed_ids:
            continue

        question = _extract_question(row.get("source_info", {}))
        context = _extract_context(row.get("source_info", {}))
        if not question or not context:
            continue

        desc = f"RAGTruth QA ({row.get('source', 'unknown source')})"
        cases.append(
            TestCase(
                question=question,
                ground_truth=context,
                description=desc,
            )
        )

        if limit is not None and len(cases) >= limit:
            break

    return cases
