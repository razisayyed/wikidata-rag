"""Utility helpers for semantic equivalence evaluation."""

from __future__ import annotations

import json
import re
import string
from typing import Any

from kb_project.benchmark.llm_judge import call_openai_judge
from kb_project.settings import OPENAI_JUDGE_MODEL

from .config import TEMPERATURE

_TRAILING_PUNCT = set(".!?;:,")
_INNER_SPACES_PATTERN = re.compile(r"\s+")
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def strip_trailing_punctuation(text: str) -> str:
    """Remove trailing punctuation used for style-only variations."""
    value = (text or "").strip()
    while value and value[-1] in _TRAILING_PUNCT:
        value = value[:-1].rstrip()
    return value


def normalize_text(text: str) -> str:
    """Normalize plain text for deterministic exact matching."""
    lowered = (text or "").strip().lower()
    lowered = strip_trailing_punctuation(lowered)
    return _INNER_SPACES_PATTERN.sub(" ", lowered)


def normalize_triple_token(text: str) -> str:
    """Normalize a triple token for set comparisons."""
    token = (text or "").strip().lower()
    token = token.translate(str.maketrans("", "", string.punctuation))
    token = _INNER_SPACES_PATTERN.sub(" ", token).strip()
    return token


def safe_json_extract(raw: str) -> dict[str, Any] | list[Any] | None:
    """Best-effort JSON extraction from LLM output."""
    if not raw:
        return None

    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    fence_match = _JSON_FENCE_PATTERN.search(text)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

    start_candidates = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not start_candidates:
        return None
    start = min(start_candidates)
    for end in range(len(text), start, -1):
        candidate = text[start:end].strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            continue

    return None


def factual_judge_adapter(ground_truth: str, answer: str) -> dict[str, Any]:
    """
    Factual judge wrapper used as final fallback after equivalence checks.

    Returns:
        {
          "equivalent": bool,
          "score": float,
          "details": dict
        }
    """
    try:
        judge = call_openai_judge(
            question="Is ANSWER factually equivalent to GROUND_TRUTH?",
            rag_response=answer,
            prompt_only_response=ground_truth,
            reference_context=ground_truth,
            model=OPENAI_JUDGE_MODEL,
            temperature=TEMPERATURE,
            verbose=False,
        )
    except Exception as exc:
        return {
            "equivalent": False,
            "score": 0.0,
            "details": {
                "error": f"factual_judge_exception: {exc}",
            },
        }

    if judge.error:
        return {
            "equivalent": False,
            "score": 0.0,
            "details": {
                "error": judge.error,
                "confidence": judge.confidence,
                "winner": judge.winner,
            },
        }

    equivalent = not bool(judge.rag_has_factual_error)
    return {
        "equivalent": equivalent,
        "score": 1.0 if equivalent else 0.0,
        "details": {
            "rag_has_factual_error": judge.rag_has_factual_error,
            "rag_hallucination_details": judge.rag_hallucination_details,
            "confidence": judge.confidence,
            "winner": judge.winner,
            "reasoning": judge.reasoning,
        },
    }

