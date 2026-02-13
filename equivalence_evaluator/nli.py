"""Bidirectional NLI entailment with local-first fallback strategy."""

from __future__ import annotations

import os
from typing import Any

from .config import NLI_MODEL, TEMPERATURE
from .utils import safe_json_extract

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore[assignment]

from kb_project.settings import OPENAI_JUDGE_MODEL

_TOKENIZER: Any = None
_MODEL: Any = None


def _load_local_nli() -> tuple[Any, Any]:
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        raise RuntimeError("transformers backend unavailable")
    _TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    _MODEL.eval()
    return _TOKENIZER, _MODEL


def _label_lookup(model: Any) -> dict[int, str]:
    id2label = getattr(model.config, "id2label", {}) or {}
    result: dict[int, str] = {}
    for idx, label in id2label.items():
        result[int(idx)] = str(label).upper()
    return result


def _run_local_pair(premise: str, hypothesis: str) -> dict[str, Any]:
    tokenizer, model = _load_local_nli()
    encoded = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]
    best_idx = int(torch.argmax(probs).item())
    labels = _label_lookup(model)
    label = labels.get(best_idx, "UNKNOWN")
    confidence = float(probs[best_idx].item())
    return {
        "label": label,
        "confidence": confidence,
    }


def _is_entailment(label: str) -> bool:
    normalized = (label or "").upper()
    return "ENTAIL" in normalized


def _run_openai_nli(ground_truth: str, rag_output: str) -> dict[str, Any]:
    if ChatOpenAI is None:
        raise RuntimeError("langchain_openai unavailable")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    llm = ChatOpenAI(model=OPENAI_JUDGE_MODEL, temperature=TEMPERATURE)
    prompt = f"""
Evaluate textual entailment bidirectionally.

A = ground truth
B = candidate answer

A: {ground_truth}
B: {rag_output}

Return strict JSON with keys:
{{
  "forward_label": "entailment|neutral|contradiction",
  "backward_label": "entailment|neutral|contradiction",
  "forward_confidence": 0.0,
  "backward_confidence": 0.0
}}
No extra text.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content="You are a strict NLI evaluator."),
            HumanMessage(content=prompt),
        ]
    )
    parsed = safe_json_extract(str(response.content))
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI NLI returned non-JSON response")

    forward_label = str(parsed.get("forward_label", "neutral")).upper()
    backward_label = str(parsed.get("backward_label", "neutral")).upper()
    forward_conf = float(parsed.get("forward_confidence", 0.0) or 0.0)
    backward_conf = float(parsed.get("backward_confidence", 0.0) or 0.0)
    return {
        "entailment_forward": _is_entailment(forward_label),
        "entailment_backward": _is_entailment(backward_label),
        "confidence": max(0.0, min(1.0, (forward_conf + backward_conf) / 2.0)),
        "backend": "openai",
        "error": "",
    }


def evaluate_bidirectional_entailment(ground_truth: str, rag_output: str) -> dict[str, Any]:
    """Evaluate entailment in both directions, local model first."""
    try:
        forward = _run_local_pair(ground_truth, rag_output)
        backward = _run_local_pair(rag_output, ground_truth)
        forward_entails = _is_entailment(forward["label"])
        backward_entails = _is_entailment(backward["label"])
        return {
            "entailment_forward": forward_entails,
            "entailment_backward": backward_entails,
            "confidence": max(0.0, min(1.0, (forward["confidence"] + backward["confidence"]) / 2.0)),
            "backend": "transformers",
            "error": "",
            "labels": {
                "forward": forward["label"],
                "backward": backward["label"],
            },
        }
    except Exception as local_exc:
        local_error = str(local_exc)

    try:
        result = _run_openai_nli(ground_truth, rag_output)
        result["error"] = ""
        return result
    except Exception as openai_exc:
        openai_error = str(openai_exc)

    return {
        "entailment_forward": False,
        "entailment_backward": False,
        "confidence": 0.0,
        "backend": "none",
        "error": f"local_nli_error={local_error}; openai_nli_error={openai_error}",
    }

