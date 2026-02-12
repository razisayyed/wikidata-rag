"""Helpers for extracting plain text from structured chat message content."""

from __future__ import annotations

from typing import Any, Iterable, List


def _flatten_parts(parts: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in parts:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if text:
                out.append(str(text).strip())
                continue
            # Some providers emit {"type": "output_text", "text": "..."} blocks.
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                out.append(content.strip())
                continue
        # Fallback for unknown block types
        rendered = str(item).strip()
        if rendered:
            out.append(rendered)
    return out


def _looks_incremental_prefix_sequence(parts: List[str]) -> bool:
    """
    Detect list outputs where each item is an incremental prefix of the final one.

    Example:
    ["A", "A B", "A B C"] -> True
    """
    if len(parts) < 2:
        return False

    prev = parts[0]
    for current in parts[1:]:
        if not current.startswith(prev):
            return False
        prev = current
    return True


def content_to_text(content: Any) -> str:
    """Convert provider-specific message content into clean plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text")
        if text:
            return str(text).strip()
        return str(content).strip()
    if isinstance(content, list):
        parts = _flatten_parts(content)
        if not parts:
            return ""
        if _looks_incremental_prefix_sequence(parts):
            return parts[-1].strip()
        return " ".join(p for p in parts if p).strip()
    return str(content).strip()
