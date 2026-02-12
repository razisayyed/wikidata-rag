"""Tool package for the KB project (lazy exports)."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "fetch_entity_properties": (
        "kb_project.tools.fetch_entity_properties",
        "fetch_entity_properties",
    ),
    "wikidata_sparql": ("kb_project.tools.wikidata_sparql", "wikidata_sparql"),
    "search_entity_candidates": (
        "kb_project.tools.search_entity_candidates",
        "search_entity_candidates",
    ),
    "fetch_wikipedia_article_tool": (
        "kb_project.tools.fetch_wikipedia_article",
        "fetch_wikipedia_article_tool",
    ),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'kb_project.tools' has no attribute '{name}'")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS.keys())
