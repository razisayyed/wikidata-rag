"""
KB Project - Wikidata RAG Hallucination Reduction
========================================
A RAG-based approach to reduce LLM hallucinations using Wikidata as a knowledge source.
"""

from .settings import LLM_MODEL, DEFAULT_TEMPERATURE


def build_agent(*args, **kwargs):
    """Lazily import and build the Wikidata RAG agent."""
    from .wikidata_rag_agent import build_agent as _build_agent

    return _build_agent(*args, **kwargs)


def build_prompt_only_agent(*args, **kwargs):
    """Lazily import and build the prompt-only baseline agent."""
    from .prompt_only_llm import build_prompt_only_agent as _build_prompt_only_agent

    return _build_prompt_only_agent(*args, **kwargs)


def answer_question_prompt_only(*args, **kwargs):
    """Lazily import and execute prompt-only answering."""
    from .prompt_only_llm import answer_question_prompt_only as _answer

    return _answer(*args, **kwargs)

__all__ = [
    "build_agent",
    "build_prompt_only_agent",
    "answer_question_prompt_only",
    "LLM_MODEL",
    "DEFAULT_TEMPERATURE",
]
