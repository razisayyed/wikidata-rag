"""
KB Project - Wikidata RAG Hallucination Reduction
========================================
A RAG-based approach to reduce LLM hallucinations using Wikidata as a knowledge source.
"""

from .wikidata_rag_agent import build_agent
from .prompt_only_llm import build_prompt_only_agent, answer_question_prompt_only
from .settings import LLM_MODEL, DEFAULT_TEMPERATURE

__all__ = [
    "build_agent",
    "build_prompt_only_agent",
    "answer_question_prompt_only",
    "LLM_MODEL",
    "DEFAULT_TEMPERATURE",
]
