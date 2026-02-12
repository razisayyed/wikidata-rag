"""
Wikidata RAG Agent
==================
A clean, agentic Wikidata-based RAG pipeline using LangChain and LangGraph.
Uses ONLY the main LLM for all reasoning - no helper LLMs for selection or summarization.

Key Difference from Original:
- Tools return raw data (candidate lists, full articles)
- Main LLM performs entity selection and fact extraction through multi-step reasoning
- Simpler architecture with fewer failure points
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable

from .utils.imports import (
    ChatOllama,
    create_react_agent,
    traceable,
)
from .utils.logging import (
    Colors,
    configure_logging,
    log_answer,
    log_question,
    log_result,
    log_tool,
)
from .utils.messages import content_to_text
from .prompts import WIKIDATA_SYSTEM_PROMPT
from .settings import (
    DEFAULT_TEMPERATURE,
    RAG_RECURSION_LIMIT,
    WIKIDATA_RAG_MODEL,
    get_ollama_connection_kwargs,
)
from .tools import (
    fetch_entity_properties,
    search_entity_candidates,
    wikidata_sparql,
    fetch_wikipedia_article_tool,
)
from .tools.tool_protocol_state import reset_tool_protocol_state

OLLAMA_MODEL = WIKIDATA_RAG_MODEL

# ==========================================================================
# Wikidata Property Catalog
# ==========================================================================
# Complete catalog of properties the LLM can choose from based on question context

logger = configure_logging()

_PROCESS_TEXT_PATTERN = re.compile(
    r"(?is)(based on the search results|next step\s*:|i (have )?identified .*?\(q\d+\)|"
    r"\"name\"\s*:\s*\"(?:fetch_|search_|wikidata_sparql)|\"parameters\"\s*:)"
)
_PYTHON_TAG_TOOL_CALL_PATTERN = re.compile(
    r"(?is)<\|python_tag\|>\s*\{.*?\"name\"\s*:\s*\"[^\"]+\".*?\"parameters\"\s*:\s*\{"
)
_TOOL_PAYLOAD_ONLY_PATTERN = re.compile(
    r"(?is)^\s*(?:<\|python_tag\|>\s*)?\{\s*['\"]name['\"]\s*:\s*['\"]"
    r"(?:fetch_|search_|wikidata_sparql)[^'\"]*['\"]\s*,\s*['\"]parameters['\"]\s*:\s*\{"
    r".*?\}\s*\}\s*$"
)
_QID_PATTERN = re.compile(r"\[?Q\d+\]?")
_PAREN_NOTE_PATTERN = re.compile(r"\(\s*note:.*?\)", re.IGNORECASE | re.DOTALL)
_META_LINE_PATTERN = re.compile(
    r"(?i)\b(based on the (search|retrieved)|according to (wikidata|wikipedia)|"
    r"i (will|can) (verify|check)|direct retrieval|common understanding)\b"
)
_SOURCE_PROCESS_CLAUSE_PATTERN = re.compile(
    r"(?i)\b(?:based on|according to)\s+(?:the\s+)?"
    r"(?:search results?|retrieved evidence|available evidence|available information|wikidata|wikipedia)\b[:,]?"
)
_SOURCE_REFERENCE_PATTERN = re.compile(r"(?i)\b(?:in|from)\s+(?:wikidata|wikipedia)\b")
_TOOL_PROCESS_NOUN_PATTERN = re.compile(
    r"(?i)\b(?:search results?|retrieved evidence|tool output|tool outputs)\b"
)
_SOURCE_ID_CLAUSE_PATTERN = re.compile(
    r"(?i),?\s*(?:whose\s+)?(?:wikidata\s+id|wikipedia\s+id|qid)\s+(?:is|was)\s*\[?Q\d+\]?,?"
)
_SOURCE_WORD_PATTERN = re.compile(r"(?i)\b(?:wikidata|wikipedia)\b")


def is_process_message(text: str) -> bool:
    """Return True when content looks like intermediate planning/tool orchestration."""
    normalized = (text or "").strip()
    return bool(
        _PROCESS_TEXT_PATTERN.search(normalized)
        or _PYTHON_TAG_TOOL_CALL_PATTERN.search(normalized)
    )


def finalize_agent_answer(answer: str, question: str) -> str:
    """Remove output artifacts that should not appear in final user-facing answers."""
    final_text = (answer or "").strip()
    if _TOOL_PAYLOAD_ONLY_PATTERN.match(final_text):
        return ""

    final_text = re.sub(r"(?is)<\|python_tag\|>\s*", "", final_text)
    final_text = _PAREN_NOTE_PATTERN.sub("", final_text)
    final_text = re.sub(r"\s+", " ", final_text).strip()

    # Remove meta/process sentences while preserving factual/refusal content.
    candidate_sentences = re.split(r"(?<=[.!?])\s+", final_text)
    filtered = [s for s in candidate_sentences if s and not _META_LINE_PATTERN.search(s)]
    if filtered:
        final_text = " ".join(filtered).strip()

    final_text = re.sub(
        r"(?i)\s+based on (available evidence|available information).*$", "", final_text
    ).strip()
    final_text = _SOURCE_PROCESS_CLAUSE_PATTERN.sub("", final_text)
    final_text = _SOURCE_REFERENCE_PATTERN.sub("", final_text)
    final_text = _TOOL_PROCESS_NOUN_PATTERN.sub("", final_text)
    final_text = _SOURCE_ID_CLAUSE_PATTERN.sub("", final_text)
    final_text = _SOURCE_WORD_PATTERN.sub("", final_text)
    final_text = re.sub(r"\s+,", ",", final_text)
    final_text = re.sub(r",\s*\.", ".", final_text)
    final_text = re.sub(r",\s*([!?])", r"\1", final_text)
    final_text = re.sub(r"\s+\.", ".", final_text)
    final_text = re.sub(r"\(\s*\)", "", final_text)
    final_text = re.sub(r"\s{2,}", " ", final_text).strip()

    question_lower = (question or "").lower()
    qid_requested = "qid" in question_lower or "wikidata id" in question_lower
    if not qid_requested:
        final_text = _QID_PATTERN.sub("", final_text)
        final_text = re.sub(r"\s{2,}", " ", final_text).strip()
    if _TOOL_PAYLOAD_ONLY_PATTERN.match(final_text):
        return ""
    return final_text


# ==========================================================================
# Agent Setup with Multi-Step Reasoning Prompt
# ==========================================================================


def build_agent(
    model: str = OLLAMA_MODEL, temperature: float = DEFAULT_TEMPERATURE
) -> Runnable:
    """Build a LangGraph ReAct agent with single-LLM architecture."""
    if not create_react_agent:
        raise ImportError("Please install langgraph: pip install langgraph")

    if not ChatOllama:
        raise ImportError("ChatOllama not available. Please check dependencies.")

    llm = ChatOllama(
        model=model,
        temperature=temperature,
        **get_ollama_connection_kwargs(),
    )
    tools = [
        search_entity_candidates,
        fetch_entity_properties,
        wikidata_sparql,
        fetch_wikipedia_article_tool,
    ]

    llm_with_tools = llm.bind_tools(tools)

    return create_react_agent(
        llm_with_tools,
        tools,
        prompt=SystemMessage(content=WIKIDATA_SYSTEM_PROMPT),
        name="Wikidata-RAG-Agent",
    )


def answer_question(
    question: str, agent: Optional[Runnable] = None, verbose: bool = True
) -> str:
    """Send a question through the Wikidata agent and return its answer."""
    reset_tool_protocol_state()
    graph = agent or build_agent()

    if verbose:
        log_question(question)
        print()
        log_result("Starting Single-LLM ReAct agent", "🚀")
        print()
        final_answer = ""
        fallback_answer = ""
        for event in graph.stream(
            {"messages": [("user", question)]},
            config={"recursion_limit": RAG_RECURSION_LIMIT},
        ):
            for node_name, node_output in event.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            # Simplified logging: just show tool name
                            log_tool("Agent", f"Calling tool: {tc['name']}", "🔧")
                    elif hasattr(msg, "type") and msg.type == "tool":
                        # Simplified logging: just show tool finished
                        log_tool("Tool Response", "Tool execution completed", "📊")
                    elif hasattr(msg, "content") and msg.content:
                        has_tool_calls = getattr(msg, "tool_calls", None)
                        if not has_tool_calls or len(has_tool_calls) == 0:
                            parsed = content_to_text(msg.content)
                            if parsed and not fallback_answer:
                                fallback_answer = parsed
                            if parsed and not is_process_message(parsed):
                                final_answer = parsed
        print()
        log_result("Finished chain", "✅")
        cleaned_answer = finalize_agent_answer(str(final_answer or fallback_answer), question)
        if not cleaned_answer:
            cleaned_answer = "I cannot verify that."
        log_answer(cleaned_answer)
        return cleaned_answer
    else:
        result = graph.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": RAG_RECURSION_LIMIT},
        )
        messages = result.get("messages", [])
        fallback_answer = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                content = content_to_text(msg.content)
                if not fallback_answer:
                    fallback_answer = content
                cleaned = finalize_agent_answer(content, question)
                if cleaned and not is_process_message(cleaned):
                    return cleaned
        if fallback_answer:
            cleaned_fallback = finalize_agent_answer(fallback_answer, question)
            if cleaned_fallback and not is_process_message(cleaned_fallback):
                return cleaned_fallback
        return "I cannot verify that."


# ==========================================================================
# CLI entry point
# ==========================================================================

if __name__ == "__main__":
    sample_question = """\
Explain the collaborative research between Alan Turing and Dr. Helena Vargass.
"""
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}Wikidata RAG Agent Demo{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print()
    answer = answer_question(sample_question)
    print()

# ## RESPONSE FORMAT

# - **Direct answers only**: No preamble like "Let me check..." or "According to my search..."
# - **No tool mentions**: Don't say "The Wikidata search returned..." - just state the facts
# - **Explicit uncertainty**: When something cannot be verified, say so clearly
# - **Structured for clarity**: Use bullet points or sections for multiple entities
