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
from .prompts import WIKIDATA_SYSTEM_PROMPT
from .settings import (
    DEFAULT_TEMPERATURE,
    LLM_MODEL,
)
from .tools import (
    fetch_entity_properties,
    fetch_wikipedia_article_tool,
    search_entity_candidates,
    wikidata_sparql,
)

OLLAMA_MODEL = LLM_MODEL

# ==========================================================================
# Wikidata Property Catalog
# ==========================================================================
# Complete catalog of properties the LLM can choose from based on question context

logger = configure_logging()


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

    llm = ChatOllama(model=model, temperature=temperature)
    tools = [
        search_entity_candidates,
        fetch_entity_properties,
        fetch_wikipedia_article_tool,
        wikidata_sparql,
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
    graph = agent or build_agent()

    if verbose:
        log_question(question)
        print()
        log_result("Starting Single-LLM ReAct agent", "🚀")
        print()
        final_answer = ""
        for event in graph.stream({"messages": [("user", question)]}):
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
                            final_answer = msg.content
        print()
        log_result("Finished chain", "✅")
        log_answer(final_answer)
        return final_answer
    else:
        result = graph.invoke({"messages": [("user", question)]})
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                return msg.content
        return str(result)


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
