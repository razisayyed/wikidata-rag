"""
Prompt-Only Agent (No RAG)
==========================
A simple LLM agent that uses only a system prompt to discourage hallucination,
without any external knowledge retrieval (no Wikidata, no tools).

This serves as a baseline comparison against the RAG-based Wikidata agent.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from .utils.imports import (
    ChatOllama,
    create_react_agent,
    traceable,
)
from .utils.logging import (
    log_answer,
    log_question,
    log_result,
)
from .utils.messages import content_to_text
from .prompts import PROMPT_ONLY_SYSTEM_PROMPT
from .settings import (
    DEFAULT_TEMPERATURE,
    PROMPT_ONLY_MODEL,
    get_ollama_connection_kwargs,
)


def build_prompt_only_agent(
    model: str = PROMPT_ONLY_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Runnable:
    """
    Build a simple LangGraph "agent" with the anti-hallucination system prompt.

    This mirrors `wikidata_rag_agent.build_agent` but without any tools attached.

    Args:
        model: The Ollama model to use.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        A LangGraph agent-like object exposing `.stream()` and `.invoke()`.
    """
    if not ChatOllama:
        raise ImportError("ChatOllama not available. Please check dependencies.")

    llm = ChatOllama(
        model=model,
        temperature=temperature,
        **get_ollama_connection_kwargs(),
    )

    if create_react_agent:
        return create_react_agent(
            llm,
            [],
            prompt=SystemMessage(content=PROMPT_ONLY_SYSTEM_PROMPT),
            name="Prompt-Only-LLM-Agent",
        )
    else:
        # Fall back to returning a raw ChatOllama if langgraph is not available
        return llm


def answer_question_prompt_only(
    question: str,
    llm: Optional[Runnable] = None,
    verbose: bool = True,
) -> str:
    """
    Answer a question using only the system prompt (no RAG, no tools).

    Args:
        question: The user's question.
        llm: Optional pre-built ChatOllama instance.
        verbose: If True, print the response.

    Returns:
        The model's response as a string.
    """
    # llm here can be either a ChatOllama instance or a LangGraph agent
    agent = llm or build_prompt_only_agent()

    # Use stream API when available to mirror wikidata_rag_agent behaviour
    try:
        if verbose:
            log_question(question)
            log_result("Prompt-Only Agent: Answering...", "🤖")

        final_answer = ""
        # If agent supports stream (LangGraph agent), use it
        if hasattr(agent, "stream"):
            for event in agent.stream({"messages": [("user", question)]}):
                for node_name, node_output in event.items():
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if hasattr(msg, "content") and msg.content:
                            parsed = content_to_text(msg.content)
                            if parsed:
                                final_answer = parsed
        else:
            # Fallback: invoke LL model directly with system+human messages
            messages = [
                SystemMessage(content=PROMPT_ONLY_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
            resp = agent.invoke(messages)
            final_answer = (
                content_to_text(resp.content) if hasattr(resp, "content") else str(resp)
            )

        final_answer = content_to_text(final_answer)

        if verbose:
            log_result("Response generated.", "✅")
            log_answer(final_answer)

        return final_answer
    except Exception as e:
        # Safe fallback: try raw ChatOllama call
        if ChatOllama:
            llm_raw = ChatOllama(
                model=PROMPT_ONLY_MODEL,
                temperature=DEFAULT_TEMPERATURE,
                **get_ollama_connection_kwargs(),
            )
            messages = [
                SystemMessage(content=PROMPT_ONLY_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
            resp = llm_raw.invoke(messages)
            answer = content_to_text(resp.content) if hasattr(resp, "content") else str(resp)
            answer = content_to_text(answer)
            if verbose:
                log_result("Fallback Response:", "⚠️")
                log_answer(answer)
            return answer
        else:
            return f"Error: {e}"


if __name__ == "__main__":
    # Test questions
    TEST_QUESTIONS = [
        "Who is Albert Einstein?",
        "What is the relationship between Alan Turing and Dr. Helena Vargass?",
        "Tell me about the collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
        "What is the capital of France?",
    ]

    log_result("Prompt-Only Agent Demo (No RAG)", "🧪")

    llm = build_prompt_only_agent()

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n--- Test {i}/{len(TEST_QUESTIONS)} ---")
        answer_question_prompt_only(question, llm=llm, verbose=True)
