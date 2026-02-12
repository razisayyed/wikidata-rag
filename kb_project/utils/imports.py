"""
Centralized imports for the KB Project to handle conditional dependencies and backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

# ==========================================================================
# LangSmith Tracing
# ==========================================================================

F = TypeVar("F", bound=Callable[..., Any])

try:
    from langsmith import traceable as _traceable

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

    def _traceable(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        """No-op decorator when langsmith is not installed."""

        def decorator(func: F) -> F:
            return func

        return decorator


traceable = _traceable

# ==========================================================================
# LangChain Models
# ==========================================================================

try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore
    except ImportError:
        # Fallback or raise error if neither is available,
        # but for now we assume one is present as per original code.
        logging.getLogger(__name__).warning(
            "Could not import ChatOllama from langchain_ollama or langchain_community."
        )
        ChatOllama = None  # type: ignore

# ==========================================================================
# LangGraph
# ==========================================================================

try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    create_react_agent = None  # type: ignore
