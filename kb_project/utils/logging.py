"""Logging utilities for the KB Project."""

from __future__ import annotations

import json
import logging as std_logging
from datetime import datetime
from typing import Any, Dict, List

from ..settings import LOG_FILE


_tool_usage_log: List[Dict[str, Any]] = []


def configure_logging() -> std_logging.Logger:
    """Configure project-wide logging."""
    if not std_logging.getLogger().handlers:
        std_logging.basicConfig(
            level=std_logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[std_logging.FileHandler(LOG_FILE)],
            # handlers=[std_logging.FileHandler(LOG_FILE), std_logging.StreamHandler()],
        )
    return std_logging.getLogger(__name__)


def log_tool_usage(tool_name: str, input_data: Any, output_data: Any) -> None:
    """Log tool usage for tracking."""
    _tool_usage_log.append(
        {
            "tool": tool_name,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.now().isoformat(),
        }
    )
    logger = std_logging.getLogger(__name__)
    logger.info(f"Tool Used: {tool_name}")
    logger.info(f"Tool Input: {json.dumps(input_data, indent=2)}")
    logger.info(f"Tool Output: {str(output_data)[:500]}...")


def get_tool_usage_log() -> List[Dict[str, Any]]:
    """Get the tool usage log."""
    return _tool_usage_log.copy()


def clear_tool_usage_log() -> None:
    """Clear the tool usage log."""
    global _tool_usage_log
    _tool_usage_log = []


class Colors:
    """ANSI color codes for terminal output."""

    GRAY = "\033[90m"
    BLACK = "\033[30m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"


def log_tool(tool_name: str, message: str, emoji: str = "🔧") -> None:
    """Log tool execution with gray color."""

    print(f"{Colors.GRAY}{emoji} [{tool_name}] {message}{Colors.RESET}")


def log_llm(llm_name: str, message: str, emoji: str = "🤖") -> None:
    """Log LLM execution with cyan color."""

    print(f"{Colors.CYAN}{emoji} [{llm_name}] {message}{Colors.RESET}")


def log_result(message: str, emoji: str = "✨") -> None:
    """Log results with bold text."""

    print(f"{Colors.BOLD}{emoji} {message}{Colors.RESET}")


def log_question(question: str) -> None:
    """Log user question with gray color."""

    print(f"{Colors.GRAY}❓ QUESTION: {question}{Colors.RESET}")


def log_answer(answer: str) -> None:
    """Log agent answer with bold black text."""

    print(f"{Colors.BOLD}💬 ANSWER: {answer}{Colors.RESET}")
