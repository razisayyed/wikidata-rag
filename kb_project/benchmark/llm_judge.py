"""
LLM-as-a-Judge: OpenAI-based Hallucination Evaluation
======================================================
Uses an OpenAI judge model to evaluate and compare responses for hallucinations.

This provides a complementary evaluation signal to the Vectara model:
- Vectara: Embedding-based similarity to context
- LLM Judge: Semantic understanding of factual accuracy

The judge evaluates both responses and determines:
1. Which response has fewer hallucinations
2. Specific hallucination issues in each response
3. Whether appropriate refusals were made for unknown entities
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Any

from ..settings import OPENAI_JUDGE_MODEL

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
    from langsmith import traceable
except ImportError:
    ChatOpenAI = None  # type: ignore

    class SystemMessage:  # type: ignore
        def __init__(self, content):
            pass

    class HumanMessage:  # type: ignore
        def __init__(self, content):
            pass

    def traceable(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


# ==========================================================================
# Data Structures
# ==========================================================================


@dataclass
class JudgeResult:
    """Result from the LLM judge evaluation."""

    # Overall verdict
    winner: str  # "RAG", "Prompt-Only", "Tie", "Both-Bad", "Both-Good"
    confidence: str  # "High", "Medium", "Low"

    # Backward-compatible individual evaluations
    rag_has_hallucination: bool
    rag_hallucination_details: str
    rag_strengths: str

    prompt_has_hallucination: bool
    prompt_hallucination_details: str
    prompt_strengths: str

    # Dual-track individual evaluations
    rag_has_factual_error: bool = False
    rag_completeness: str = "insufficient"  # complete|partial|insufficient
    rag_missing_required_info: str = ""

    prompt_has_factual_error: bool = False
    prompt_completeness: str = "insufficient"  # complete|partial|insufficient
    prompt_missing_required_info: str = ""

    # Reasoning
    reasoning: str = ""

    # Raw response for debugging
    raw_response: Optional[str] = None

    # Error handling
    error: Optional[str] = None


# ==========================================================================
# Prompt Design
# ==========================================================================

# The prompt is carefully designed to:
# 1. Define hallucination clearly and precisely
# 2. Handle mixed real/fictional entity cases
# 3. Reward appropriate refusals ("I don't know")
# 4. Detect fabricated facts, dates, relationships
# 5. Return structured JSON for easy parsing

JUDGE_SYSTEM_PROMPT = """You are an expert fact-checker evaluating two AI responses.

You must output TWO independent judgments per response:
1) FACTUAL ERROR: whether the response states any false/unsupported claim.
2) COMPLETENESS: whether the response fully answers what was asked.

IMPORTANT POLICY:
- Omitted required information is NOT automatically a factual error.
- Omission should lower completeness instead.
- Hallucination/factual-error means invented, contradictory, or unsupported claims.
- A truthful refusal ("I cannot verify") for unknown entities is factual.
- Scope discipline: extra details beyond the asked question are NOT a positive signal.
- Risk awareness: unsupported/scope-bloating extra claims should count as negative behavior and potential factual error.

Use provided reference context first; use world knowledge as fallback only when needed."""


def build_judge_prompt(
    question: str,
    rag_response: str,
    prompt_only_response: str,
    reference_context: str = "",
) -> str:
    """
    Build the evaluation prompt for the LLM judge.

    The prompt structure is carefully designed:
    1. Clear section headers for each piece of information
    2. Reference context provided for verification
    3. Both responses presented neutrally
    4. Specific JSON output format requested
    """
    context_section = ""
    if reference_context.strip():
        context_section = f"## REFERENCE CONTEXT (Ground Truth / Retrieved Facts)\n{reference_context}\n\n"

    return f"""{context_section}## QUESTION
{question}

## RAG RESPONSE (Uses Wikidata Knowledge Retrieval)
{rag_response}

## PROMPT-ONLY RESPONSE (No External Knowledge)
{prompt_only_response}

## YOUR TASK

Analyze both responses for:
1. Cross-referencing: Do they accurately reflect the provided REFERENCE CONTEXT?
2. Fictional entity detection: (good responses state "cannot verify", not invent details)
3. Fact verification: dates, places, accomplishments
4. Completeness: did the response answer all required parts of the question?
5. Scope discipline: extra information must not improve the score.
6. Penalize unsupported or unnecessary extra claims as risky behavior (potential hallucination/factual error).

Respond with JSON in this format:
```json
{{
    "winner": "<RAG|Prompt-Only|Tie|Both-Bad|Both-Good>",
    "confidence": "<High|Medium|Low>",
    "rag_evaluation": {{
        "has_factual_error": <true|false>,
        "has_hallucination": <true|false>,
        "completeness": "<complete|partial|insufficient>",
        "missing_required_info": "<short missing info summary or 'None'>",
        "hallucination_details": "<specific issues or 'None detected'>",
        "strengths": "<what this response did well>"
    }},
    "prompt_evaluation": {{
        "has_factual_error": <true|false>,
        "has_hallucination": <true|false>,
        "completeness": "<complete|partial|insufficient>",
        "missing_required_info": "<short missing info summary or 'None'>",
        "hallucination_details": "<specific issues or 'None detected'>",
        "strengths": "<what this response did well>"
    }},
    "reasoning": "<2-3 sentence explanation>"
}}
```
Note: Missing details by itself should lower completeness, not trigger factual error."""


# ==========================================================================
# OpenAI API Functions
# ==========================================================================


def get_llm_judge(
    model: str = OPENAI_JUDGE_MODEL, temperature: float = 0.1
) -> Optional[Any]:
    """
    Initialize ChatOpenAI client using API key from environment.
    """
    if ChatOpenAI is None:
        print(
            "[LLM Judge] langchain-openai package not installed. Run: pip install langchain-openai"
        )
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[LLM Judge] OPENAI_API_KEY environment variable not set")
        return None

    return ChatOpenAI(model=model, temperature=temperature)


@traceable(name="LLM Judge Evaluation", run_type="llm")
def call_openai_judge(
    question: str,
    rag_response: str,
    prompt_only_response: str,
    reference_context: str = "",
    model: str = OPENAI_JUDGE_MODEL,
    temperature: float = 0.1,
    verbose: bool = False,
) -> JudgeResult:
    """
    Call OpenAI to evaluate both responses for hallucinations.

    The judge uses provided context AND its own knowledge to verify facts.

    Args:
        question: The original question asked
        rag_response: Response from the RAG model
        prompt_only_response: Response from the prompt-only model
        reference_context: Facts/Ground truth to use for verification
        model: OpenAI model to use (configurable via OPENAI_JUDGE_MODEL)
        temperature: Low temperature for consistent evaluation
        verbose: Print debug information

    Returns:
        JudgeResult with evaluation details
    """
    llm = get_llm_judge(model=model, temperature=temperature)
    if llm is None:
        return JudgeResult(
            winner="Error",
            confidence="N/A",
            rag_has_hallucination=False,
            rag_hallucination_details="",
            rag_strengths="",
            prompt_has_hallucination=False,
            prompt_hallucination_details="",
            prompt_strengths="",
            reasoning="",
            error="OpenAI client not available. Check API key.",
        )

    # Build the prompt
    user_prompt = build_judge_prompt(
        question=question,
        rag_response=rag_response,
        prompt_only_response=prompt_only_response,
        reference_context=reference_context,
    )

    if verbose:
        print(f"[LLM Judge] Sending evaluation request to OpenAI {model}...")

    try:
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        raw_content = response.content

        if verbose:
            print(f"[LLM Judge] Raw response:\n{raw_content}")

        # Parse JSON from response
        result = parse_judge_response(raw_content)
        result.raw_response = raw_content

        return result

    except Exception as e:
        return JudgeResult(
            winner="Error",
            confidence="N/A",
            rag_has_hallucination=False,
            rag_hallucination_details="",
            rag_strengths="",
            prompt_has_hallucination=False,
            prompt_hallucination_details="",
            prompt_strengths="",
            reasoning="",
            error=f"OpenAI API error: {str(e)}",
        )


def parse_judge_response(raw_response: str) -> JudgeResult:
    """
    Parse the JSON response from the LLM judge.

    Handles various edge cases and malformed responses.
    """
    try:
        # Extract JSON from markdown code block if present
        if "```json" in raw_response:
            json_start = raw_response.find("```json") + 7
            json_end = raw_response.find("```", json_start)
            json_str = raw_response[json_start:json_end].strip()
        elif "```" in raw_response:
            json_start = raw_response.find("```") + 3
            json_end = raw_response.find("```", json_start)
            json_str = raw_response[json_start:json_end].strip()
        else:
            # Try to find JSON object directly
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            json_str = raw_response[json_start:json_end]

        data = json.loads(json_str)

        rag_eval = data.get("rag_evaluation", {}) or {}
        prompt_eval = data.get("prompt_evaluation", {}) or {}

        def _as_bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0"}:
                    return False
            return default

        def _normalize_completeness(value: Any) -> str:
            candidate = str(value or "").strip().lower()
            if candidate not in {"complete", "partial", "insufficient"}:
                return "insufficient"
            return candidate

        rag_has_factual_error = _as_bool(
            rag_eval.get("has_factual_error"),
            default=_as_bool(rag_eval.get("has_hallucination"), False),
        )
        prompt_has_factual_error = _as_bool(
            prompt_eval.get("has_factual_error"),
            default=_as_bool(prompt_eval.get("has_hallucination"), False),
        )

        return JudgeResult(
            winner=data.get("winner", "Error"),
            confidence=data.get("confidence", "Unknown"),
            rag_has_hallucination=_as_bool(
                rag_eval.get("has_hallucination"),
                default=rag_has_factual_error,
            ),
            rag_hallucination_details=rag_eval.get("hallucination_details", ""),
            rag_strengths=rag_eval.get("strengths", ""),
            prompt_has_hallucination=_as_bool(
                prompt_eval.get("has_hallucination"),
                default=prompt_has_factual_error,
            ),
            prompt_hallucination_details=prompt_eval.get("hallucination_details", ""),
            prompt_strengths=prompt_eval.get("strengths", ""),
            rag_has_factual_error=rag_has_factual_error,
            rag_completeness=_normalize_completeness(
                rag_eval.get(
                    "completeness",
                    "insufficient" if rag_has_factual_error else "complete",
                )
            ),
            rag_missing_required_info=str(
                rag_eval.get("missing_required_info", "")
            ).strip(),
            prompt_has_factual_error=prompt_has_factual_error,
            prompt_completeness=_normalize_completeness(
                prompt_eval.get(
                    "completeness",
                    "insufficient" if prompt_has_factual_error else "complete",
                )
            ),
            prompt_missing_required_info=str(
                prompt_eval.get("missing_required_info", "")
            ).strip(),
            reasoning=data.get("reasoning", ""),
        )

    except json.JSONDecodeError as e:
        return JudgeResult(
            winner="Error",
            confidence="N/A",
            rag_has_hallucination=False,
            rag_hallucination_details="",
            rag_strengths="",
            prompt_has_hallucination=False,
            prompt_hallucination_details="",
            prompt_strengths="",
            reasoning="",
            raw_response=raw_response,
            error=f"Failed to parse JSON response: {str(e)}",
        )


# ==========================================================================
# Convenience Functions
# ==========================================================================


def judge_responses(
    question: str,
    rag_response: str,
    prompt_only_response: str,
    reference_context: str = "",
    model: str = OPENAI_JUDGE_MODEL,
    verbose: bool = False,
) -> JudgeResult:
    """
    Main entry point for LLM-as-a-judge evaluation.

    The judge uses the configured OpenAI model to verify facts against context.

    Example usage:
        result = judge_responses(
            question="Who is Niels Bohr?",
            rag_response="Niels Bohr was a Danish physicist born in 1885...",
            prompt_only_response="Niels Bohr was a physicist who worked on atomic theory...",
            reference_context="Niels Henrik David Bohr was a Danish physicist born Oct 7 1885..."
        )
    """
    return call_openai_judge(
        question=question,
        rag_response=rag_response,
        prompt_only_response=prompt_only_response,
        reference_context=reference_context,
        model=model,
        verbose=verbose,
    )


def format_judge_result_short(result: JudgeResult) -> str:
    """Format judge result for table display."""
    if result.error:
        return f"Error: {result.error[:30]}..."
    return f"{result.winner} ({result.confidence})"


def format_judge_result_detailed(result: JudgeResult) -> str:
    """Format judge result for detailed report."""
    if result.error:
        return f"**Error:** {result.error}"

    return f"""**LLM Judge Verdict:** {result.winner} (Confidence: {result.confidence})

**Reasoning:** {result.reasoning}

**RAG Evaluation:**
- Factual Error: {"Yes" if result.rag_has_factual_error else "No"}
- Completeness: {result.rag_completeness}
- Missing Required Info: {result.rag_missing_required_info or "None"}
- Hallucination (legacy field): {"Yes" if result.rag_has_hallucination else "No"}
- Details: {result.rag_hallucination_details}
- Strengths: {result.rag_strengths}

**Prompt-Only Evaluation:**
- Factual Error: {"Yes" if result.prompt_has_factual_error else "No"}
- Completeness: {result.prompt_completeness}
- Missing Required Info: {result.prompt_missing_required_info or "None"}
- Hallucination (legacy field): {"Yes" if result.prompt_has_hallucination else "No"}
- Details: {result.prompt_hallucination_details}
- Strengths: {result.prompt_strengths}"""


# ==========================================================================
# Testing
# ==========================================================================


if __name__ == "__main__":
    # Quick test
    print("Testing LLM Judge (using GPT-5.1's own knowledge to verify facts)...")
    print("No ground truth provided - the judge will fact-check using its knowledge.\n")

    test_result = judge_responses(
        question="Describe the joint research between Einstein, Bohr, and Dr. Selwyn Hartmere on quantum mechanics.",
        rag_response="""Marie Curie was born on November 7, 1867. Her major achievements include winning the Nobel Prize in Physics (1903) and the Nobel Prize in Chemistry (1911), becoming the first person to hold two Nobel Prizes. She also authored the seminal work "Treatise on Radioactivity" and pioneered research on radioactivity, discovering polonium and radium.""",
        prompt_only_response="""Marie Curie was born on November 7, 1867. Her major achievements include:  
- Winning the **Nobel Prize in Physics (1903)** for her research on radioactivity, shared with her husband Pierre Curie and Henri Becquerel.  
- Winning the **Nobel Prize in Chemistry (1911)** for isolating polonium and radium and studying their radioactive properties.  
- Being the first person to win **two Nobel Prizes** in different scientific fields.  
- Pioneering research on radioactivity, which laid the foundation for nuclear physics and cancer treatment.  
- Developing **mobile X-ray units** during World War I to aid in medical care for wounded soldiers.  

I do not have verified information about her personal life beyond these facts.""",
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(format_judge_result_detailed(test_result))
