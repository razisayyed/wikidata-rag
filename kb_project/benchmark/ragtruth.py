"""
RAGTruth Hallucination Evaluator
================================

Implements hallucination detection based on the RAGTruth methodology.
RAGTruth is a corpus for hallucination detection in RAG systems with
~18,000 annotated responses at span-level.

This module uses a prompt-based approach to detect hallucinated spans
in model responses given the source context.

References:
- RAGTruth Dataset: https://github.com/CodingLL/RAGTruth
- RAGTruth Evaluation: https://github.com/CodingLL/RAGTruth_Eval
- RAGTruth Model: https://huggingface.co/HanningZhang/RAG-Truth-Model
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..settings import RAGTRUTH_MODEL, get_ollama_connection_kwargs
from ..utils.imports import ChatOllama
from .evaluation import build_primary_context

# ==========================================================================
# Data Structures
# ==========================================================================


@dataclass
class HallucinatedSpan:
    """Represents a detected hallucinated text span."""

    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    reason: str = ""


@dataclass
class RAGTruthResult:
    """Result from RAGTruth hallucination evaluation."""

    # Core results
    has_hallucination: bool
    hallucination_score: float  # 0.0 = no hallucination, 1.0 = fully hallucinated
    hallucinated_spans: List[HallucinatedSpan] = field(default_factory=list)

    # Additional info
    span_count: int = 0
    analysis: str = ""
    raw_output: str = ""
    error: Optional[str] = None

    @property
    def case_label(self) -> str:
        """Binary label: hallucinated or not."""
        return "hallucinated" if self.has_hallucination else "factual"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_hallucination": self.has_hallucination,
            "hallucination_score": self.hallucination_score,
            "span_count": self.span_count,
            "hallucinated_spans": [
                {"text": s.text, "reason": s.reason} for s in self.hallucinated_spans
            ],
            "analysis": self.analysis,
            "case_label": self.case_label,
            "error": self.error,
        }


# ==========================================================================
# RAGTruth Prompt Templates
# ==========================================================================

# Based on RAGTruth_Eval/hall_prompt.py
RAGTRUTH_QA_PROMPT = """You are an expert fact-checker evaluating a question-answering response for hallucinations.

A hallucination is any claim in the response that:
1. Cannot be verified from the provided source context
2. Contradicts the source context
3. Invents entities, relationships, dates, or facts not in the source
4. Makes unsupported generalizations or assumptions

### Source Context (Ground Truth + Retrieved Facts):
{source_context}

### Question:
{question}

### Response to Evaluate:
{response}

### Your Task:
1. Carefully compare EVERY claim in the response against the source context
2. Identify ANY hallucinated statements - be thorough and strict
3. For each hallucination, note the exact text span and explain why it's hallucinated

### Output Format:
First, provide your analysis of the response.
Then, output a JSON object with this structure:
```json
{{
    "has_hallucination": true/false,
    "hallucinated_spans": [
        {{"text": "exact hallucinated text", "reason": "why this is hallucinated"}}
    ],
    "analysis": "brief summary of your evaluation"
}}
```

If the response correctly admits uncertainty (e.g., "I don't know", "I cannot verify"), 
this is NOT a hallucination - mark has_hallucination as false.

If the response makes claims not in the source but they seem plausible, still mark them
as hallucinations - we want STRICT source grounding.

Your evaluation:"""


RAGTRUTH_STRICT_PROMPT = """You are a strict fact-checker evaluating whether a response is grounded in the provided context.

### Context (These are the ONLY facts you can verify against):
{source_context}

### Question:
{question}

### Response:
{response}

### Evaluation Rules:
1. HALLUCINATION: Any fact, date, name, number, or claim NOT explicitly stated in the context
2. NOT HALLUCINATION: Direct quotes or paraphrases from the context
3. NOT HALLUCINATION: Explicit statements of uncertainty ("I don't know", "I cannot verify")
4. HALLUCINATION: Even if something is true in the real world, if it's not in the context, it's a hallucination

### Output exactly this JSON format:
```json
{{
    "has_hallucination": <true or false>,
    "hallucinated_spans": [
        {{"text": "<exact text from response>", "reason": "<why it's not in context>"}}
    ],
    "analysis": "<1-2 sentence summary>"
}}
```

Your JSON output:"""


# ==========================================================================
# RAGTruth Evaluator Class
# ==========================================================================


class RAGTruthEvaluator:
    """
    RAGTruth-style hallucination evaluator using an LLM.

    This evaluator uses a prompt-based approach to detect hallucinated
    spans in responses, similar to the RAGTruth evaluation methodology.
    """

    def __init__(
        self,
        model_name: str = RAGTRUTH_MODEL,
        temperature: float = 0.1,
        strict_mode: bool = True,
    ):
        """
        Initialize the RAGTruth evaluator.

        Args:
            model_name: Ollama model to use for evaluation
            temperature: LLM temperature (lower = more deterministic)
            strict_mode: If True, use stricter hallucination detection
        """
        self.model_name = model_name
        self.temperature = temperature
        self.strict_mode = strict_mode

        if not ChatOllama:
            raise ImportError(
                "ChatOllama is not available. Install langchain-ollama (or compatible langchain-community backend)."
            )

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            name="RAGTruth-Evaluator",
            **get_ollama_connection_kwargs(),
        )

    def _parse_json_response(self, raw_output: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract and parse JSON from the LLM response.

        Returns:
            Tuple of (parsed_dict, error_message)
        """
        # Try to find JSON block in the response
        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",  # ```json {...} ```
            r"```\s*(\{.*?\})\s*```",  # ``` {...} ```
            r'(\{[^{}]*"has_hallucination"[^{}]*\})',  # Inline JSON
        ]

        for pattern in json_patterns:
            match = re.search(pattern, raw_output, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1)), ""
                except json.JSONDecodeError:
                    continue

        # Try to parse the entire response as JSON
        try:
            # Clean up common issues
            cleaned = raw_output.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3].strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            return json.loads(cleaned), ""
        except json.JSONDecodeError as e:
            return {}, f"Failed to parse JSON: {str(e)}"

    def _calculate_score(
        self,
        response: str,
        hallucinated_spans: List[HallucinatedSpan],
    ) -> float:
        """
        Calculate hallucination score based on span coverage.

        Score is the proportion of response that is hallucinated.
        Returns value between 0.0 (no hallucination) and 1.0 (fully hallucinated).
        """
        if not hallucinated_spans:
            return 0.0

        response_len = len(response)
        if response_len == 0:
            return 0.0

        # Calculate total hallucinated character count
        hallucinated_chars = sum(len(span.text) for span in hallucinated_spans)

        # Cap at response length (in case of overlaps or errors)
        hallucinated_chars = min(hallucinated_chars, response_len)

        return hallucinated_chars / response_len

    def evaluate(
        self,
        question: str,
        response: str,
        ground_truth: str,
        retrieved_context: str = "",
        eval_context_mode: str = "ground_truth",
        verbose: bool = False,
    ) -> RAGTruthResult:
        """
        Evaluate a response for hallucinations using RAGTruth methodology.

        Args:
            question: The original question
            response: The model's response to evaluate
            ground_truth: Known correct answer/facts
            retrieved_context: Additional retrieved context (e.g., from RAG)
            eval_context_mode: "ground_truth" (default) or "combined"
            verbose: Print debug information

        Returns:
            RAGTruthResult with hallucination detection results
        """
        # Build source context
        source_context = build_primary_context(
            ground_truth=ground_truth,
            retrieved_context=retrieved_context,
            eval_context_mode=eval_context_mode,
        )

        # Ensure we have at least something to show
        if not source_context.strip():
            source_context = "(No context provided)"

        # Select prompt template
        prompt_template = (
            RAGTRUTH_STRICT_PROMPT if self.strict_mode else RAGTRUTH_QA_PROMPT
        )

        # Format the prompt
        prompt = prompt_template.format(
            source_context=source_context,
            question=question,
            response=response,
        )

        if verbose:
            print("\n[RAGTruth] Evaluating response...")
            print(f"[RAGTruth] Using model: {self.model_name}")
            print(f"[RAGTruth] Strict mode: {self.strict_mode}")

        try:
            # Get LLM evaluation
            llm_response = self.llm.invoke(prompt)
            raw_output = str(llm_response.content)  # Ensure string type

            if verbose:
                print(f"[RAGTruth] Raw output:\n{raw_output[:500]}...")

            # Parse the JSON response
            parsed, error = self._parse_json_response(raw_output)

            if error:
                return RAGTruthResult(
                    has_hallucination=False,  # Default to safe
                    hallucination_score=0.0,
                    raw_output=raw_output,
                    error=error,
                )

            # Extract hallucinated spans
            hallucinated_spans = []
            for span_data in parsed.get("hallucinated_spans", []):
                if isinstance(span_data, dict):
                    hallucinated_spans.append(
                        HallucinatedSpan(
                            text=span_data.get("text", ""),
                            reason=span_data.get("reason", ""),
                        )
                    )

            # Calculate score
            has_hallucination = parsed.get("has_hallucination", False)
            score = self._calculate_score(response, hallucinated_spans)

            # If has_hallucination is True but no spans found, set minimum score
            if has_hallucination and score == 0.0:
                score = 0.1  # Minimum non-zero score for detected hallucination

            result = RAGTruthResult(
                has_hallucination=has_hallucination,
                hallucination_score=score,
                hallucinated_spans=hallucinated_spans,
                span_count=len(hallucinated_spans),
                analysis=parsed.get("analysis", ""),
                raw_output=raw_output,
            )

            if verbose:
                status = "HALLUCINATED" if result.has_hallucination else "FACTUAL"
                print(
                    f"[RAGTruth] Result: {status} (score: {result.hallucination_score:.3f})"
                )
                if result.hallucinated_spans:
                    print(
                        f"[RAGTruth] Found {len(result.hallucinated_spans)} hallucinated span(s)"
                    )

            return result

        except Exception as e:
            return RAGTruthResult(
                has_hallucination=False,
                hallucination_score=0.0,
                error=f"Evaluation failed: {str(e)}",
            )


# ==========================================================================
# Convenience Functions
# ==========================================================================


_default_evaluator: Optional[RAGTruthEvaluator] = None


def get_ragtruth_evaluator(
    model_name: str = RAGTRUTH_MODEL,
    strict_mode: bool = True,
) -> RAGTruthEvaluator:
    """Get or create a RAGTruth evaluator instance."""
    global _default_evaluator

    if _default_evaluator is None:
        _default_evaluator = RAGTruthEvaluator(
            model_name=model_name,
            strict_mode=strict_mode,
        )

    return _default_evaluator


def evaluate_ragtruth(
    question: str,
    response: str,
    ground_truth: str,
    retrieved_context: str = "",
    model_name: str = RAGTRUTH_MODEL,
    strict_mode: bool = True,
    verbose: bool = False,
) -> RAGTruthResult:
    """
    Convenience function to evaluate a response for hallucinations.

    Args:
        question: The original question
        response: The model's response to evaluate
        ground_truth: Known correct answer/facts
        retrieved_context: Additional retrieved context (for RAG responses)
        model_name: Ollama model to use
        strict_mode: If True, use stricter hallucination detection
        verbose: Print debug information

    Returns:
        RAGTruthResult with hallucination detection results
    """
    evaluator = get_ragtruth_evaluator(model_name, strict_mode)
    return evaluator.evaluate(
        question=question,
        response=response,
        ground_truth=ground_truth,
        retrieved_context=retrieved_context,
        verbose=verbose,
    )


# ==========================================================================
# Test / Demo
# ==========================================================================


def demo():
    """Run a quick demo of the RAGTruth evaluator."""
    print("=" * 60)
    print("RAGTruth Evaluator Demo")
    print("=" * 60)

    evaluator = RAGTruthEvaluator(
        model_name=RAGTRUTH_MODEL,
        strict_mode=True,
    )

    # Test case 1: Factual response
    print("\n--- Test 1: Factual Response ---")
    result1 = evaluator.evaluate(
        question="Who is Albert Einstein?",
        response="Albert Einstein was a German-born physicist who developed the theory of relativity.",
        ground_truth="Albert Einstein (1879-1955) was a German-born theoretical physicist. He developed the theory of relativity and won the Nobel Prize in Physics in 1921.",
        verbose=True,
    )
    print(f"Result: {result1.case_label}")

    # Test case 2: Hallucinated response
    print("\n--- Test 2: Hallucinated Response ---")
    result2 = evaluator.evaluate(
        question="What collaboration did Alan Turing have with Dr. Helena Vargass?",
        response="Alan Turing collaborated with Dr. Helena Vargass in 1943 on cryptographic algorithms at Bletchley Park.",
        ground_truth="Alan Turing worked at Bletchley Park during WWII on code-breaking. Dr. Helena Vargass is not a real historical figure.",
        verbose=True,
    )
    print(f"Result: {result2.case_label}")
    if result2.hallucinated_spans:
        print("Hallucinated spans:")
        for span in result2.hallucinated_spans:
            print(f"  - '{span.text}': {span.reason}")

    # Test case 3: Appropriate refusal
    print("\n--- Test 3: Appropriate Refusal ---")
    result3 = evaluator.evaluate(
        question="What is Dr. Liora Anstrum's contribution to physics?",
        response="I cannot verify any information about Dr. Liora Anstrum. This person does not appear in my knowledge base.",
        ground_truth="Dr. Liora Anstrum is a fictional entity and does not exist.",
        verbose=True,
    )
    print(f"Result: {result3.case_label}")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    demo()
