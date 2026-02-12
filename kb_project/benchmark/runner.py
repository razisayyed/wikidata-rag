"""
Benchmark Runner Module
=======================
Contains the main comparison test suite runner with minimal console output.
Detailed logging is saved to report files.
"""

from __future__ import annotations

import os
import shutil
import textwrap
from typing import Any, Dict, List, Optional

from ..wikidata_rag_agent import build_agent
from ..prompt_only_llm import (
    answer_question_prompt_only,
    build_prompt_only_agent,
)
from ..settings import OPENAI_JUDGE_MODEL, RAGTRUTH_MODEL

from .models import ComparisonResult, Colors
from .evaluation import evaluate_response, evaluate_rag_faithfulness
from .ragtruth import RAGTruthEvaluator
from .aimon import AimonEvaluator
from .vectra import (
    GROUND_TRUTH_TEST_CASES,
    TestCase,
    load_hallucination_model,
    run_agent_with_capture,
)
from .llm_judge import judge_responses

VALID_GROUND_TRUTH_STYLES = {"concise", "rich"}


# ==========================================================================
# Test Functions (with minimal console output)
# ==========================================================================


def build_reference_ground_truth(
    test_case: TestCase,
    ground_truth_style: str = "concise",
    max_ground_truth_facts: Optional[int] = None,
) -> str:
    """
    Build benchmark reference context from the test case.

    Styles:
    - concise: canonical answer only (default, fairer for short answers).
    - rich: canonical answer plus key-fact bullets.
    """
    canonical = test_case.ground_truth.strip()
    style = (ground_truth_style or "concise").strip().lower()
    if style not in VALID_GROUND_TRUTH_STYLES:
        style = "concise"

    if style == "concise":
        return canonical

    key_facts = [fact.strip() for fact in test_case.key_facts if fact and fact.strip()]
    if max_ground_truth_facts is not None and max_ground_truth_facts > 0:
        key_facts = key_facts[:max_ground_truth_facts]

    if not key_facts:
        return canonical

    parts: List[str] = [canonical]
    if key_facts:
        parts.append("Key facts:")
        parts.extend(f"- {fact}" for fact in key_facts)

    return "\n".join(parts).strip()


def _render_three_column_console_table(
    ground_truth: str,
    rag_output: str,
    prompt_output: str,
) -> str:
    """Render a wrapped console table with columns: Truth | RAG | Prompt-Only."""
    terminal_width = shutil.get_terminal_size(fallback=(180, 24)).columns
    table_width = max(120, terminal_width)
    inner_width = table_width - 4  # border + separators
    col_width = max(24, inner_width // 3)

    # Keep exact table width aligned with computed columns.
    table_width = (col_width * 3) + 4

    def _wrap_cell(text: str) -> List[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [p for p in normalized.split("\n") if p.strip()] or [""]
        lines: List[str] = []
        for paragraph in paragraphs:
            wrapped = textwrap.wrap(
                paragraph,
                width=col_width,
                break_long_words=True,
                break_on_hyphens=False,
            )
            lines.extend(wrapped or [""])
        return lines

    gt_lines = _wrap_cell(ground_truth)
    rag_lines = _wrap_cell(rag_output)
    prompt_lines = _wrap_cell(prompt_output)
    row_count = max(len(gt_lines), len(rag_lines), len(prompt_lines))

    def _line(left: str, middle: str, right: str) -> str:
        return (
            f"|{left.ljust(col_width)}|{middle.ljust(col_width)}|{right.ljust(col_width)}|"
        )

    border = "+" + ("-" * col_width) + "+" + ("-" * col_width) + "+" + ("-" * col_width) + "+"
    header = _line("GROUND TRUTH", "RAG OUTPUT", "PROMPT-ONLY OUTPUT")

    rows = [border, header, border]
    for idx in range(row_count):
        rows.append(
            _line(
                gt_lines[idx] if idx < len(gt_lines) else "",
                rag_lines[idx] if idx < len(rag_lines) else "",
                prompt_lines[idx] if idx < len(prompt_lines) else "",
            )
        )
    rows.append(border)
    return "\n".join(rows)


def test_rag_model(
    test_case: TestCase,
    reference_ground_truth: str,
    rag_agent,
    hallucination_model,
    threshold: float = 0.5,
    eval_context_mode: str = "ground_truth",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Test the Wikidata RAG agent on a single question."""
    # Run agent with verbose=False to suppress detailed output
    run = run_agent_with_capture(test_case.question, agent=rag_agent, verbose=False)

    primary_retrieved_context = run.retrieved_context if eval_context_mode == "combined" else ""

    eval_result = evaluate_response(
        response=run.final_answer,
        ground_truth=reference_ground_truth,
        retrieved_context=primary_retrieved_context,
        model=hallucination_model,
        threshold=threshold,
        eval_context_mode=eval_context_mode,
    )

    return {
        "response": run.final_answer,
        "retrieved_context": run.retrieved_context,
        "sanitized_retrieved_context": run.sanitized_retrieved_context,
        "score": eval_result["score"],
        "is_hallucination": eval_result["is_hallucination"],
    }


def test_prompt_only_model(
    test_case: TestCase,
    reference_ground_truth: str,
    prompt_llm,
    hallucination_model,
    threshold: float = 0.5,
    eval_context_mode: str = "ground_truth",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Test the prompt-only agent on a single question."""
    # Run with verbose=False to suppress detailed output
    response = answer_question_prompt_only(
        test_case.question,
        llm=prompt_llm,
        verbose=False,
    )

    # No retrieved context for prompt-only
    eval_result = evaluate_response(
        response=response,
        ground_truth=reference_ground_truth,
        retrieved_context="",  # No retrieval
        model=hallucination_model,
        threshold=threshold,
        eval_context_mode=eval_context_mode,
    )

    return {
        "response": response,
        "score": eval_result["score"],
        "is_hallucination": eval_result["is_hallucination"],
    }


def test_both_models(
    test_case: TestCase,
    rag_agent,
    prompt_llm,
    hallucination_model,
    ragtruth_evaluator: Optional[RAGTruthEvaluator] = None,
    aimon_evaluator: Optional[AimonEvaluator] = None,
    threshold: float = 0.5,
    eval_context_mode: str = "ground_truth",
    ground_truth_style: str = "concise",
    max_ground_truth_facts: Optional[int] = None,
    compute_rag_faithfulness: bool = True,
    use_llm_judge: bool = True,
    use_ragtruth: bool = True,
    use_aimon: bool = True,
    verbose: bool = True,
) -> ComparisonResult:
    """
    Run the same question through both models and compare results.

    Console output is minimal - shows only question and scores.
    Detailed information is saved to report files.
    """
    # Test RAG model
    reference_ground_truth = build_reference_ground_truth(
        test_case=test_case,
        ground_truth_style=ground_truth_style,
        max_ground_truth_facts=max_ground_truth_facts,
    )

    # Test RAG model
    rag_result = test_rag_model(
        test_case,
        reference_ground_truth,
        rag_agent,
        hallucination_model,
        threshold,
        eval_context_mode=eval_context_mode,
        verbose=False,
    )

    # Test Prompt-Only model
    prompt_result = test_prompt_only_model(
        test_case,
        reference_ground_truth,
        prompt_llm,
        hallucination_model,
        threshold,
        eval_context_mode=eval_context_mode,
        verbose=False,
    )

    rag_primary_retrieved_context = (
        rag_result["retrieved_context"] if eval_context_mode == "combined" else ""
    )

    # Run LLM-as-a-judge evaluation if enabled
    llm_judge_result = None
    if use_llm_judge:
        llm_judge_result = judge_responses(
            question=test_case.question,
            rag_response=rag_result["response"],
            prompt_only_response=prompt_result["response"],
            model=OPENAI_JUDGE_MODEL,
            verbose=False,
        )

    # Run RAGTruth evaluation if enabled
    rag_ragtruth_result = None
    prompt_only_ragtruth_result = None

    if use_ragtruth and ragtruth_evaluator is not None:
        # Evaluate RAG response
        rag_ragtruth_result = ragtruth_evaluator.evaluate(
            question=test_case.question,
            response=rag_result["response"],
            ground_truth=reference_ground_truth,
            retrieved_context=rag_primary_retrieved_context,
            verbose=False,
        )

        # Evaluate Prompt-Only response
        prompt_only_ragtruth_result = ragtruth_evaluator.evaluate(
            question=test_case.question,
            response=prompt_result["response"],
            ground_truth=reference_ground_truth,
            retrieved_context="",  # No retrieved context for prompt-only
            verbose=False,
        )

    # Run AIMon evaluation if enabled
    rag_aimon_result = None
    prompt_only_aimon_result = None

    if use_aimon and aimon_evaluator is not None:
        # Evaluate RAG response
        rag_aimon_result = aimon_evaluator.evaluate_response(
            question=test_case.question,
            ground_truth=reference_ground_truth,
            retrieved_context=rag_primary_retrieved_context,
            response=rag_result["response"],
        )

        # Evaluate Prompt-Only response
        prompt_only_aimon_result = aimon_evaluator.evaluate_response(
            question=test_case.question,
            ground_truth=reference_ground_truth,
            retrieved_context="",  # No retrieved context for prompt-only
            response=prompt_result["response"],
        )

    rag_faithfulness_score = None
    rag_faithfulness_is_hallucination = None
    if compute_rag_faithfulness:
        rag_faithfulness_result = evaluate_rag_faithfulness(
            response=rag_result["response"],
            retrieved_context=rag_result["sanitized_retrieved_context"],
            model=hallucination_model,
            threshold=threshold,
        )
        if rag_faithfulness_result is not None:
            rag_faithfulness_score = rag_faithfulness_result["score"]
            rag_faithfulness_is_hallucination = rag_faithfulness_result[
                "is_hallucination"
            ]

    return ComparisonResult(
        question=test_case.question,
        description=test_case.description,
        ground_truth=reference_ground_truth,
        # RAG results
        rag_response=rag_result["response"],
        rag_retrieved_context=rag_result["retrieved_context"],
        rag_score=rag_result["score"],
        rag_is_hallucination=rag_result["is_hallucination"],
        # Prompt-only results
        prompt_only_response=prompt_result["response"],
        prompt_only_score=prompt_result["score"],
        prompt_only_is_hallucination=prompt_result["is_hallucination"],
        evaluation_mode=eval_context_mode,
        rag_faithfulness_score=rag_faithfulness_score,
        rag_faithfulness_is_hallucination=rag_faithfulness_is_hallucination,
        # LLM Judge results
        llm_judge_result=llm_judge_result,
        # RAGTruth results
        rag_ragtruth_result=rag_ragtruth_result,
        prompt_only_ragtruth_result=prompt_only_ragtruth_result,
        # AIMon results
        rag_aimon_result=rag_aimon_result,
        prompt_only_aimon_result=prompt_only_aimon_result,
    )


# ==========================================================================
# Test Suite Runner
# ==========================================================================


def run_comparison_suite(
    test_cases: Optional[List[TestCase]] = None,
    threshold: float = 0.5,
    eval_context_mode: str = "ground_truth",
    ground_truth_style: str = "concise",
    max_ground_truth_facts: Optional[int] = None,
    benchmark_temperature: float = 0.0,
    compute_rag_faithfulness: bool = True,
    use_llm_judge: bool = True,
    use_ragtruth: bool = True,
    use_aimon: bool = True,
    verbose: bool = True,
) -> List[ComparisonResult]:
    """
    Run the full comparison test suite.

    Console output is minimal - shows only:
    - Current question (truncated)
    - RAG Score
    - Prompt-Only Score
    - RAGTruth Decision (if enabled)
    - AIMon Decision (if enabled)

    Detailed information is saved to report files.
    """
    if test_cases is None:
        test_cases = GROUND_TRUTH_TEST_CASES

    print("Loading models...")

    # Load models once
    hallucination_model = load_hallucination_model()
    rag_agent = build_agent(temperature=benchmark_temperature)
    prompt_llm = build_prompt_only_agent(temperature=benchmark_temperature)

    # Load RAGTruth evaluator if enabled
    ragtruth_evaluator = None
    if use_ragtruth:
        ragtruth_evaluator = RAGTruthEvaluator(
            model_name=RAGTRUTH_MODEL,
            strict_mode=False,
        )

    # Load AIMon evaluator if enabled
    aimon_evaluator = None
    if use_aimon:
        try:
            aimon_evaluator = AimonEvaluator(threshold=threshold)
            aimon_evaluator.load_model()
        except ImportError:
            print("Warning: hdm2 package not installed. AIMon evaluation disabled.")
            use_aimon = False

    if use_llm_judge:
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "Warning: --llm-judge requested but OPENAI_API_KEY is not set. "
                "LLM Judge evaluation disabled."
            )
            use_llm_judge = False

    print(f"Running {len(test_cases)} test cases...\n")
    normalized_gt_style = (ground_truth_style or "concise").strip().lower()
    if normalized_gt_style not in VALID_GROUND_TRUTH_STYLES:
        normalized_gt_style = "concise"

    print(f"Primary evaluation mode: {eval_context_mode}")
    print(f"Ground-truth style: {normalized_gt_style}")
    if normalized_gt_style == "rich" and max_ground_truth_facts:
        print(f"Ground-truth fact cap: {max_ground_truth_facts}")
    print(f"Benchmark temperature: {benchmark_temperature}\n")

    results: List[ComparisonResult] = []

    for i, test_case in enumerate(test_cases, 1):
        result = test_both_models(
            test_case=test_case,
            rag_agent=rag_agent,
            prompt_llm=prompt_llm,
            hallucination_model=hallucination_model,
            ragtruth_evaluator=ragtruth_evaluator,
            aimon_evaluator=aimon_evaluator,
            threshold=threshold,
            eval_context_mode=eval_context_mode,
            ground_truth_style=normalized_gt_style,
            max_ground_truth_facts=max_ground_truth_facts,
            compute_rag_faithfulness=compute_rag_faithfulness,
            use_llm_judge=use_llm_judge,
            use_ragtruth=use_ragtruth,
            use_aimon=use_aimon,
            verbose=verbose,
        )
        results.append(result)

        # Block-style console output
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}TEST {i}/{len(test_cases)}: {test_case.description}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"Question: {test_case.question}")
        print()
        print(
            _render_three_column_console_table(
                result.ground_truth,
                result.rag_response,
                result.prompt_only_response,
            )
        )
        print()

        # Vectara results
        rag_status = (
            f"{Colors.RED}❌ HALLUC{Colors.RESET}"
            if result.rag_is_hallucination
            else f"{Colors.GREEN}✅ FACTUAL{Colors.RESET}"
        )
        prompt_status = (
            f"{Colors.RED}❌ HALLUC{Colors.RESET}"
            if result.prompt_only_is_hallucination
            else f"{Colors.GREEN}✅ FACTUAL{Colors.RESET}"
        )
        print(f"{Colors.BOLD}VECTARA:{Colors.RESET}")
        print(f"  RAG:    {result.rag_score:.3f} {rag_status}")
        print(f"  Prompt: {result.prompt_only_score:.3f} {prompt_status}")
        print(f"  Winner: {result.winner}")
        if result.rag_faithfulness_score is not None:
            faith_status = (
                f"{Colors.RED}❌ HALLUC{Colors.RESET}"
                if result.rag_faithfulness_is_hallucination
                else f"{Colors.GREEN}✅ FACTUAL{Colors.RESET}"
            )
            print(
                f"  RAG Faithfulness: {result.rag_faithfulness_score:.3f} {faith_status}"
            )

        # RAGTruth results
        if result.rag_ragtruth_result is not None:
            rag_rt = result.rag_ragtruth_result
            prompt_rt = result.prompt_only_ragtruth_result
            rag_rt_status = (
                f"{Colors.RED}❌ HALLUC{Colors.RESET}"
                if rag_rt.has_hallucination
                else f"{Colors.GREEN}✅ FACTUAL{Colors.RESET}"
            )
            prompt_rt_status = (
                f"{Colors.RED}❌ HALLUC{Colors.RESET}"
                if (prompt_rt and prompt_rt.has_hallucination)
                else f"{Colors.GREEN}✅ FACTUAL{Colors.RESET}"
            )
            print()
            print(f"{Colors.BOLD}RAGTRUTH:{Colors.RESET}")
            print(
                f"  RAG:    score={rag_rt.hallucination_score:.3f}, spans={rag_rt.span_count} {rag_rt_status}"
            )
            if prompt_rt:
                print(
                    f"  Prompt: score={prompt_rt.hallucination_score:.3f}, spans={prompt_rt.span_count} {prompt_rt_status}"
                )
            print(f"  Winner: {result.ragtruth_winner}")

        # AIMon results
        if result.rag_aimon_result is not None:
            rag_am = result.rag_aimon_result
            prompt_am = result.prompt_only_aimon_result
            rag_am_status = (
                f"{Colors.RED}❌ HALLUC{Colors.RESET}"
                if rag_am.has_hallucination
                else f"{Colors.GREEN}✅ FACTUAL{Colors.RESET}"
            )
            prompt_am_status = (
                f"{Colors.RED}❌ HALLUC{Colors.RESET}"
                if (prompt_am and prompt_am.has_hallucination)
                else f"{Colors.GREEN}✅ FACTUAL{Colors.RESET}"
            )
            print()
            print(f"{Colors.BOLD}AIMON HDM-2:{Colors.RESET}")
            print(
                f"  RAG:    severity={rag_am.hallucination_severity:.3f}, sentences={len(rag_am.hallucinated_sentences)} {rag_am_status}"
            )
            if prompt_am:
                print(
                    f"  Prompt: severity={prompt_am.hallucination_severity:.3f}, sentences={len(prompt_am.hallucinated_sentences)} {prompt_am_status}"
                )
            print(f"  Winner: {result.aimon_winner}")

        # LLM Judge results
        if result.llm_judge_result is not None:
            judge = result.llm_judge_result
            print()
            print(f"{Colors.BOLD}LLM JUDGE ({OPENAI_JUDGE_MODEL}):{Colors.RESET}")
            if judge.error:
                print(f"  Error: {judge.error}")
            else:
                rag_status = (
                    f"{Colors.RED}❌ HALLUC{Colors.RESET}"
                    if judge.rag_has_hallucination
                    else f"{Colors.GREEN}✅ FACTUAL{Colors.RESET}"
                )
                prompt_status = (
                    f"{Colors.RED}❌ HALLUC{Colors.RESET}"
                    if judge.prompt_has_hallucination
                    else f"{Colors.GREEN}✅ FACTUAL{Colors.RESET}"
                )
                print(f"  RAG:    {rag_status}")
                print(f"  Prompt: {prompt_status}")
                print(f"  Winner: {result.llm_judge_winner} ({judge.confidence})")

        print()

    print("=" * 80)
    print(f"BENCHMARK COMPLETE: {len(results)} tests run")
    print("=" * 80)

    return results
