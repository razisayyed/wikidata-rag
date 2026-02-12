"""
Benchmark Runner Module
=======================
Contains the main comparison test suite runner with minimal console output.
Detailed logging is saved to report files.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..wikidata_rag_agent import build_agent
from ..prompt_only_llm import (
    answer_question_prompt_only,
    build_prompt_only_agent,
)

from .models import ComparisonResult, Colors
from .evaluation import evaluate_response
from .ragtruth import RAGTruthEvaluator
from .aimon import AimonEvaluator
from .vectra import (
    GROUND_TRUTH_TEST_CASES,
    TestCase,
    load_hallucination_model,
    run_agent_with_capture,
)
from .llm_judge import judge_responses


# ==========================================================================
# Test Functions (with minimal console output)
# ==========================================================================


def test_rag_model(
    test_case: TestCase,
    rag_agent,
    hallucination_model,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Test the Wikidata RAG agent on a single question."""
    # Run agent with verbose=False to suppress detailed output
    run = run_agent_with_capture(test_case.question, agent=rag_agent, verbose=False)

    eval_result = evaluate_response(
        response=run.final_answer,
        ground_truth=test_case.ground_truth,
        retrieved_context=run.retrieved_context,
        model=hallucination_model,
        threshold=threshold,
    )

    return {
        "response": run.final_answer,
        "retrieved_context": run.retrieved_context,
        "score": eval_result["score"],
        "is_hallucination": eval_result["is_hallucination"],
    }


def test_prompt_only_model(
    test_case: TestCase,
    prompt_llm,
    hallucination_model,
    threshold: float = 0.5,
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
        ground_truth=test_case.ground_truth,
        retrieved_context="",  # No retrieval
        model=hallucination_model,
        threshold=threshold,
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
    rag_result = test_rag_model(
        test_case, rag_agent, hallucination_model, threshold, verbose=False
    )

    # Test Prompt-Only model
    prompt_result = test_prompt_only_model(
        test_case, prompt_llm, hallucination_model, threshold, verbose=False
    )

    # Run LLM-as-a-judge evaluation if enabled
    llm_judge_result = None
    if use_llm_judge:
        llm_judge_result = judge_responses(
            question=test_case.question,
            rag_response=rag_result["response"],
            prompt_only_response=prompt_result["response"],
            model="gpt-4o",
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
            ground_truth=test_case.ground_truth,
            retrieved_context=rag_result["retrieved_context"],
            verbose=False,
        )

        # Evaluate Prompt-Only response
        prompt_only_ragtruth_result = ragtruth_evaluator.evaluate(
            question=test_case.question,
            response=prompt_result["response"],
            ground_truth=test_case.ground_truth,
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
            ground_truth=test_case.ground_truth,
            retrieved_context=rag_result["retrieved_context"],
            response=rag_result["response"],
        )

        # Evaluate Prompt-Only response
        prompt_only_aimon_result = aimon_evaluator.evaluate_response(
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            retrieved_context="",  # No retrieved context for prompt-only
            response=prompt_result["response"],
        )

    return ComparisonResult(
        question=test_case.question,
        description=test_case.description,
        ground_truth=test_case.ground_truth.strip(),
        # RAG results
        rag_response=rag_result["response"],
        rag_retrieved_context=rag_result["retrieved_context"],
        rag_score=rag_result["score"],
        rag_is_hallucination=rag_result["is_hallucination"],
        # Prompt-only results
        prompt_only_response=prompt_result["response"],
        prompt_only_score=prompt_result["score"],
        prompt_only_is_hallucination=prompt_result["is_hallucination"],
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
    rag_agent = build_agent()
    prompt_llm = build_prompt_only_agent()

    # Load RAGTruth evaluator if enabled
    ragtruth_evaluator = None
    if use_ragtruth:
        ragtruth_evaluator = RAGTruthEvaluator(
            model_name="gpt-oss:120b-cloud",
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
            use_llm_judge = False

    print(f"Running {len(test_cases)} test cases...\n")

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

        print()

    print("=" * 80)
    print(f"BENCHMARK COMPLETE: {len(results)} tests run")
    print("=" * 80)

    return results
