"""Legacy-simple benchmark runner with independent evaluator tracks."""

from __future__ import annotations

import os
import shutil
import textwrap
from typing import Dict, List, Optional

from equivalence_evaluator.evaluator import EquivalenceEvaluator, is_equivalence_method
from equivalence_evaluator.utils import factual_judge_adapter

from ..prompt_only_llm import answer_question_prompt_only, build_prompt_only_agent
from ..settings import LANGSMITH_TRACE_MODE, OPENAI_JUDGE_MODEL, RAGTRUTH_MODEL
from ..utils.langsmith import langsmith_tracing, normalize_trace_mode, should_trace_component
from ..wikidata_rag_agent import build_agent
from .aimon import AimonEvaluator
from .llm_judge import judge_responses
from .models import (
    AIMON_WINNER_EPSILON,
    ANALYSIS_VERSION,
    CaseResult,
    Colors,
    EvaluatorResult,
    ModelOutput,
    SuiteResult,
    TestCase,
    label_from_hallucination_flag,
    winner_from_labels,
)
from .ragtruth import RAGTruthEvaluator
from .vectra import GROUND_TRUTH_TEST_CASES, run_agent_with_capture

EVALUATOR_ORDER = ["vectara", "aimon", "llm_judge", "ragtruth"]
_REFUSAL_MARKERS = (
    "i cannot verify",
    "i can't verify",
    "cannot be verified",
    "could not be verified",
    "i cannot determine",
    "cannot determine",
    "i don't know",
    "i do not know",
    "unknown",
    "no verified",
)
_REFUSAL_EXPECTED_HINTS = (
    "fictional",
    "does not exist",
    "not a real",
    "no verified",
    "cannot verify",
    "cannot be established",
    "no real-world record",
    "no reliable record",
)


def _is_refusal_response(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _is_refusal_expected_case(test_case: TestCase) -> bool:
    if bool(getattr(test_case, "refusal_expected", False)):
        return True
    context = " ".join([test_case.question or "", test_case.ground_truth or ""]).lower()
    return any(hint in context for hint in _REFUSAL_EXPECTED_HINTS)


def _resolve_winner_for_evaluator(result: EvaluatorResult) -> str:
    if result.name == "aimon":
        return winner_from_labels(
            rag_label=result.rag_label,
            baseline_label=result.baseline_label,
            rag_score=result.rag_score,
            baseline_score=result.baseline_score,
            lower_is_better=True,
            epsilon=AIMON_WINNER_EPSILON,
        )
    if result.name == "ragtruth":
        return winner_from_labels(
            rag_label=result.rag_label,
            baseline_label=result.baseline_label,
            rag_score=result.rag_score,
            baseline_score=result.baseline_score,
            lower_is_better=True,
        )
    if result.name == "vectara":
        return winner_from_labels(
            rag_label=result.rag_label,
            baseline_label=result.baseline_label,
            rag_score=result.rag_score,
            baseline_score=result.baseline_score,
            lower_is_better=False,
        )
    return winner_from_labels(
        rag_label=result.rag_label,
        baseline_label=result.baseline_label,
    )


def _apply_refusal_policy(
    result: EvaluatorResult,
    test_case: TestCase,
    rag_response: str,
    baseline_response: str,
) -> EvaluatorResult:
    if result.status != "completed":
        return result

    refusal_expected = _is_refusal_expected_case(test_case)
    rag_is_refusal = _is_refusal_response(rag_response)
    baseline_is_refusal = _is_refusal_response(baseline_response)
    policy_notes: List[str] = []

    if rag_is_refusal:
        new_rag_label = "factual" if refusal_expected else "hallucinated"
        if result.rag_label != new_rag_label:
            result.rag_label = new_rag_label
            policy_notes.append(f"RAG refusal relabeled as {new_rag_label}")

    if baseline_is_refusal:
        new_baseline_label = "factual" if refusal_expected else "hallucinated"
        if result.baseline_label != new_baseline_label:
            result.baseline_label = new_baseline_label
            policy_notes.append(f"BASELINE refusal relabeled as {new_baseline_label}")

    if policy_notes:
        result.winner = _resolve_winner_for_evaluator(result)
        prefix = f"Refusal policy (refusal_expected={refusal_expected}). "
        suffix = "; ".join(policy_notes)
        result.notes = f"{result.notes} | {prefix}{suffix}" if result.notes else f"{prefix}{suffix}"

    return result


def build_reference_ground_truth(test_case: TestCase, include_aliases: bool = False) -> str:
    """Build canonical ground-truth context, with optional alias hints."""
    canonical = (test_case.ground_truth or "").strip()
    if not include_aliases:
        return canonical

    alias_lines: List[str] = []
    for group in test_case.accepted_aliases:
        values = [str(value).strip() for value in group if str(value).strip()]
        if len(values) < 2:
            continue
        alias_lines.append(f"- {values[0]} ~= {', '.join(values[1:])}")

    if not alias_lines:
        return canonical

    return "\n".join([canonical, "", "Accepted equivalent wording:", *alias_lines]).strip()


def _to_model_output(response: str, retrieved_context: str = "", tool_calls=None) -> ModelOutput:
    serialized_calls = []
    for tool_call in tool_calls or []:
        serialized_calls.append(
            {
                "name": tool_call.name,
                "args": tool_call.args,
                "output": tool_call.output,
            }
        )
    return ModelOutput(
        response=(response or "").strip(),
        retrieved_context=(retrieved_context or "").strip(),
        tool_calls=serialized_calls,
    )


def _skipped_result(name: str, reason: str) -> EvaluatorResult:
    return EvaluatorResult(
        name=name,
        status="skipped",
        rag_label="skipped",
        baseline_label="skipped",
        winner="N/A",
        notes=reason,
    )


def _error_result(name: str, reason: str) -> EvaluatorResult:
    return EvaluatorResult(
        name=name,
        status="error",
        rag_label="error",
        baseline_label="error",
        winner="N/A",
        notes=reason,
    )


def _evaluate_vectara(
    rag_output: ModelOutput,
    baseline_output: ModelOutput,
    reference_ground_truth: str,
    threshold: float,  # preserved for stable internal API
    hallucination_model,  # preserved for stable internal API
    equivalence_evaluator: EquivalenceEvaluator,
) -> EvaluatorResult:
    _ = threshold, hallucination_model
    try:
        rag_equiv = equivalence_evaluator.evaluate(
            ground_truth=reference_ground_truth,
            rag_output=rag_output.response,
            baseline_output=baseline_output.response,
        )
        baseline_equiv = equivalence_evaluator.evaluate(
            ground_truth=reference_ground_truth,
            rag_output=baseline_output.response,
            baseline_output=rag_output.response,
        )

        rag_method = str(rag_equiv.get("method", ""))
        baseline_method = str(baseline_equiv.get("method", ""))
        original_rag_method = rag_method
        original_baseline_method = baseline_method
        forced_fair_method = False

        # Fairness guardrail: if methods differ, force both outputs through
        # the same final method so winner comparison stays apples-to-apples.
        if rag_method != baseline_method:
            forced_fair_method = True
            rag_forced = factual_judge_adapter(reference_ground_truth, rag_output.response)
            baseline_forced = factual_judge_adapter(
                reference_ground_truth, baseline_output.response
            )

            rag_equiv = {
                "equivalent": bool(rag_forced.get("equivalent", False)),
                "method": "factual_judge",
                "score": float(rag_forced.get("score", 0.0) or 0.0),
                "details": {
                    "forced_for_fair_comparison": True,
                    "original_method": rag_method,
                },
            }
            baseline_equiv = {
                "equivalent": bool(baseline_forced.get("equivalent", False)),
                "method": "factual_judge",
                "score": float(baseline_forced.get("score", 0.0) or 0.0),
                "details": {
                    "forced_for_fair_comparison": True,
                    "original_method": baseline_method,
                },
            }
            rag_method = "factual_judge"
            baseline_method = "factual_judge"

        rag_score = float(rag_equiv.get("score", 0.0) or 0.0)
        baseline_score = float(baseline_equiv.get("score", 0.0) or 0.0)

        if rag_method == "factual_judge":
            rag_label = "factual" if bool(rag_equiv.get("equivalent", False)) else "hallucinated"
        else:
            rag_label = "factual" if is_equivalence_method(rag_method) else "hallucinated"

        if baseline_method == "factual_judge":
            baseline_label = (
                "factual" if bool(baseline_equiv.get("equivalent", False)) else "hallucinated"
            )
        else:
            baseline_label = "factual" if is_equivalence_method(baseline_method) else "hallucinated"

        notes = "Equivalence pipeline. "
        if forced_fair_method:
            notes += (
                "Mixed methods detected; forced common method=factual_judge. "
                f"Original RAG method={original_rag_method}; "
                f"Original BASELINE method={original_baseline_method}"
            )
        else:
            notes += f"RAG method={rag_method}; BASELINE method={baseline_method}"
        return EvaluatorResult(
            name="vectara",
            status="completed",
            rag_label=rag_label,
            baseline_label=baseline_label,
            rag_score=rag_score,
            baseline_score=baseline_score,
            winner=winner_from_labels(
                rag_label=rag_label,
                baseline_label=baseline_label,
                rag_score=rag_score,
                baseline_score=baseline_score,
                lower_is_better=False,
            ),
            notes=notes,
        )
    except Exception as exc:
        return _skipped_result("vectara", f"Vectara evaluation unavailable: {exc}")


def _evaluate_aimon(
    rag_output: ModelOutput,
    baseline_output: ModelOutput,
    test_case: TestCase,
    reference_ground_truth: str,
    aimon_evaluator: Optional[AimonEvaluator],
) -> EvaluatorResult:
    if aimon_evaluator is None:
        return _skipped_result("aimon", "AIMon evaluator unavailable.")

    try:
        rag_result = aimon_evaluator.evaluate_response(
            question=test_case.question,
            ground_truth=reference_ground_truth,
            retrieved_context="",
            response=rag_output.response,
            eval_context_mode="ground_truth",
        )
        baseline_result = aimon_evaluator.evaluate_response(
            question=test_case.question,
            ground_truth=reference_ground_truth,
            retrieved_context="",
            response=baseline_output.response,
            eval_context_mode="ground_truth",
        )

        if rag_result.error or baseline_result.error:
            return _skipped_result(
                "aimon",
                f"AIMon evaluation failed: {rag_result.error or baseline_result.error}",
            )

        rag_label = label_from_hallucination_flag(rag_result.has_hallucination)
        baseline_label = label_from_hallucination_flag(baseline_result.has_hallucination)
        return EvaluatorResult(
            name="aimon",
            status="completed",
            rag_label=rag_label,
            baseline_label=baseline_label,
            rag_score=float(rag_result.hallucination_severity),
            baseline_score=float(baseline_result.hallucination_severity),
            winner=winner_from_labels(
                rag_label=rag_label,
                baseline_label=baseline_label,
                rag_score=float(rag_result.hallucination_severity),
                baseline_score=float(baseline_result.hallucination_severity),
                lower_is_better=True,
                epsilon=AIMON_WINNER_EPSILON,
            ),
            notes=(
                "Sentence-level hallucination severity. "
                f"RAG sentences={len(rag_result.hallucinated_sentences)}, "
                f"BASELINE sentences={len(baseline_result.hallucinated_sentences)}"
            ),
        )
    except Exception as exc:
        return _skipped_result("aimon", f"AIMon evaluation unavailable: {exc}")


def _evaluate_llm_judge(
    rag_output: ModelOutput,
    baseline_output: ModelOutput,
    test_case: TestCase,
    reference_ground_truth: str,
) -> EvaluatorResult:
    if not os.environ.get("OPENAI_API_KEY"):
        return _skipped_result("llm_judge", "OPENAI_API_KEY is not set.")

    try:
        judge = judge_responses(
            question=test_case.question,
            rag_response=rag_output.response,
            prompt_only_response=baseline_output.response,
            reference_context=reference_ground_truth,
            model=OPENAI_JUDGE_MODEL,
            verbose=False,
        )

        if judge.error:
            return _skipped_result("llm_judge", f"LLM judge unavailable: {judge.error}")

        rag_label = "hallucinated" if judge.rag_has_factual_error else "factual"
        baseline_label = "hallucinated" if judge.prompt_has_factual_error else "factual"

        winner = winner_from_labels(rag_label=rag_label, baseline_label=baseline_label)
        if winner == "Tie":
            if judge.winner == "RAG":
                winner = "RAG"
            elif judge.winner == "Prompt-Only":
                winner = "BASELINE"

        return EvaluatorResult(
            name="llm_judge",
            status="completed",
            rag_label=rag_label,
            baseline_label=baseline_label,
            winner=winner,
            notes=f"confidence={judge.confidence}; winner_hint={judge.winner}",
        )
    except Exception as exc:
        return _skipped_result("llm_judge", f"LLM judge unavailable: {exc}")


def _evaluate_ragtruth(
    rag_output: ModelOutput,
    baseline_output: ModelOutput,
    test_case: TestCase,
    reference_ground_truth: str,
    ragtruth_evaluator: Optional[RAGTruthEvaluator],
) -> EvaluatorResult:
    if ragtruth_evaluator is None:
        return _skipped_result("ragtruth", "RAGTruth evaluator unavailable.")

    try:
        rag_result = ragtruth_evaluator.evaluate(
            question=test_case.question,
            response=rag_output.response,
            ground_truth=reference_ground_truth,
            retrieved_context="",
            eval_context_mode="ground_truth",
            verbose=False,
        )
        baseline_result = ragtruth_evaluator.evaluate(
            question=test_case.question,
            response=baseline_output.response,
            ground_truth=reference_ground_truth,
            retrieved_context="",
            eval_context_mode="ground_truth",
            verbose=False,
        )

        if rag_result.error or baseline_result.error:
            return _skipped_result(
                "ragtruth",
                f"RAGTruth evaluation failed: {rag_result.error or baseline_result.error}",
            )

        rag_label = label_from_hallucination_flag(rag_result.has_hallucination)
        baseline_label = label_from_hallucination_flag(baseline_result.has_hallucination)
        return EvaluatorResult(
            name="ragtruth",
            status="completed",
            rag_label=rag_label,
            baseline_label=baseline_label,
            rag_score=float(rag_result.hallucination_score),
            baseline_score=float(baseline_result.hallucination_score),
            winner=winner_from_labels(
                rag_label=rag_label,
                baseline_label=baseline_label,
                rag_score=float(rag_result.hallucination_score),
                baseline_score=float(baseline_result.hallucination_score),
                lower_is_better=True,
            ),
            notes=f"RAG spans={rag_result.span_count}; BASELINE spans={baseline_result.span_count}",
        )
    except Exception as exc:
        return _skipped_result("ragtruth", f"RAGTruth evaluation unavailable: {exc}")


def _render_three_column_console_table(ground_truth: str, rag_output: str, baseline_output: str) -> str:
    terminal_width = shutil.get_terminal_size(fallback=(180, 24)).columns
    table_width = max(120, terminal_width)
    inner_width = table_width - 4
    col_width = max(24, inner_width // 3)

    def _wrap(text: str) -> List[str]:
        lines: List[str] = []
        for paragraph in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            lines.extend(
                textwrap.wrap(
                    paragraph,
                    width=col_width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return lines or [""]

    gt_lines = _wrap(ground_truth)
    rag_lines = _wrap(rag_output)
    baseline_lines = _wrap(baseline_output)
    row_count = max(len(gt_lines), len(rag_lines), len(baseline_lines))

    border = "+" + ("-" * col_width) + "+" + ("-" * col_width) + "+" + ("-" * col_width) + "+"

    def _line(left: str, middle: str, right: str) -> str:
        return f"|{left.ljust(col_width)}|{middle.ljust(col_width)}|{right.ljust(col_width)}|"

    rows = [border, _line("GROUND TRUTH", "RAG OUTPUT", "BASELINE OUTPUT"), border]
    for idx in range(row_count):
        rows.append(
            _line(
                gt_lines[idx] if idx < len(gt_lines) else "",
                rag_lines[idx] if idx < len(rag_lines) else "",
                baseline_lines[idx] if idx < len(baseline_lines) else "",
            )
        )
    rows.append(border)
    return "\n".join(rows)


def _print_case_console(case_result: CaseResult, index: int, total: int) -> None:
    case = case_result.test_case
    print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(
        f"{Colors.BOLD}TEST {index}/{total}: {case.id} ({case.category}){Colors.RESET}"
    )
    print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(f"Question: {case.question}\n")
    print(
        _render_three_column_console_table(
            case.ground_truth,
            case_result.rag_output.response,
            case_result.baseline_output.response,
        )
    )
    print()

    for evaluator in EVALUATOR_ORDER:
        result = case_result.evaluations[evaluator]
        name = evaluator.upper() if evaluator != "llm_judge" else "LLM_JUDGE"
        print(f"{Colors.BOLD}{name}:{Colors.RESET}")
        print(f"  status: {result.status}")
        print(f"  RAG:      label={result.rag_label}, score={result.rag_score}")
        print(
            f"  BASELINE: label={result.baseline_label}, score={result.baseline_score}"
        )
        print(f"  winner:   {result.winner}")
        if result.notes:
            print(f"  notes:    {result.notes}")
        print()


def _init_summary() -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for evaluator in EVALUATOR_ORDER:
        summary[evaluator] = {
            "rag_wins": 0,
            "baseline_wins": 0,
            "ties": 0,
            "rag_factual": 0,
            "rag_hallucinated": 0,
            "baseline_factual": 0,
            "baseline_hallucinated": 0,
            "skipped": 0,
            "errors": 0,
        }
    return summary


def _compute_evaluator_summary(cases: List[CaseResult]) -> Dict[str, Dict[str, int]]:
    summary = _init_summary()

    for case_result in cases:
        for evaluator in EVALUATOR_ORDER:
            result = case_result.evaluations[evaluator]
            row = summary[evaluator]

            if result.status == "completed":
                if result.rag_label == "factual":
                    row["rag_factual"] += 1
                elif result.rag_label == "hallucinated":
                    row["rag_hallucinated"] += 1

                if result.baseline_label == "factual":
                    row["baseline_factual"] += 1
                elif result.baseline_label == "hallucinated":
                    row["baseline_hallucinated"] += 1

                if result.winner == "RAG":
                    row["rag_wins"] += 1
                elif result.winner == "BASELINE":
                    row["baseline_wins"] += 1
                elif result.winner == "Tie":
                    row["ties"] += 1
            elif result.status == "skipped":
                row["skipped"] += 1
            else:
                row["errors"] += 1

    return summary


def run_comparison_suite(
    test_cases: Optional[List[TestCase]] = None,
    threshold: float = 0.5,
    temperature: float = 0.0,
    trace_mode: Optional[str] = None,
    verbose: bool = True,
) -> SuiteResult:
    """Run benchmark suite in legacy-simple mode."""
    resolved_trace_mode = normalize_trace_mode(trace_mode or LANGSMITH_TRACE_MODE)
    cases_to_run = test_cases or GROUND_TRUTH_TEST_CASES

    if verbose:
        print("Loading benchmark runtime...")
        print(f"LangSmith trace mode: {resolved_trace_mode}")

    equivalence_evaluator = EquivalenceEvaluator(
        factual_judge_callable=factual_judge_adapter,
        verbose=False,
    )
    rag_agent = build_agent(temperature=temperature)
    baseline_agent = build_prompt_only_agent(temperature=temperature)

    ragtruth_evaluator: Optional[RAGTruthEvaluator] = None
    try:
        ragtruth_evaluator = RAGTruthEvaluator(model_name=RAGTRUTH_MODEL, strict_mode=False)
    except Exception:
        ragtruth_evaluator = None

    aimon_evaluator: Optional[AimonEvaluator] = None
    try:
        aimon_evaluator = AimonEvaluator(threshold=threshold)
        aimon_evaluator.load_model()
    except Exception:
        aimon_evaluator = None

    case_results: List[CaseResult] = []

    for index, test_case in enumerate(cases_to_run, 1):
        reference_ground_truth = build_reference_ground_truth(
            test_case,
            include_aliases=True,
        )

        rag_error = ""
        baseline_error = ""

        try:
            with langsmith_tracing(
                should_trace_component(resolved_trace_mode, "rag")
            ):
                rag_run = run_agent_with_capture(
                    question=test_case.question,
                    agent=rag_agent,
                    verbose=False,
                )
            rag_output = _to_model_output(
                response=rag_run.final_answer,
                retrieved_context=rag_run.retrieved_context,
                tool_calls=rag_run.tool_calls,
            )
        except Exception as exc:
            rag_error = str(exc)
            rag_output = _to_model_output(response=f"Error: {exc}")

        try:
            with langsmith_tracing(
                should_trace_component(resolved_trace_mode, "baseline")
            ):
                baseline_response = answer_question_prompt_only(
                    test_case.question,
                    llm=baseline_agent,
                    verbose=False,
                )
            baseline_output = _to_model_output(response=baseline_response)
        except Exception as exc:
            baseline_error = str(exc)
            baseline_output = _to_model_output(response=f"Error: {exc}")

        if rag_error or baseline_error:
            reason = "Model execution failed"
            details = "; ".join(part for part in [rag_error, baseline_error] if part)
            evals = {
                evaluator: _error_result(evaluator, f"{reason}: {details}")
                for evaluator in EVALUATOR_ORDER
            }
        else:
            with langsmith_tracing(
                should_trace_component(resolved_trace_mode, "evaluator")
            ):
                evals = {
                    "vectara": _evaluate_vectara(
                        rag_output=rag_output,
                        baseline_output=baseline_output,
                        reference_ground_truth=reference_ground_truth,
                        threshold=threshold,
                        hallucination_model=None,
                        equivalence_evaluator=equivalence_evaluator,
                    ),
                    "aimon": _evaluate_aimon(
                        rag_output=rag_output,
                        baseline_output=baseline_output,
                        test_case=test_case,
                        reference_ground_truth=reference_ground_truth,
                        aimon_evaluator=aimon_evaluator,
                    ),
                    "llm_judge": _evaluate_llm_judge(
                        rag_output=rag_output,
                        baseline_output=baseline_output,
                        test_case=test_case,
                        reference_ground_truth=reference_ground_truth,
                    ),
                    "ragtruth": _evaluate_ragtruth(
                        rag_output=rag_output,
                        baseline_output=baseline_output,
                        test_case=test_case,
                        reference_ground_truth=reference_ground_truth,
                        ragtruth_evaluator=ragtruth_evaluator,
                    ),
                }
            for evaluator_name, evaluator_result in list(evals.items()):
                evals[evaluator_name] = _apply_refusal_policy(
                    result=evaluator_result,
                    test_case=test_case,
                    rag_response=rag_output.response,
                    baseline_response=baseline_output.response,
                )

        case_result = CaseResult(
            test_case=test_case,
            rag_output=rag_output,
            baseline_output=baseline_output,
            evaluations=evals,
        )
        case_results.append(case_result)

        if verbose:
            _print_case_console(case_result, index=index, total=len(cases_to_run))

    summary = _compute_evaluator_summary(case_results)

    if verbose:
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}HEAD-TO-HEAD SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        for evaluator in EVALUATOR_ORDER:
            row = summary[evaluator]
            print(
                f"{evaluator}: RAG={row['rag_wins']} BASELINE={row['baseline_wins']} "
                f"Tie={row['ties']} skipped={row['skipped']} errors={row['errors']}"
            )

    return SuiteResult(
        analysis_version=ANALYSIS_VERSION,
        threshold=threshold,
        temperature=temperature,
        cases=case_results,
        evaluator_summary=summary,
    )
