"""Reporting utilities for the legacy-simple benchmark."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from .models import CaseResult, Colors, EvaluatorResult, SuiteResult

EVALUATOR_ORDER = ["vectara", "aimon", "llm_judge", "ragtruth"]


def _score_text(value):
    if value is None:
        return "N/A"
    return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)


def _escape_cell(text: str) -> str:
    if text is None:
        return ""
    return text.replace("|", "\\|").replace("\n", "<br>")


def _to_summary_block(summary: Dict[str, int]) -> Dict[str, object]:
    return {
        "head_to_head": {
            "rag_wins": summary.get("rag_wins", 0),
            "baseline_wins": summary.get("baseline_wins", 0),
            "ties": summary.get("ties", 0),
        },
        "factual_vs_hallucination": {
            "rag_factual": summary.get("rag_factual", 0),
            "rag_hallucinated": summary.get("rag_hallucinated", 0),
            "baseline_factual": summary.get("baseline_factual", 0),
            "baseline_hallucinated": summary.get("baseline_hallucinated", 0),
        },
        "skipped": summary.get("skipped", 0),
        "errors": summary.get("errors", 0),
    }


def generate_comparison_table(suite: SuiteResult, use_emoji: bool = True) -> str:
    """Console-friendly head-to-head summary per evaluator."""
    lines: List[str] = []
    lines.append(f"{Colors.BOLD}{'=' * 72}{Colors.RESET}")
    lines.append(f"{Colors.BOLD}HEAD-TO-HEAD RESULTS{Colors.RESET}")
    lines.append(f"{Colors.BOLD}{'=' * 72}{Colors.RESET}")

    for evaluator in EVALUATOR_ORDER:
        row = suite.evaluator_summary.get(evaluator, {})
        rag = row.get("rag_wins", 0)
        baseline = row.get("baseline_wins", 0)
        ties = row.get("ties", 0)
        skipped = row.get("skipped", 0)
        errors = row.get("errors", 0)
        lines.append(
            f"{evaluator:<10} RAG={rag:<3} BASELINE={baseline:<3} Tie={ties:<3} "
            f"skipped={skipped:<3} errors={errors:<3}"
        )

    return "\n".join(lines)


def generate_summary_stats(suite: SuiteResult) -> str:
    """Generate compact summary text for console/markdown."""
    total_cases = len(suite.cases)
    return (
        f"Analysis Version: {suite.analysis_version}\n"
        f"Threshold: {suite.threshold}\n"
        f"Temperature: {suite.temperature}\n"
        f"Total Cases: {total_cases}"
    )


def _markdown_head_to_head(suite: SuiteResult) -> str:
    lines = [
        "## Head-to-Head by Evaluator",
        "",
        "| Evaluator | RAG Wins | BASELINE Wins | Ties | Skipped | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for evaluator in EVALUATOR_ORDER:
        row = suite.evaluator_summary.get(evaluator, {})
        lines.append(
            f"| {evaluator} | {row.get('rag_wins', 0)} | {row.get('baseline_wins', 0)} | "
            f"{row.get('ties', 0)} | {row.get('skipped', 0)} | {row.get('errors', 0)} |"
        )

    return "\n".join(lines)


def _markdown_eval_table(suite: SuiteResult, evaluator_name: str) -> str:
    lines = [
        f"## {evaluator_name} Results",
        "",
        "| # | Case ID | Question | RAG | BASELINE | RAG Score | BASELINE Score | Winner | Status |",
        "|---:|---|---|---|---|---:|---:|---|---|",
    ]

    for idx, case in enumerate(suite.cases, 1):
        eval_result = case.evaluations.get(evaluator_name)
        if eval_result is None:
            eval_result = EvaluatorResult(
                name=evaluator_name,
                status="skipped",
                rag_label="skipped",
                baseline_label="skipped",
                winner="N/A",
                notes="Evaluator not run.",
            )

        lines.append(
            f"| {idx} | {case.test_case.id} | {_escape_cell(case.test_case.question)} | "
            f"{eval_result.rag_label} | {eval_result.baseline_label} | "
            f"{_score_text(eval_result.rag_score)} | {_score_text(eval_result.baseline_score)} | "
            f"{eval_result.winner} | {eval_result.status} |"
        )

    return "\n".join(lines)


def _markdown_diagnostics(suite: SuiteResult) -> str:
    lines = ["## Skipped/Error Diagnostics", ""]
    found = False

    for case in suite.cases:
        for evaluator in EVALUATOR_ORDER:
            result = case.evaluations.get(evaluator)
            if result is None:
                continue
            if result.status == "completed":
                continue
            found = True
            lines.append(
                f"- `{case.test_case.id}` {evaluator}: {result.status} ({result.notes or 'No details'})"
            )

    if not found:
        lines.append("- None")

    return "\n".join(lines)


def generate_markdown_table(suite: SuiteResult) -> str:
    """Backward-compatible alias: returns the core markdown sections."""
    sections = [_markdown_head_to_head(suite)]
    for evaluator in EVALUATOR_ORDER:
        sections.append("")
        sections.append(_markdown_eval_table(suite, evaluator))
    sections.append("")
    sections.append(_markdown_diagnostics(suite))
    return "\n".join(sections)


def generate_full_report(suite: SuiteResult) -> str:
    """Build full markdown report."""
    lines = [
        "# RAG vs BASELINE Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Analysis Version: `{suite.analysis_version}`",
        "",
        "## Run Configuration",
        "",
        f"- Threshold: `{suite.threshold}`",
        f"- Temperature: `{suite.temperature}`",
        f"- Total Cases: `{len(suite.cases)}`",
        "",
        generate_markdown_table(suite),
    ]
    return "\n".join(lines)


def _serialize_evaluation(result: EvaluatorResult) -> Dict[str, object]:
    return {
        "status": result.status,
        "rag_label": result.rag_label,
        "baseline_label": result.baseline_label,
        "rag_score": result.rag_score,
        "baseline_score": result.baseline_score,
        "winner": result.winner,
        "notes": result.notes,
    }


def _serialize_case(case: CaseResult) -> Dict[str, object]:
    return {
        "id": case.test_case.id,
        "question": case.test_case.question,
        "ground_truth": case.test_case.ground_truth,
        "category": case.test_case.category,
        "refusal_expected": case.test_case.refusal_expected,
        "rag": {
            "response": case.rag_output.response,
            "retrieved_context": case.rag_output.retrieved_context,
            "tool_calls": case.rag_output.tool_calls,
        },
        "baseline": {
            "response": case.baseline_output.response,
        },
        "evaluations": {
            evaluator: _serialize_evaluation(case.evaluations[evaluator])
            for evaluator in EVALUATOR_ORDER
            if evaluator in case.evaluations
        },
    }


def generate_json_payload(suite: SuiteResult) -> Dict[str, object]:
    """Generate top-level JSON payload."""
    return {
        "analysis_version": suite.analysis_version,
        "config": {
            "threshold": suite.threshold,
            "temperature": suite.temperature,
            "total_cases": len(suite.cases),
        },
        "evaluator_summary": {
            evaluator: _to_summary_block(summary)
            for evaluator, summary in suite.evaluator_summary.items()
        },
        "cases": [_serialize_case(case) for case in suite.cases],
    }


def save_benchmark_report(
    suite: SuiteResult,
    json_path: str = "benchmark_results.json",
    md_path: str = "benchmark_report.md",
) -> None:
    """Save benchmark results to JSON and Markdown files."""
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(generate_json_payload(suite), json_file, indent=2)

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(generate_full_report(suite))
