"""
Report Generation for Benchmark Module
=======================================
Contains functions for generating comparison tables and reports.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List

from .models import ComparisonResult, Colors
from .llm_judge import format_judge_result_detailed
from ..settings import OPENAI_JUDGE_MODEL


# ==========================================================================
# Table Generation
# ==========================================================================


def _escape_markdown_cell(text: str) -> str:
    """Escape markdown table cell content and preserve line breaks."""
    if text is None:
        return ""
    return text.replace("|", "\\|").replace("\n", "<br>")


def generate_comparison_table(
    results: List[ComparisonResult], use_emoji: bool = True
) -> str:
    """Generate a block-style summary for console output."""
    # Use ASCII for console, emoji for markdown
    if use_emoji:
        ok, fail = "✅", "❌"
    else:
        ok, fail = f"{Colors.GREEN}OK{Colors.RESET}", f"{Colors.RED}FAIL{Colors.RESET}"

    lines = []
    lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    lines.append(f"{Colors.BOLD}BENCHMARK RESULTS SUMMARY{Colors.RESET}")
    lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    if results:
        lines.append(f"Primary evaluation mode: {results[0].evaluation_mode}")

    for i, r in enumerate(results, 1):
        lines.append("")
        lines.append(f"{Colors.BOLD}Test {i}: {r.description}{Colors.RESET}")
        lines.append("-" * 40)

        # Vectara
        rag_v = fail if r.rag_is_hallucination else ok
        prompt_v = fail if r.prompt_only_is_hallucination else ok
        lines.append(
            f"  Vectara:  RAG={r.rag_score:.3f}{rag_v}  Prompt={r.prompt_only_score:.3f}{prompt_v}  → {r.winner}"
        )
        if r.rag_faithfulness_score is not None:
            rag_f = fail if r.rag_faithfulness_is_hallucination else ok
            lines.append(
                f"  Faithful: RAG={r.rag_faithfulness_score:.3f}{rag_f}  (retrieved-evidence grounding)"
            )

        # RAGTruth
        if r.rag_ragtruth_result is not None:
            rag_rt = r.rag_ragtruth_result
            prompt_rt = r.prompt_only_ragtruth_result
            rag_rt_mark = fail if rag_rt.has_hallucination else ok
            prompt_rt_mark = fail if (prompt_rt and prompt_rt.has_hallucination) else ok
            prompt_score = prompt_rt.hallucination_score if prompt_rt else 0
            prompt_spans = prompt_rt.span_count if prompt_rt else 0
            lines.append(
                f"  RAGTruth: RAG={rag_rt.hallucination_score:.3f}({rag_rt.span_count}sp){rag_rt_mark}  "
                f"Prompt={prompt_score:.3f}({prompt_spans}sp){prompt_rt_mark}  → {r.ragtruth_winner}"
            )

        # AIMon
        if r.rag_aimon_result is not None:
            rag_am = r.rag_aimon_result
            prompt_am = r.prompt_only_aimon_result
            rag_am_mark = fail if rag_am.has_hallucination else ok
            prompt_am_mark = fail if (prompt_am and prompt_am.has_hallucination) else ok
            prompt_sev = prompt_am.hallucination_severity if prompt_am else 0
            prompt_sent = len(prompt_am.hallucinated_sentences) if prompt_am else 0
            lines.append(
                f"  AIMon:    RAG={rag_am.hallucination_severity:.3f}({len(rag_am.hallucinated_sentences)}sent){rag_am_mark}  "
                f"Prompt={prompt_sev:.3f}({prompt_sent}sent){prompt_am_mark}  → {r.aimon_winner}"
            )

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def generate_markdown_table(results: List[ComparisonResult]) -> str:
    """Generate markdown tables for the report file - one per evaluator."""
    output = ""

    # ==========================================================================
    # Vectara Table
    # ==========================================================================
    output += "### Vectara Hallucination Model Results\n\n"
    output += "| # | Question | RAG Score | RAG | Prompt Score | Prompt | Winner |\n"
    output += "|---|----------|-----------|-----|--------------|--------|--------|\n"

    for i, r in enumerate(results, 1):
        q_short = r.question[:40] + "..." if len(r.question) > 40 else r.question
        rag_result = "❌" if r.rag_is_hallucination else "✅"
        prompt_result = "❌" if r.prompt_only_is_hallucination else "✅"
        output += f"| {i} | {q_short} | {r.rag_score:.3f} | {rag_result} | {r.prompt_only_score:.3f} | {prompt_result} | {r.winner} |\n"

    faithfulness_results = [r for r in results if r.rag_faithfulness_score is not None]
    if faithfulness_results:
        output += "\n### RAG Retrieval-Faithfulness (Vectara)\n\n"
        output += "| # | Question | RAG Faithfulness Score | RAG |\n"
        output += "|---|----------|------------------------|-----|\n"
        for i, r in enumerate(results, 1):
            if r.rag_faithfulness_score is None:
                continue
            q_short = r.question[:40] + "..." if len(r.question) > 40 else r.question
            rag_result = "❌" if r.rag_faithfulness_is_hallucination else "✅"
            output += (
                f"| {i} | {q_short} | {r.rag_faithfulness_score:.3f} | {rag_result} |\n"
            )

    # ==========================================================================
    # RAGTruth Table (if available)
    # ==========================================================================
    ragtruth_results = [r for r in results if r.rag_ragtruth_result is not None]
    if ragtruth_results:
        output += "\n### RAGTruth Span-Level Detection Results\n\n"
        output += "| # | Question | RAG Score | RAG Spans | RAG | Prompt Score | Prompt Spans | Prompt | Winner |\n"
        output += "|---|----------|-----------|-----------|-----|--------------|--------------|--------|--------|\n"

        for i, r in enumerate(results, 1):
            if r.rag_ragtruth_result is None:
                continue
            q_short = r.question[:35] + "..." if len(r.question) > 35 else r.question
            rag_rt = r.rag_ragtruth_result
            prompt_rt = r.prompt_only_ragtruth_result

            rag_result = "❌" if rag_rt.has_hallucination else "✅"
            prompt_result = (
                "❌" if (prompt_rt and prompt_rt.has_hallucination) else "✅"
            )
            prompt_score = prompt_rt.hallucination_score if prompt_rt else 0
            prompt_spans = prompt_rt.span_count if prompt_rt else 0

            output += (
                f"| {i} | {q_short} | {rag_rt.hallucination_score:.3f} | {rag_rt.span_count} | {rag_result} | "
                f"{prompt_score:.3f} | {prompt_spans} | {prompt_result} | {r.ragtruth_winner} |\n"
            )

    # ==========================================================================
    # AIMon Table (if available)
    # ==========================================================================
    aimon_results = [r for r in results if r.rag_aimon_result is not None]
    if aimon_results:
        output += "\n### AIMon HDM-2 Sentence-Level Detection Results\n\n"
        output += "| # | Question | RAG Severity | RAG Sentences | RAG | Prompt Severity | Prompt Sentences | Prompt | Winner |\n"
        output += "|---|----------|--------------|---------------|-----|-----------------|------------------|--------|--------|\n"

        for i, r in enumerate(results, 1):
            if r.rag_aimon_result is None:
                continue
            q_short = r.question[:30] + "..." if len(r.question) > 30 else r.question
            rag_am = r.rag_aimon_result
            prompt_am = r.prompt_only_aimon_result

            rag_result = "❌" if rag_am.has_hallucination else "✅"
            prompt_result = (
                "❌" if (prompt_am and prompt_am.has_hallucination) else "✅"
            )
            prompt_severity = prompt_am.hallucination_severity if prompt_am else 0
            prompt_sentences = len(prompt_am.hallucinated_sentences) if prompt_am else 0

            output += (
                f"| {i} | {q_short} | {rag_am.hallucination_severity:.3f} | {len(rag_am.hallucinated_sentences)} | {rag_result} | "
                f"{prompt_severity:.3f} | {prompt_sentences} | {prompt_result} | {r.aimon_winner} |\n"
            )

    return output


def generate_summary_stats(results: List[ComparisonResult]) -> str:
    """Generate summary statistics including all evaluation methods."""
    total = len(results)
    evaluation_mode = results[0].evaluation_mode if results else "ground_truth"

    # RAG stats (Vectara)
    rag_hallucinations = sum(1 for r in results if r.rag_is_hallucination)
    rag_factual = total - rag_hallucinations
    rag_avg_score = sum(r.rag_score for r in results) / total if total > 0 else 0

    # Prompt-only stats (Vectara)
    prompt_hallucinations = sum(1 for r in results if r.prompt_only_is_hallucination)
    prompt_factual = total - prompt_hallucinations
    prompt_avg_score = (
        sum(r.prompt_only_score for r in results) / total if total > 0 else 0
    )

    # Vectara Winner stats
    rag_wins = sum(1 for r in results if r.winner == "RAG")
    prompt_wins = sum(1 for r in results if r.winner == "Prompt-Only")
    ties = sum(1 for r in results if r.winner == "Tie")

    output = f"""
**Primary Evaluation Mode:** `{evaluation_mode}`

## Summary Statistics (Vectara Model)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Total Tests | {total} | {total} |
| Factual Responses | {rag_factual} | {prompt_factual} |
| Hallucinations | {rag_hallucinations} | {prompt_hallucinations} |
| Hallucination Rate | {rag_hallucinations/total*100:.1f}% | {prompt_hallucinations/total*100:.1f}% |
| Average Score | {rag_avg_score:.3f} | {prompt_avg_score:.3f} |

## Head-to-Head (Vectara)

| Winner | Count |
|--------|-------|
| RAG (Wikidata) | {rag_wins} |
| Prompt-Only | {prompt_wins} |
| Tie | {ties} |
"""

    faithfulness_results = [r for r in results if r.rag_faithfulness_score is not None]
    if faithfulness_results:
        rag_faith_hallucinations = sum(
            1 for r in faithfulness_results if r.rag_faithfulness_is_hallucination
        )
        rag_faith_avg_score = (
            sum(r.rag_faithfulness_score for r in faithfulness_results if r.rag_faithfulness_score is not None)
            / len(faithfulness_results)
        )
        output += f"""
## RAG Retrieval-Faithfulness (Secondary)

| Metric | RAG |
|--------|-----|
| Cases with retrieval evidence | {len(faithfulness_results)} |
| Non-faithful responses | {rag_faith_hallucinations} |
| Non-faithful rate | {rag_faith_hallucinations/len(faithfulness_results)*100:.1f}% |
| Average faithfulness score | {rag_faith_avg_score:.3f} |
"""

    # Add LLM Judge stats if available
    judge_results = [r for r in results if r.llm_judge_result is not None]
    if judge_results:
        # Count verdicts
        judge_rag_wins = sum(1 for r in judge_results if r.llm_judge_winner == "RAG")
        judge_prompt_wins = sum(
            1 for r in judge_results if r.llm_judge_winner == "Prompt-Only"
        )
        judge_ties = sum(1 for r in judge_results if r.llm_judge_winner == "Tie")
        judge_both_good = sum(
            1 for r in judge_results if r.llm_judge_winner == "Both-Good"
        )
        judge_both_bad = sum(
            1 for r in judge_results if r.llm_judge_winner == "Both-Bad"
        )
        judge_errors = sum(1 for r in judge_results if r.llm_judge_winner == "Error")

        # Count hallucinations detected by judge
        judge_rag_halluc = sum(
            1
            for r in judge_results
            if r.llm_judge_result and r.llm_judge_result.rag_has_hallucination
        )
        judge_prompt_halluc = sum(
            1
            for r in judge_results
            if r.llm_judge_result and r.llm_judge_result.prompt_has_hallucination
        )

        output += f"""
## LLM Judge Statistics ({OPENAI_JUDGE_MODEL})

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Hallucinations Detected | {judge_rag_halluc} | {judge_prompt_halluc} |
| Hallucination Rate | {judge_rag_halluc/len(judge_results)*100:.1f}% | {judge_prompt_halluc/len(judge_results)*100:.1f}% |

## Head-to-Head (LLM Judge)

| Verdict | Count |
|---------|-------|
| RAG Wins | {judge_rag_wins} |
| Prompt-Only Wins | {judge_prompt_wins} |
| Tie | {judge_ties} |
| Both Good | {judge_both_good} |
| Both Bad | {judge_both_bad} |
| Errors | {judge_errors} |
"""

    # Add RAGTruth stats if available
    ragtruth_results = [r for r in results if r.rag_ragtruth_result is not None]
    if ragtruth_results:
        # Count hallucinations detected by RAGTruth
        rt_rag_halluc = sum(
            1
            for r in ragtruth_results
            if r.rag_ragtruth_result and r.rag_ragtruth_result.has_hallucination
        )
        rt_prompt_halluc = sum(
            1
            for r in ragtruth_results
            if r.prompt_only_ragtruth_result
            and r.prompt_only_ragtruth_result.has_hallucination
        )

        # Calculate average hallucination scores
        rt_rag_avg_score = sum(
            r.rag_ragtruth_result.hallucination_score
            for r in ragtruth_results
            if r.rag_ragtruth_result
        ) / len(ragtruth_results)
        rt_prompt_avg_score = sum(
            r.prompt_only_ragtruth_result.hallucination_score
            for r in ragtruth_results
            if r.prompt_only_ragtruth_result
        ) / len(ragtruth_results)

        # Calculate average span counts
        rt_rag_avg_spans = sum(
            r.rag_ragtruth_result.span_count
            for r in ragtruth_results
            if r.rag_ragtruth_result
        ) / len(ragtruth_results)
        rt_prompt_avg_spans = sum(
            r.prompt_only_ragtruth_result.span_count
            for r in ragtruth_results
            if r.prompt_only_ragtruth_result
        ) / len(ragtruth_results)

        # Head-to-head
        rt_rag_wins = sum(1 for r in ragtruth_results if r.ragtruth_winner == "RAG")
        rt_prompt_wins = sum(
            1 for r in ragtruth_results if r.ragtruth_winner == "Prompt-Only"
        )
        rt_ties = sum(1 for r in ragtruth_results if r.ragtruth_winner == "Tie")

        output += f"""
## RAGTruth Statistics (Span-Level Detection)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Hallucinations Detected | {rt_rag_halluc} | {rt_prompt_halluc} |
| Hallucination Rate | {rt_rag_halluc/len(ragtruth_results)*100:.1f}% | {rt_prompt_halluc/len(ragtruth_results)*100:.1f}% |
| Avg Hallucination Score | {rt_rag_avg_score:.3f} | {rt_prompt_avg_score:.3f} |
| Avg Hallucinated Spans | {rt_rag_avg_spans:.1f} | {rt_prompt_avg_spans:.1f} |

## Head-to-Head (RAGTruth)

| Winner | Count |
|--------|-------|
| RAG (Wikidata) | {rt_rag_wins} |
| Prompt-Only | {rt_prompt_wins} |
| Tie | {rt_ties} |
"""

    # Add AIMon stats if available
    aimon_results = [r for r in results if r.rag_aimon_result is not None]
    if aimon_results:
        # Count hallucinations detected by AIMon
        am_rag_halluc = sum(
            1
            for r in aimon_results
            if r.rag_aimon_result and r.rag_aimon_result.has_hallucination
        )
        am_prompt_halluc = sum(
            1
            for r in aimon_results
            if r.prompt_only_aimon_result
            and r.prompt_only_aimon_result.has_hallucination
        )

        # Calculate average hallucination severity
        am_rag_avg_severity = sum(
            r.rag_aimon_result.hallucination_severity
            for r in aimon_results
            if r.rag_aimon_result
        ) / len(aimon_results)
        am_prompt_avg_severity = sum(
            r.prompt_only_aimon_result.hallucination_severity
            for r in aimon_results
            if r.prompt_only_aimon_result
        ) / len(aimon_results)

        # Calculate average hallucinated sentence counts
        am_rag_avg_sentences = sum(
            len(r.rag_aimon_result.hallucinated_sentences)
            for r in aimon_results
            if r.rag_aimon_result
        ) / len(aimon_results)
        am_prompt_avg_sentences = sum(
            len(r.prompt_only_aimon_result.hallucinated_sentences)
            for r in aimon_results
            if r.prompt_only_aimon_result
        ) / len(aimon_results)

        # Head-to-head
        am_rag_wins = sum(1 for r in aimon_results if r.aimon_winner == "RAG")
        am_prompt_wins = sum(
            1 for r in aimon_results if r.aimon_winner == "Prompt-Only"
        )
        am_ties = sum(1 for r in aimon_results if r.aimon_winner == "Tie")

        output += f"""
## AIMon HDM-2 Statistics (Sentence-Level Detection)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Hallucinations Detected | {am_rag_halluc} | {am_prompt_halluc} |
| Hallucination Rate | {am_rag_halluc/len(aimon_results)*100:.1f}% | {am_prompt_halluc/len(aimon_results)*100:.1f}% |
| Avg Hallucination Severity | {am_rag_avg_severity:.3f} | {am_prompt_avg_severity:.3f} |
| Avg Hallucinated Sentences | {am_rag_avg_sentences:.1f} | {am_prompt_avg_sentences:.1f} |

## Head-to-Head (AIMon)

| Winner | Count |
|--------|-------|
| RAG (Wikidata) | {am_rag_wins} |
| Prompt-Only | {am_prompt_wins} |
| Tie | {am_ties} |
"""

    return output


def generate_full_report(results: List[ComparisonResult]) -> str:
    """Generate a complete markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluation_mode = results[0].evaluation_mode if results else "ground_truth"

    report = f"""# Hallucination Comparison Report

**Generated:** {timestamp}
**Primary Evaluation Mode:** `{evaluation_mode}`

## Overview

This report compares two approaches to reducing LLM hallucinations:

1. **RAG (Wikidata)**: Retrieves facts from Wikidata before responding
2. **Prompt-Only**: Uses an anti-hallucination system prompt without retrieval

## Results Table

{generate_markdown_table(results)}

{generate_summary_stats(results)}

## Detailed Results

"""

    for i, r in enumerate(results, 1):
        rag_status = "❌ HALLUCINATION" if r.rag_is_hallucination else "✅ FACTUAL"
        prompt_status = (
            "❌ HALLUCINATION" if r.prompt_only_is_hallucination else "✅ FACTUAL"
        )

        report += f"""
### Test {i}: {r.description}

**Question:** {r.question}

| Ground Truth | RAG Output | Prompt-Only Output |
|---|---|---|
| {_escape_markdown_cell(r.ground_truth)} | {_escape_markdown_cell(r.rag_response)} | {_escape_markdown_cell(r.prompt_only_response)} |

#### RAG Model ({rag_status}, Score: {r.rag_score:.3f})

#### Prompt-Only Model ({prompt_status}, Score: {r.prompt_only_score:.3f})

"""
        if r.rag_faithfulness_score is not None:
            rag_faith_status = (
                "❌ NON-FAITHFUL"
                if r.rag_faithfulness_is_hallucination
                else "✅ FAITHFUL"
            )
            report += f"""#### RAG Retrieval-Faithfulness (Secondary, Retrieved Evidence Only)

**Status:** {rag_faith_status}  
**Score:** {r.rag_faithfulness_score:.3f}

"""
        # Add LLM Judge evaluation if available
        if r.llm_judge_result is not None:
            report += f"""#### LLM Judge Evaluation

{format_judge_result_detailed(r.llm_judge_result)}

"""

        # Add RAGTruth evaluation if available
        if (
            r.rag_ragtruth_result is not None
            and r.prompt_only_ragtruth_result is not None
        ):
            rag_rt = r.rag_ragtruth_result
            prompt_rt = r.prompt_only_ragtruth_result

            rag_rt_status = "HALLUCINATED" if rag_rt.has_hallucination else "FACTUAL"
            prompt_rt_status = (
                "HALLUCINATED" if prompt_rt.has_hallucination else "FACTUAL"
            )

            report += f"""#### RAGTruth Evaluation

| Model | Status | Score | Spans |
|-------|--------|-------|-------|
| RAG | {rag_rt_status} | {rag_rt.hallucination_score:.3f} | {rag_rt.span_count} |
| Prompt-Only | {prompt_rt_status} | {prompt_rt.hallucination_score:.3f} | {prompt_rt.span_count} |

"""
            # Add hallucinated spans for RAG
            if rag_rt.hallucinated_spans:
                report += "**RAG Hallucinated Spans:**\n"
                for span in rag_rt.hallucinated_spans:
                    report += f'- "{span.text}"\n'
                    if span.reason:
                        report += f"  - Reason: {span.reason}\n"
                report += "\n"

            # Add hallucinated spans for Prompt-Only
            if prompt_rt.hallucinated_spans:
                report += "**Prompt-Only Hallucinated Spans:**\n"
                for span in prompt_rt.hallucinated_spans:
                    report += f'- "{span.text}"\n'
                    if span.reason:
                        report += f"  - Reason: {span.reason}\n"
                report += "\n"

            # Add analysis summaries
            if rag_rt.analysis or prompt_rt.analysis:
                report += "**Analysis:**\n"
                if rag_rt.analysis:
                    report += f"- RAG: {rag_rt.analysis}\n"
                if prompt_rt.analysis:
                    report += f"- Prompt-Only: {prompt_rt.analysis}\n"
                report += "\n"

        # Add AIMon evaluation if available
        if r.rag_aimon_result is not None and r.prompt_only_aimon_result is not None:
            rag_am = r.rag_aimon_result
            prompt_am = r.prompt_only_aimon_result

            rag_am_status = "HALLUCINATED" if rag_am.has_hallucination else "FACTUAL"
            prompt_am_status = (
                "HALLUCINATED" if prompt_am.has_hallucination else "FACTUAL"
            )

            report += f"""#### AIMon HDM-2 Evaluation

| Model | Status | Severity | Sentences |
|-------|--------|----------|-----------|
| RAG | {rag_am_status} | {rag_am.hallucination_severity:.3f} | {len(rag_am.hallucinated_sentences)} |
| Prompt-Only | {prompt_am_status} | {prompt_am.hallucination_severity:.3f} | {len(prompt_am.hallucinated_sentences)} |

"""
            # Add hallucinated sentences for RAG
            if rag_am.hallucinated_sentences:
                report += "**RAG Hallucinated Sentences:**\n"
                for sent in rag_am.hallucinated_sentences:
                    ck_marker = (
                        " [Common Knowledge]" if sent.is_common_knowledge else ""
                    )
                    report += (
                        f'- "{sent.text}" (prob: {sent.probability:.3f}){ck_marker}\n'
                    )
                report += "\n"

            # Add hallucinated sentences for Prompt-Only
            if prompt_am.hallucinated_sentences:
                report += "**Prompt-Only Hallucinated Sentences:**\n"
                for sent in prompt_am.hallucinated_sentences:
                    ck_marker = (
                        " [Common Knowledge]" if sent.is_common_knowledge else ""
                    )
                    report += (
                        f'- "{sent.text}" (prob: {sent.probability:.3f}){ck_marker}\n'
                    )
                report += "\n"

        report += "---\n"

    return report


def save_benchmark_report(
    results: List[ComparisonResult],
    json_path: str = "benchmark_results.json",
    md_path: str = "benchmark_report.md",
) -> None:
    """Save results to JSON and markdown files."""
    # Save JSON
    json_data = []
    for r in results:
        entry = {
            "question": r.question,
            "description": r.description,
            "ground_truth": r.ground_truth,
            "evaluation_mode": r.evaluation_mode,
            "rag": {
                "response": r.rag_response,
                "retrieved_context": r.rag_retrieved_context,
                "score": r.rag_score,
                "is_hallucination": r.rag_is_hallucination,
                "faithfulness_score": r.rag_faithfulness_score,
                "faithfulness_is_hallucination": r.rag_faithfulness_is_hallucination,
            },
            "prompt_only": {
                "response": r.prompt_only_response,
                "score": r.prompt_only_score,
                "is_hallucination": r.prompt_only_is_hallucination,
            },
            "vectara_winner": r.winner,
        }

        # Add LLM Judge results if available
        if r.llm_judge_result is not None:
            judge = r.llm_judge_result
            entry["llm_judge"] = {
                "winner": judge.winner,
                "confidence": judge.confidence,
                "reasoning": judge.reasoning,
                "rag_evaluation": {
                    "has_hallucination": judge.rag_has_hallucination,
                    "details": judge.rag_hallucination_details,
                    "strengths": judge.rag_strengths,
                },
                "prompt_evaluation": {
                    "has_hallucination": judge.prompt_has_hallucination,
                    "details": judge.prompt_hallucination_details,
                    "strengths": judge.prompt_strengths,
                },
                "error": judge.error,
            }

        # Add RAGTruth results if available
        if r.rag_ragtruth_result is not None:
            entry["ragtruth"] = {
                "winner": r.ragtruth_winner,
                "rag_evaluation": r.rag_ragtruth_result.to_dict(),
                "prompt_only_evaluation": (
                    r.prompt_only_ragtruth_result.to_dict()
                    if r.prompt_only_ragtruth_result
                    else None
                ),
            }

        # Add AIMon results if available
        if r.rag_aimon_result is not None:
            entry["aimon"] = {
                "winner": r.aimon_winner,
                "rag_evaluation": r.rag_aimon_result.to_dict(),
                "prompt_only_evaluation": (
                    r.prompt_only_aimon_result.to_dict()
                    if r.prompt_only_aimon_result
                    else None
                ),
            }

        json_data.append(entry)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Save markdown report
    report = generate_full_report(results)
    with open(md_path, "w") as f:
        f.write(report)
