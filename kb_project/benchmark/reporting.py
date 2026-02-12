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


def _resolve_factual_winner(result: object) -> str:
    """
    Resolve factual winner robustly for both dataclass results and light test doubles.

    Preference:
    1) explicit factual_winner attribute/property if available
    2) derive from consensus factual labels
    3) fallback to legacy winner
    """
    explicit = getattr(result, "factual_winner", None)
    if isinstance(explicit, str) and explicit:
        return explicit

    rag_factual = getattr(result, "rag_factual_consensus", None)
    prompt_factual = getattr(result, "prompt_factual_consensus", None)
    if rag_factual is True and prompt_factual is False:
        return "RAG"
    if rag_factual is False and prompt_factual is True:
        return "Prompt-Only"
    if rag_factual is not None and prompt_factual is not None:
        return "Tie"

    legacy = getattr(result, "winner", None)
    if isinstance(legacy, str) and legacy:
        return legacy
    return "N/A"


def generate_comparison_table(
    results: List[ComparisonResult],
    use_emoji: bool = True,
    legacy_single_winner: bool = False,
) -> str:
    """Generate a block-style summary for console output."""
    if use_emoji:
        ok, fail = "✅", "❌"
    else:
        ok, fail = f"{Colors.GREEN}OK{Colors.RESET}", f"{Colors.RED}FAIL{Colors.RESET}"

    def _fact_label(value: bool | None) -> str:
        if value is None:
            return "N/A"
        return "FACTUAL" if value else "FACTUAL-ERROR"

    lines = []
    lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    lines.append(f"{Colors.BOLD}BENCHMARK RESULTS SUMMARY (DUAL-TRACK){Colors.RESET}")
    lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    if results:
        lines.append(f"Analysis version: {getattr(results[0], 'analysis_version', 'v2_dual_track')}")
        lines.append(f"Primary factual mode: {getattr(results[0], 'factual_mode', 'ground_truth')}")
        lines.append(f"Diagnostic mode: {getattr(results[0], 'diagnostic_mode', 'combined')}")

    for i, r in enumerate(results, 1):
        lines.append("")
        lines.append(f"{Colors.BOLD}Test {i}: {r.description}{Colors.RESET}")
        lines.append("-" * 40)
        lines.append(
            f"  Factual:  RAG={_fact_label(getattr(r, 'rag_factual_consensus', None))}  "
            f"Prompt={_fact_label(getattr(r, 'prompt_factual_consensus', None))}  "
            f"→ {_resolve_factual_winner(r)}"
        )
        lines.append(
            f"  Complete: RAG={getattr(r, 'rag_completeness', 'insufficient')}  "
            f"Prompt={getattr(r, 'prompt_completeness', 'insufficient')}"
        )
        lines.append(
            f"  Grounding: RAG={getattr(r, 'rag_grounding_status', 'unavailable')}"
            + (
                f" ({r.rag_grounding_score:.3f})"
                if getattr(r, "rag_grounding_score", None) is not None
                else ""
            )
        )
        if (
            getattr(r, "rag_factual_disagreement_rate", None) is not None
            and getattr(r, "prompt_factual_disagreement_rate", None) is not None
        ):
            lines.append(
                f"  Disagree: RAG={r.rag_factual_disagreement_rate:.2f}  "
                f"Prompt={r.prompt_factual_disagreement_rate:.2f}"
            )

        if legacy_single_winner:
            rag_v = fail if r.rag_is_hallucination else ok
            prompt_v = fail if r.prompt_only_is_hallucination else ok
            lines.append(
                f"  Legacy Vectara: RAG={r.rag_score:.3f}{rag_v}  Prompt={r.prompt_only_score:.3f}{prompt_v}  → {r.winner}"
            )

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def generate_markdown_table(
    results: List[ComparisonResult],
    legacy_single_winner: bool = False,
) -> str:
    """Generate markdown tables for the report file."""
    output = ""

    # ==========================================================================
    # Dual-Track Primary Tables
    # ==========================================================================
    output += "### Factual Track (Primary)\n\n"
    output += "| # | Question | RAG Factual | Prompt Factual | Winner |\n"
    output += "|---|----------|-------------|----------------|--------|\n"
    for i, r in enumerate(results, 1):
        q_short = r.question[:40] + "..." if len(r.question) > 40 else r.question
        rag_fact = (
            "✅"
            if getattr(r, "rag_factual_consensus", None) is True
            else "❌" if getattr(r, "rag_factual_consensus", None) is False else "N/A"
        )
        prompt_fact = (
            "✅"
            if getattr(r, "prompt_factual_consensus", None) is True
            else "❌"
            if getattr(r, "prompt_factual_consensus", None) is False
            else "N/A"
        )
        output += (
            f"| {i} | {q_short} | {rag_fact} | {prompt_fact} | {_resolve_factual_winner(r)} |\n"
        )

    output += "\n### Completeness Track (Primary)\n\n"
    output += "| # | Question | RAG Completeness | Prompt Completeness |\n"
    output += "|---|----------|------------------|---------------------|\n"
    for i, r in enumerate(results, 1):
        q_short = r.question[:40] + "..." if len(r.question) > 40 else r.question
        output += (
            f"| {i} | {q_short} | {getattr(r, 'rag_completeness', 'insufficient')} | "
            f"{getattr(r, 'prompt_completeness', 'insufficient')} |\n"
        )

    output += "\n### Grounding Track (RAG)\n\n"
    output += "| # | Question | Grounding Status | Grounding Score |\n"
    output += "|---|----------|------------------|-----------------|\n"
    for i, r in enumerate(results, 1):
        q_short = r.question[:40] + "..." if len(r.question) > 40 else r.question
        gscore = (
            f"{r.rag_grounding_score:.3f}"
            if getattr(r, "rag_grounding_score", None) is not None
            else "N/A"
        )
        output += (
            f"| {i} | {q_short} | {getattr(r, 'rag_grounding_status', 'unavailable')} | {gscore} |\n"
        )

    output += "\n### Evaluator Disagreement (Factual)\n\n"
    output += "| # | Question | RAG Disagreement | Prompt Disagreement |\n"
    output += "|---|----------|------------------|---------------------|\n"
    for i, r in enumerate(results, 1):
        q_short = r.question[:40] + "..." if len(r.question) > 40 else r.question
        rag_d = (
            f"{r.rag_factual_disagreement_rate:.2f}"
            if getattr(r, "rag_factual_disagreement_rate", None) is not None
            else "N/A"
        )
        prompt_d = (
            f"{r.prompt_factual_disagreement_rate:.2f}"
            if getattr(r, "prompt_factual_disagreement_rate", None) is not None
            else "N/A"
        )
        output += f"| {i} | {q_short} | {rag_d} | {prompt_d} |\n"

    # ==========================================================================
    # Legacy Vectara Table (optional)
    # ==========================================================================
    if legacy_single_winner:
        output += "\n### Legacy Vectara Hallucination Model Results\n\n"
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


def generate_summary_stats(
    results: List[ComparisonResult],
    legacy_single_winner: bool = False,
) -> str:
    """Generate summary statistics including primary dual-track and diagnostics."""
    total = len(results)
    evaluation_mode = results[0].evaluation_mode if results else "ground_truth"
    analysis_version = results[0].analysis_version if results else "v2_dual_track"
    factual_mode = results[0].factual_mode if results else "ground_truth"
    diagnostic_mode = results[0].diagnostic_mode if results else "combined"

    if total == 0:
        return (
            f"**Analysis Version:** `{analysis_version}`\n\n"
            "**No results available.**"
        )

    # Dual-track factual consensus stats
    rag_factual = sum(1 for r in results if r.rag_factual_consensus is True)
    rag_factual_error = sum(1 for r in results if r.rag_factual_consensus is False)
    prompt_factual = sum(1 for r in results if r.prompt_factual_consensus is True)
    prompt_factual_error = sum(
        1 for r in results if r.prompt_factual_consensus is False
    )
    factual_rag_wins = sum(1 for r in results if _resolve_factual_winner(r) == "RAG")
    factual_prompt_wins = sum(
        1 for r in results if _resolve_factual_winner(r) == "Prompt-Only"
    )
    factual_ties = sum(1 for r in results if _resolve_factual_winner(r) == "Tie")

    # Completeness distributions
    rag_complete = sum(1 for r in results if r.rag_completeness == "complete")
    rag_partial = sum(1 for r in results if r.rag_completeness == "partial")
    rag_insufficient = sum(
        1 for r in results if r.rag_completeness == "insufficient"
    )
    prompt_complete = sum(
        1 for r in results if r.prompt_completeness == "complete"
    )
    prompt_partial = sum(1 for r in results if r.prompt_completeness == "partial")
    prompt_insufficient = sum(
        1 for r in results if r.prompt_completeness == "insufficient"
    )

    # Grounding stats
    rag_ground_faithful = sum(
        1 for r in results if getattr(r, "rag_grounding_status", "") == "faithful"
    )
    rag_ground_non_faithful = sum(
        1 for r in results if getattr(r, "rag_grounding_status", "") == "non_faithful"
    )
    rag_ground_unavailable = sum(
        1 for r in results if getattr(r, "rag_grounding_status", "") == "unavailable"
    )
    grounding_scores = [
        r.rag_grounding_score
        for r in results
        if getattr(r, "rag_grounding_score", None) is not None
    ]
    avg_grounding = (
        (sum(grounding_scores) / len(grounding_scores)) if grounding_scores else 0.0
    )

    # Disagreement stats
    rag_disagreement_values = [
        r.rag_factual_disagreement_rate
        for r in results
        if r.rag_factual_disagreement_rate is not None
    ]
    prompt_disagreement_values = [
        r.prompt_factual_disagreement_rate
        for r in results
        if r.prompt_factual_disagreement_rate is not None
    ]
    rag_disagreement_avg = (
        sum(rag_disagreement_values) / len(rag_disagreement_values)
        if rag_disagreement_values
        else 0.0
    )
    prompt_disagreement_avg = (
        sum(prompt_disagreement_values) / len(prompt_disagreement_values)
        if prompt_disagreement_values
        else 0.0
    )

    # RAG stats (Vectara)
    rag_hallucinations = sum(1 for r in results if r.rag_is_hallucination)
    rag_factual_legacy = total - rag_hallucinations
    rag_avg_score = sum(r.rag_score for r in results) / total if total > 0 else 0

    # Prompt-only stats (Vectara)
    prompt_hallucinations = sum(1 for r in results if r.prompt_only_is_hallucination)
    prompt_factual_legacy = total - prompt_hallucinations
    prompt_avg_score = (
        sum(r.prompt_only_score for r in results) / total if total > 0 else 0
    )

    # Vectara Winner stats
    rag_wins = sum(1 for r in results if r.winner == "RAG")
    prompt_wins = sum(1 for r in results if r.winner == "Prompt-Only")
    ties = sum(1 for r in results if r.winner == "Tie")

    output = f"""
**Analysis Version:** `{analysis_version}`  
**Primary Factual Mode:** `{factual_mode}`  
**Diagnostic Mode:** `{diagnostic_mode}`  
**Legacy Evaluation Mode Field:** `{evaluation_mode}`

## Factual Track Summary (Primary)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Total Tests | {total} | {total} |
| Factual Responses | {rag_factual} | {prompt_factual} |
| Factual Errors | {rag_factual_error} | {prompt_factual_error} |
| Factual Error Rate | {rag_factual_error/total*100:.1f}% | {prompt_factual_error/total*100:.1f}% |

## Factual Head-to-Head (Primary)

| Winner | Count |
|--------|-------|
| RAG (Wikidata) | {factual_rag_wins} |
| Prompt-Only | {factual_prompt_wins} |
| Tie | {factual_ties} |

## Completeness Track (Primary)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Complete | {rag_complete} | {prompt_complete} |
| Partial | {rag_partial} | {prompt_partial} |
| Insufficient | {rag_insufficient} | {prompt_insufficient} |

## Grounding Track (RAG)

| Metric | RAG |
|--------|-----|
| Faithful | {rag_ground_faithful} |
| Non-Faithful | {rag_ground_non_faithful} |
| Grounding Unavailable | {rag_ground_unavailable} |
| Average Grounding Score | {avg_grounding:.3f} |

## Evaluator Disagreement (Factual)

| Metric | RAG | Prompt-Only |
|--------|-----|-------------|
| Avg Disagreement Rate | {rag_disagreement_avg:.3f} | {prompt_disagreement_avg:.3f} |
"""

    if legacy_single_winner:
        output += f"""
## Legacy Vectara Summary (Diagnostic)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Total Tests | {total} | {total} |
| Factual Responses | {rag_factual_legacy} | {prompt_factual_legacy} |
| Hallucinations | {rag_hallucinations} | {prompt_hallucinations} |
| Hallucination Rate | {rag_hallucinations/total*100:.1f}% | {prompt_hallucinations/total*100:.1f}% |
| Average Score | {rag_avg_score:.3f} | {prompt_avg_score:.3f} |

## Legacy Head-to-Head (Vectara)

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

        # Count factual errors detected by judge
        judge_rag_halluc = sum(
            1
            for r in judge_results
            if r.llm_judge_result and r.llm_judge_result.rag_has_factual_error
        )
        judge_prompt_halluc = sum(
            1
            for r in judge_results
            if r.llm_judge_result and r.llm_judge_result.prompt_has_factual_error
        )

        output += f"""
## LLM Judge Statistics ({OPENAI_JUDGE_MODEL}) (Diagnostic)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Factual Errors Detected | {judge_rag_halluc} | {judge_prompt_halluc} |
| Factual Error Rate | {judge_rag_halluc/len(judge_results)*100:.1f}% | {judge_prompt_halluc/len(judge_results)*100:.1f}% |

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


def generate_full_report(
    results: List[ComparisonResult],
    legacy_single_winner: bool = False,
) -> str:
    """Generate a complete markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluation_mode = results[0].evaluation_mode if results else "ground_truth"
    analysis_version = results[0].analysis_version if results else "v2_dual_track"
    factual_mode = results[0].factual_mode if results else "ground_truth"
    diagnostic_mode = results[0].diagnostic_mode if results else "combined"

    report = f"""# Hallucination Comparison Report

**Generated:** {timestamp}
**Analysis Version:** `{analysis_version}`
**Primary Factual Mode:** `{factual_mode}`
**Diagnostic Mode:** `{diagnostic_mode}`
**Legacy Evaluation Mode Field:** `{evaluation_mode}`

## Overview

This report compares two approaches to reducing LLM hallucinations:

1. **RAG (Wikidata)**: Retrieves facts from Wikidata before responding
2. **Prompt-Only**: Uses an anti-hallucination system prompt without retrieval

## Results Table

{generate_markdown_table(results, legacy_single_winner=legacy_single_winner)}

{generate_summary_stats(results, legacy_single_winner=legacy_single_winner)}

## Detailed Results

"""

    if legacy_single_winner:
        report += (
            "Note: This report includes both dual-track primary metrics and legacy "
            "single-winner diagnostic tables for backward compatibility.\n\n"
        )

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
    legacy_single_winner: bool = False,
) -> None:
    """Save results to JSON and markdown files."""
    # Save JSON
    json_data = []
    for r in results:
        entry = {
            "question": r.question,
            "description": r.description,
            "ground_truth": r.ground_truth,
            "analysis_version": getattr(r, "analysis_version", "v2_dual_track"),
            "benchmark_axis": getattr(r, "benchmark_axis", "dual_track"),
            "evaluation_mode": r.evaluation_mode,
            "factual_mode": getattr(r, "factual_mode", "ground_truth"),
            "diagnostic_mode": getattr(r, "diagnostic_mode", "combined"),
            "rag": {
                "response": r.rag_response,
                "retrieved_context": r.rag_retrieved_context,
                "score": r.rag_score,
                "is_hallucination": r.rag_is_hallucination,
                "faithfulness_score": r.rag_faithfulness_score,
                "faithfulness_is_hallucination": r.rag_faithfulness_is_hallucination,
                "grounding_status": getattr(r, "rag_grounding_status", "unavailable"),
                "grounding_score": getattr(r, "rag_grounding_score", None),
            },
            "prompt_only": {
                "response": r.prompt_only_response,
                "score": r.prompt_only_score,
                "is_hallucination": r.prompt_only_is_hallucination,
            },
            "vectara_winner": r.winner,
            "factual_track": {
                "winner": _resolve_factual_winner(r),
                "rag": {
                    "vectara_factual": getattr(r, "rag_factual_vectara", None),
                    "llm_factual": getattr(r, "rag_factual_llm", None),
                    "ragtruth_factual": getattr(r, "rag_factual_ragtruth", None),
                    "consensus_factual": getattr(r, "rag_factual_consensus", None),
                },
                "prompt_only": {
                    "vectara_factual": getattr(r, "prompt_factual_vectara", None),
                    "llm_factual": getattr(r, "prompt_factual_llm", None),
                    "ragtruth_factual": getattr(r, "prompt_factual_ragtruth", None),
                    "consensus_factual": getattr(
                        r, "prompt_factual_consensus", None
                    ),
                },
            },
            "completeness_track": {
                "rag": getattr(r, "rag_completeness", "insufficient"),
                "prompt_only": getattr(r, "prompt_completeness", "insufficient"),
            },
            "grounding_track": {
                "rag_status": getattr(r, "rag_grounding_status", "unavailable"),
                "rag_score": getattr(r, "rag_grounding_score", None),
            },
            "disagreement": {
                "rag_factual_disagreement_rate": getattr(
                    r, "rag_factual_disagreement_rate", None
                ),
                "prompt_factual_disagreement_rate": getattr(
                    r, "prompt_factual_disagreement_rate", None
                ),
            },
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
                    "has_factual_error": judge.rag_has_factual_error,
                    "completeness": judge.rag_completeness,
                    "missing_required_info": judge.rag_missing_required_info,
                    "details": judge.rag_hallucination_details,
                    "strengths": judge.rag_strengths,
                },
                "prompt_evaluation": {
                    "has_hallucination": judge.prompt_has_hallucination,
                    "has_factual_error": judge.prompt_has_factual_error,
                    "completeness": judge.prompt_completeness,
                    "missing_required_info": judge.prompt_missing_required_info,
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
    report = generate_full_report(results, legacy_single_winner=legacy_single_winner)
    with open(md_path, "w") as f:
        f.write(report)
