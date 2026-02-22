from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from kb_project.benchmark import (
    Colors,
    GROUND_TRUTH_TEST_CASES,
    generate_comparison_table,
    generate_summary_stats,
    run_comparison_suite,
    save_benchmark_report,
)
from kb_project.benchmark.evaluator_registry import (
    DEFAULT_ENABLED_EVALUATOR_ORDER,
    EVALUATOR_DISPLAY_NAMES,
)

JSON_REPORT_PATH = "benchmark_results.json"
MD_REPORT_PATH = "benchmark_report.md"


def _normalize_question(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _collect_cases_from_payload(
    payload: Dict[str, Any],
    case_ids: Set[str],
    questions: Set[str],
) -> None:
    for case in payload.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id", "")).strip()
        if case_id:
            case_ids.add(case_id)
        question = _normalize_question(str(case.get("question", "")))
        if question:
            questions.add(question)


def _load_processed_cases_from_json(json_path: str) -> Tuple[Set[str], Set[str]]:
    path = Path(json_path)
    case_ids: Set[str] = set()
    questions: Set[str] = set()
    if not path.exists():
        return case_ids, questions

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return case_ids, questions

    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        for entry in data.get("entries", []):
            if not isinstance(entry, dict):
                continue
            payload = entry.get("payload")
            if isinstance(payload, dict):
                _collect_cases_from_payload(payload, case_ids, questions)
        latest = data.get("latest")
        if isinstance(latest, dict):
            _collect_cases_from_payload(latest, case_ids, questions)
    elif isinstance(data, dict):
        _collect_cases_from_payload(data, case_ids, questions)

    return case_ids, questions


def _load_processed_case_ids_from_markdown(md_path: str) -> Set[str]:
    path = Path(md_path)
    if not path.exists():
        return set()
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return set()

    case_ids: Set[str] = set()
    row_pattern = re.compile(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)
    for match in row_pattern.finditer(text):
        case_id = match.group(1).strip()
        if case_id and case_id.lower() != "case id":
            case_ids.add(case_id)
    return case_ids


def _filter_unprocessed_cases(test_cases, json_path: str, md_path: str):
    processed_ids, processed_questions = _load_processed_cases_from_json(json_path)
    processed_ids.update(_load_processed_case_ids_from_markdown(md_path))

    filtered = []
    skipped = 0
    for case in test_cases:
        case_id = str(getattr(case, "id", "") or "").strip()
        question = _normalize_question(str(getattr(case, "question", "") or ""))
        if (case_id and case_id in processed_ids) or (question and question in processed_questions):
            skipped += 1
            continue
        filtered.append(case)
    return filtered, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG vs BASELINE benchmark")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Hallucination threshold (default: 0.5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature for both compared models (default: 0.0)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on number of benchmark cases.",
    )
    parser.add_argument(
        "--trace-mode",
        type=str,
        choices=["rag", "rag_baseline", "all"],
        default=None,
        help=(
            "LangSmith trace scope: rag (default via env), "
            "rag_baseline, or all."
        ),
    )
    parser.add_argument(
        "--enable-ground-truth-equivalence",
        action="store_true",
        help="Enable Ground-Truth Equivalence evaluator (`vectara` internal id). Disabled by default.",
    )
    parser.add_argument(
        "--enable-ground-truth-grounding",
        action="store_true",
        help="Enable Ground-Truth Grounding (RAGTruth-style) evaluator (`ragtruth`). Disabled by default.",
    )
    args = parser.parse_args()

    test_cases = GROUND_TRUTH_TEST_CASES
    if args.max_cases is not None and args.max_cases > 0:
        test_cases = test_cases[: args.max_cases]
    test_cases, skipped_existing = _filter_unprocessed_cases(
        test_cases,
        json_path=JSON_REPORT_PATH,
        md_path=MD_REPORT_PATH,
    )

    enabled_evaluators = list(DEFAULT_ENABLED_EVALUATOR_ORDER)
    if args.enable_ground_truth_equivalence and "vectara" not in enabled_evaluators:
        enabled_evaluators.append("vectara")
    if args.enable_ground_truth_grounding and "ragtruth" not in enabled_evaluators:
        enabled_evaluators.append("ragtruth")

    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}RAG vs BASELINE Benchmark{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(
        "Default-enabled evaluators: "
        + ", ".join(
            f"{EVALUATOR_DISPLAY_NAMES.get(e, e)} ({e})" for e in enabled_evaluators
        )
    )
    if skipped_existing:
        print(f"Skipping {skipped_existing} case(s) already present in reports.")
    if not test_cases:
        print(
            f"{Colors.GREEN}No new questions to run. Existing reports already contain all selected cases.{Colors.RESET}"
        )
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_cases = len(test_cases)

    def _save_progress_snapshot(partial_suite):
        save_benchmark_report(
            partial_suite,
            json_path=JSON_REPORT_PATH,
            md_path=MD_REPORT_PATH,
            append=True,
            metadata={
                "run_id": run_id,
                "checkpoint_type": "per_question",
                "cases_completed": len(partial_suite.cases),
                "total_cases": total_cases,
                "is_final_snapshot": len(partial_suite.cases) == total_cases,
            },
        )

    suite = run_comparison_suite(
        test_cases=test_cases,
        threshold=args.threshold,
        temperature=args.temperature,
        trace_mode=args.trace_mode,
        enabled_evaluators=enabled_evaluators,
        progress_callback=_save_progress_snapshot,
        verbose=True,
    )

    print()
    print(generate_comparison_table(suite, use_emoji=False))
    print()
    print(generate_summary_stats(suite))

    print(
        f"\n{Colors.GREEN}Results appended to {JSON_REPORT_PATH} and {MD_REPORT_PATH} after each question{Colors.RESET}"
    )


if __name__ == "__main__":
    main()
