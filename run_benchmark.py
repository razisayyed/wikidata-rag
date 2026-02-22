from __future__ import annotations

import argparse

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

    suite = run_comparison_suite(
        test_cases=test_cases,
        threshold=args.threshold,
        temperature=args.temperature,
        trace_mode=args.trace_mode,
        enabled_evaluators=enabled_evaluators,
        verbose=True,
    )

    print()
    print(generate_comparison_table(suite, use_emoji=False))
    print()
    print(generate_summary_stats(suite))

    save_benchmark_report(suite)
    print(
        f"\n{Colors.GREEN}Results saved to benchmark_results.json and benchmark_report.md{Colors.RESET}"
    )


if __name__ == "__main__":
    main()
