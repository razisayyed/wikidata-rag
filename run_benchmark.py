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
    args = parser.parse_args()

    test_cases = GROUND_TRUTH_TEST_CASES
    if args.max_cases is not None and args.max_cases > 0:
        test_cases = test_cases[: args.max_cases]

    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}RAG vs BASELINE Benchmark{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

    suite = run_comparison_suite(
        test_cases=test_cases,
        threshold=args.threshold,
        temperature=args.temperature,
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
