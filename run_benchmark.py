from kb_project.benchmark import (
    generate_comparison_table,
    generate_summary_stats,
    run_comparison_suite,
    save_benchmark_report,
    Colors,
)
from kb_project.benchmark.vectra import GROUND_TRUTH_TEST_CASES
import argparse


def main():
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}RAG vs Prompt-Only: Hallucination Comparison{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compare RAG vs Prompt-Only agents")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable LLM-as-a-Judge evaluation using OpenAI (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--ragtruth",
        action="store_true",
        default=True,
        help="Enable RAGTruth span-level hallucination evaluation (default: enabled)",
    )
    parser.add_argument(
        "--no-ragtruth",
        action="store_true",
        help="Disable RAGTruth evaluation",
    )
    parser.add_argument(
        "--aimon",
        action="store_true",
        default=True,
        help="Enable AIMon HDM-2 evaluation (default: enabled)",
    )
    parser.add_argument(
        "--no-aimon",
        action="store_true",
        help="Disable AIMon HDM-2 evaluation",
    )
    parser.add_argument(
        "--use-ragtruth-data",
        action="store_true",
        help="Use RAGTruth QA dataset instead of the built-in 8 questions",
    )
    parser.add_argument(
        "--ragtruth-limit",
        type=int,
        default=50,
        help="Number of RAGTruth QA cases to load (default: 50)",
    )
    parser.add_argument(
        "--ragtruth-split",
        choices=["train", "test"],
        default="test",
        help="RAGTruth split to use when --use-ragtruth-data is set",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Hallucination score threshold (default: 0.5)",
    )
    parser.add_argument(
        "--eval-context-mode",
        choices=["ground_truth", "combined"],
        default="ground_truth",
        help=(
            "Primary evaluation context mode: "
            "'ground_truth' (default, scientifically cleaner) or "
            "'combined' (legacy ground truth + retrieved context for RAG)."
        ),
    )
    parser.add_argument(
        "--benchmark-temperature",
        type=float,
        default=0.0,
        help="Decoding temperature used for both compared models in benchmarks (default: 0.0)",
    )
    args = parser.parse_args()

    # Handle ragtruth flag
    use_ragtruth = args.ragtruth and not args.no_ragtruth
    use_aimon = args.aimon and not args.no_aimon

    test_cases = GROUND_TRUTH_TEST_CASES

    # Optionally load RAGTruth QA cases
    if args.use_ragtruth_data:
        try:
            from kb_project.benchmark.ragtruth_dataset import load_ragtruth_qa_cases

            test_cases = load_ragtruth_qa_cases(
                split=args.ragtruth_split, limit=args.ragtruth_limit
            )
            if not test_cases:
                print(
                    "⚠️  No RAGTruth QA cases were loaded; falling back to built-in cases."
                )
                test_cases = GROUND_TRUTH_TEST_CASES
        except Exception as exc:
            print(f"⚠️  Could not load RAGTruth dataset ({exc}); using built-in cases.")
            test_cases = GROUND_TRUTH_TEST_CASES

    results = run_comparison_suite(
        test_cases=test_cases,
        threshold=args.threshold,
        eval_context_mode=args.eval_context_mode,
        benchmark_temperature=args.benchmark_temperature,
        verbose=True,
        use_llm_judge=args.llm_judge,
        use_ragtruth=use_ragtruth,
        use_aimon=use_aimon,
    )

    # Print summary
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}COMPARISON COMPLETE{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(generate_comparison_table(results, use_emoji=False))
    print(generate_summary_stats(results))

    # Save reports
    save_benchmark_report(results)
    print(
        f"\n{Colors.GREEN}Results saved to benchmark_results.json and benchmark_report.md{Colors.RESET}"
    )


if __name__ == "__main__":
    main()
