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
        default="combined",
        help=(
            "Diagnostic context mode: "
            "'combined' (default, legacy ground truth + retrieved context) or "
            "'ground_truth'. Primary factual track is configured via --factual-mode."
        ),
    )
    parser.add_argument(
        "--factual-mode",
        choices=["ground_truth", "combined"],
        default="ground_truth",
        help=(
            "Primary factual evaluation mode (default: ground_truth). "
            "In dual-track mode this controls the factual-consistency context."
        ),
    )
    parser.add_argument(
        "--benchmark-axis",
        choices=["dual_track", "legacy"],
        default="dual_track",
        help=(
            "Benchmark evaluation axis. "
            "'dual_track' (default) separates factuality and grounding; "
            "'legacy' keeps older single-axis behavior."
        ),
    )
    parser.add_argument(
        "--legacy-single-winner",
        action="store_true",
        help="Include legacy single-winner tables in reports and console summaries.",
    )
    parser.add_argument(
        "--benchmark-temperature",
        type=float,
        default=0.0,
        help="Decoding temperature used for both compared models in benchmarks (default: 0.0)",
    )
    parser.add_argument(
        "--ground-truth-style",
        choices=["concise", "rich"],
        default="concise",
        help=(
            "Ground-truth construction style: "
            "'concise' uses only canonical answers (fair default), "
            "'rich' adds key-fact bullets."
        ),
    )
    parser.add_argument(
        "--max-ground-truth-facts",
        type=int,
        default=None,
        help=(
            "Optional cap on key facts included when --ground-truth-style=rich "
            "(default: include all available facts)."
        ),
    )
    parser.add_argument(
        "--no-ground-truth-aliases",
        action="store_true",
        help=(
            "Disable accepted-alias expansion in benchmark ground-truth context "
            "(enabled by default for fair factual scoring)."
        ),
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
        factual_mode=args.factual_mode,
        benchmark_axis=args.benchmark_axis,
        ground_truth_style=args.ground_truth_style,
        max_ground_truth_facts=args.max_ground_truth_facts,
        include_ground_truth_aliases=not args.no_ground_truth_aliases,
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
    print(
        generate_comparison_table(
            results,
            use_emoji=False,
            legacy_single_winner=args.legacy_single_winner,
        )
    )
    print(
        generate_summary_stats(
            results,
            legacy_single_winner=args.legacy_single_winner,
        )
    )

    # Save reports
    save_benchmark_report(
        results,
        legacy_single_winner=args.legacy_single_winner,
    )
    print(
        f"\n{Colors.GREEN}Results saved to benchmark_results.json and benchmark_report.md{Colors.RESET}"
    )


if __name__ == "__main__":
    main()
