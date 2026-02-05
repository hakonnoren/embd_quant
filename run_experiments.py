#!/usr/bin/env python3
"""Main entry point for running quantization experiments."""
import argparse
from pathlib import Path

from config import MODELS, DATASETS, QUANTIZATION_SCHEMES, CACHE_DIR, RESULTS_DIR
from experiment_runner import ExperimentRunner
from results_reporter import ResultsReporter


def main():
    parser = argparse.ArgumentParser(
        description="Run embedding quantization experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_experiments.py

  # Quick test on smallest dataset
  python run_experiments.py --datasets NFCorpus --no-matryoshka

  # Specific model and quantization
  python run_experiments.py --models mxbai-embed-large-v1 --quantizations binary binary_rescore

  # Run on single dataset with all options
  python run_experiments.py --datasets SciFact --models all-MiniLM-L6-v2
""",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets to evaluate on (default: all)",
    )
    parser.add_argument(
        "--quantizations",
        nargs="+",
        choices=QUANTIZATION_SCHEMES,
        default=QUANTIZATION_SCHEMES,
        help="Quantization schemes to test (default: all)",
    )
    parser.add_argument(
        "--no-matryoshka",
        action="store_true",
        help="Skip Matryoshka dimension combinations (only test full dimension)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Output directory for results (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help=f"Cache directory for embeddings (default: {CACHE_DIR})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Embedding Quantization Test Lab")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Quantizations: {args.quantizations}")
    print(f"Matryoshka combos: {not args.no_matryoshka}")
    print(f"Output dir: {args.output_dir}")
    print(f"Cache dir: {args.cache_dir}")
    print()

    # Run experiments
    runner = ExperimentRunner(args.cache_dir)
    results = runner.run_all_experiments(
        models=args.models,
        datasets=args.datasets,
        quantizations=args.quantizations,
        include_matryoshka_combos=not args.no_matryoshka,
    )

    if not results:
        print("\nNo experiments were run.")
        return

    # Report results
    reporter = ResultsReporter(results)
    reporter.print_summary()
    reporter.save_results(args.output_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
