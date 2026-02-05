#!/usr/bin/env python3
"""Main entry point for running quantization experiments."""
import argparse
from pathlib import Path

from config import MODELS, DATASETS, QUANTIZATION_SCHEMES, CACHE_DIR, RESULTS_DIR
from experiment_runner import ExperimentRunner
from results_reporter import ResultsReporter


def generate_experiment_id(args) -> str:
    """Generate a descriptive experiment ID from the run configuration."""
    parts = []

    # Models (abbreviated)
    model_abbrev = {
        "mxbai-embed-large-v1": "mxbai",
        "nomic-embed-text-v1.5": "nomic",
        "all-MiniLM-L6-v2": "minilm",
    }
    models = sorted([model_abbrev.get(m, m[:6]) for m in args.models])
    parts.append("+".join(models))

    # Datasets (abbreviated)
    datasets = sorted([d[:4].lower() for d in args.datasets])
    parts.append("+".join(datasets))

    # Flags
    if args.no_matryoshka:
        parts.append("no-mrl")

    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Run embedding quantization experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (auto-generates experiment ID like 20240115_143052_mxbai+nomic+minilm_nfco+scif+fiqa)
  python run_experiments.py

  # Quick test on smallest dataset
  python run_experiments.py --datasets nfcorpus --no-matryoshka

  # Named experiment for easy reference
  python run_experiments.py --experiment-id baseline_v1 --datasets nfcorpus

  # Specific model and quantization
  python run_experiments.py --models mxbai-embed-large-v1 --quantizations binary binary_rescore
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
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID for results subfolder (default: auto-generated from timestamp + config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Base output directory for results (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help=f"Cache directory for embeddings (default: {CACHE_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32, reduce if OOM)",
    )

    args = parser.parse_args()

    # Generate experiment ID if not provided
    if args.experiment_id is None:
        args.experiment_id = generate_experiment_id(args)

    # Create experiment-specific output directory
    experiment_output_dir = args.output_dir / args.experiment_id

    print("=" * 60)
    print("Embedding Quantization Test Lab")
    print("=" * 60)
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Quantizations: {args.quantizations}")
    print(f"Matryoshka combos: {not args.no_matryoshka}")
    print(f"Output dir: {experiment_output_dir}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Run experiments
    runner = ExperimentRunner(args.cache_dir, batch_size=args.batch_size, output_dir=experiment_output_dir)
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
    reporter = ResultsReporter(results, experiment_id=args.experiment_id)
    reporter.print_summary()
    reporter.save_results(experiment_output_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
