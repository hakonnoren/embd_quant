#!/usr/bin/env python3
"""Unified CLI entry point for quantization + reranking experiments.

Usage:
  # Default sweep: binary retrieval with common rescore methods
  python run.py --datasets fiqa

  # Specific methods
  python run.py --datasets fiqa --retrieval binary --rescore none float32 int8 int4

  # Include funnel variants
  python run.py --datasets fiqa --retrieval binary --rescore float32 binary --funnel

  # Float32 baseline only
  python run.py --datasets fiqa --retrieval float32 --rescore none

  # Multi-model, multi-dataset
  python run.py --datasets fiqa nfcorpus --models mxbai-embed-large-v1 nomic-embed-text-v1.5
"""

import argparse
from pathlib import Path

from config import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RETRIEVAL,
    DEFAULT_RESCORE,
    DEFAULT_FUNNEL_FACTORS,
    DATASETS,
    MODELS,
    OVERSAMPLE,
    TRUNCATE_DIMS,
    ExperimentConfig,
    is_valid_combo,
)
from runner import ExperimentRunner


def generate_experiment_id(args) -> str:
    model_abbrev = {
        "mxbai-embed-large-v1": "mxbai",
        "nomic-embed-text-v1.5": "nomic",
        "all-MiniLM-L6-v2": "minilm",
    }
    parts = []
    models = sorted([model_abbrev.get(m, m[:6]) for m in args.models])
    parts.append("+".join(models))
    datasets = sorted([d[:4].lower() for d in args.datasets])
    parts.append("+".join(datasets))
    retrieval = sorted(args.retrieval)
    parts.append("r_" + "+".join(retrieval))
    rescore = sorted(args.rescore)
    parts.append("s_" + "+".join(rescore))
    if args.funnel:
        ffs = sorted(args.funnel_factors)
        parts.append("ff" + "+".join(str(f) for f in ffs))
    return "_".join(parts)


def build_configs(args) -> list:
    """Generate ExperimentConfig list from CLI args."""
    configs = []
    for model in args.models:
        for dataset in args.datasets:
            for dim in TRUNCATE_DIMS:
                for retrieval in args.retrieval:
                    for rescore in args.rescore:
                        if not is_valid_combo(retrieval, rescore):
                            continue
                        # Direct (no funnel)
                        configs.append(ExperimentConfig(
                            model=model,
                            dataset=dataset,
                            truncate_dim=dim,
                            retrieval=retrieval,
                            rescore=rescore,
                            funnel=False,
                            oversample=OVERSAMPLE,
                            funnel_factor=1,
                            cache_dir=args.cache_dir,
                            dataset_cache_dir=args.dataset_cache_dir,
                        ))

                    # Funnel variants (only for binary retrieval + non-none rescore)
                    if args.funnel and retrieval == "binary":
                        for rescore in args.rescore:
                            if not is_valid_combo(retrieval, rescore):
                                continue
                            for ff in args.funnel_factors:
                                configs.append(ExperimentConfig(
                                    model=model,
                                    dataset=dataset,
                                    truncate_dim=dim,
                                    retrieval=retrieval,
                                    rescore=rescore,
                                    funnel=True,
                                    oversample=OVERSAMPLE,
                                    funnel_factor=ff,
                                    cache_dir=args.cache_dir,
                                    dataset_cache_dir=args.dataset_cache_dir,
                                ))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run quantization + reranking experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help=f"Datasets (choices: {list(DATASETS.keys())}, default: {DEFAULT_DATASETS})",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS),
        help=f"Models (choices: {list(MODELS.keys())}, default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--retrieval", nargs="+", default=DEFAULT_RETRIEVAL,
        help=f"Retrieval methods (default: {DEFAULT_RETRIEVAL})",
    )
    parser.add_argument(
        "--rescore", nargs="+", default=DEFAULT_RESCORE,
        help=f"Rescore methods (default: {DEFAULT_RESCORE})",
    )
    parser.add_argument(
        "--funnel", action="store_true",
        help="Also run funnel variants (binary residual prune → rescore)",
    )
    parser.add_argument(
        "--funnel-factors", nargs="+", type=int, default=DEFAULT_FUNNEL_FACTORS,
        help=f"Funnel factors to sweep (default: {DEFAULT_FUNNEL_FACTORS})",
    )
    parser.add_argument(
        "--experiment-id", type=str, default=None,
        help="Experiment ID for results subfolder (default: auto-generated)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/embeddings"),
        help="Cache directory for embeddings",
    )
    parser.add_argument(
        "--dataset-cache-dir", type=Path, default=Path("cache/datasets"),
        help="Cache directory for datasets",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting after experiments finish",
    )

    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = generate_experiment_id(args)

    experiment_output_dir = args.output_dir / args.experiment_id
    configs = build_configs(args)

    print("=" * 60)
    print("Quantization + Reranking Experiments")
    print("=" * 60)
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Retrieval: {args.retrieval}")
    print(f"Rescore: {args.rescore}")
    print(f"Funnel: {args.funnel} (factors: {args.funnel_factors})")
    print(f"Truncate dims: {TRUNCATE_DIMS}")
    print(f"Oversample: {OVERSAMPLE}")
    print(f"Configs: {len(configs)} total")
    print(f"Output dir: {experiment_output_dir}")
    print()

    runner = ExperimentRunner(
        output_dir=experiment_output_dir,
        experiment_id=args.experiment_id,
    )
    results = runner.run_all(configs)

    if not results:
        print("\nNo experiments were run.")
        return

    results_path = experiment_output_dir / "results.json"
    print(f"\n{'=' * 60}")
    print(f"Done! {len(results)} results saved to {results_path}")
    print(f"{'=' * 60}")

    if not args.no_plot:
        from visualize import Visualizer
        viz = Visualizer(str(results_path))
        viz.plot_all(save=True)


if __name__ == "__main__":
    main()
