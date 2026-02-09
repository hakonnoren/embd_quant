#!/usr/bin/env python3
"""Download datasets and pre-compute embeddings.

Usage:
    # Prepare a single dataset with a single model
    python prepare_data.py --datasets scifact --models mxbai-embed-large-v1

    # Prepare multiple datasets and models
    python prepare_data.py --datasets scifact nfcorpus quora --models mxbai-embed-large-v1 nomic-embed-text-v1.5

    # Prepare all configured datasets and models
    python prepare_data.py --all

    # Override subsample size for large datasets
    python prepare_data.py --datasets msmarco --models all-MiniLM-L6-v2 --subsample 50000

    # Use full corpus (no subsampling) even for large datasets
    python prepare_data.py --datasets msmarco --models all-MiniLM-L6-v2 --no-subsample

    # Force re-download and re-embed (ignore cache)
    python prepare_data.py --datasets scifact --models mxbai-embed-large-v1 --force

    # List available datasets and models
    python prepare_data.py --list
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from config import MODELS, DATASETS, CACHE_DIR
from data_loader import MTEBDataLoader
from embedder import Embedder


def list_available():
    """Print available datasets and models."""
    print("Available models:")
    for key, cfg in MODELS.items():
        dims = ", ".join(str(d) for d in cfg.matryoshka_dims)
        print(f"  {key:<30} dim={cfg.dim}  matryoshka=[{dims}]")

    print("\nAvailable datasets:")
    for key, cfg in DATASETS.items():
        sub = cfg.get("subsample")
        sub_str = f"  subsample={sub}" if sub else ""
        print(f"  {key:<20} corpus={cfg['corpus_size']:>9,}  queries={cfg['query_size']:>6,}{sub_str}")


def prepare(dataset_name: str, model_key: str, subsample: Optional[int], force: bool, batch_size: int):
    """Download dataset and compute embeddings for one dataset+model pair."""
    model_config = MODELS[model_key]

    print(f"\n{'=' * 60}")
    print(f"Preparing: {model_key} / {dataset_name}")
    print(f"{'=' * 60}")

    # 1. Download and cache dataset
    loader = MTEBDataLoader(CACHE_DIR)
    corpus, queries, qrels = loader.load_dataset(dataset_name, subsample=subsample)
    doc_ids, doc_texts, query_ids, query_texts = loader.get_texts_for_embedding(corpus, queries)

    print(f"  Corpus: {len(doc_ids)} docs, Queries: {len(query_ids)}")

    # 2. Compute and cache embeddings
    embedder = Embedder(
        model_name=model_config.name,
        cache_dir=CACHE_DIR,
        query_prefix=model_config.query_prefix,
        doc_prefix=model_config.doc_prefix,
        batch_size=batch_size,
    )

    corpus_emb = embedder.embed_corpus(dataset_name, doc_texts, force_recompute=force)
    query_emb = embedder.embed_queries(dataset_name, query_texts, force_recompute=force)

    print(f"  Corpus embeddings: {corpus_emb.shape}")
    print(f"  Query embeddings:  {query_emb.shape}")
    print(f"  Done: {model_key} / {dataset_name}")

    embedder.unload_model()


def main():
    parser = argparse.ArgumentParser(
        description="Download MTEB datasets and pre-compute embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+", metavar="NAME",
        help="Dataset names to prepare (from config.py)",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="NAME",
        help="Model names to prepare (from config.py)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Prepare all configured datasets and models",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and models",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Override subsample size for large datasets",
    )
    parser.add_argument(
        "--no-subsample", action="store_true",
        help="Disable subsampling even for large datasets",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download and re-embed (ignore cache)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for embedding (default: 32)",
    )
    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if not args.all and not (args.datasets and args.models):
        parser.print_help()
        print("\nError: specify --datasets and --models, or use --all")
        sys.exit(1)

    datasets = list(DATASETS.keys()) if args.all else args.datasets
    models = list(MODELS.keys()) if args.all else args.models

    # Validate names
    for d in datasets:
        if d not in DATASETS:
            print(f"Error: unknown dataset '{d}'. Use --list to see available datasets.")
            sys.exit(1)
    for m in models:
        if m not in MODELS:
            print(f"Error: unknown model '{m}'. Use --list to see available models.")
            sys.exit(1)

    # Process each model×dataset pair (load model once per model)
    for model_key in models:
        for dataset_name in datasets:
            # Determine subsample size
            if args.no_subsample:
                subsample = None
            elif args.subsample is not None:
                subsample = args.subsample
            else:
                subsample = DATASETS[dataset_name].get("subsample")

            prepare(dataset_name, model_key, subsample, args.force, args.batch_size)

    print(f"\nAll done. Cache directory: {CACHE_DIR}")


if __name__ == "__main__":
    main()
