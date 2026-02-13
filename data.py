"""Unified data loading for experiments.

Auto-embeds datasets if cached embeddings are not found.
"""

from pathlib import Path

import numpy as np

from config import EmbeddingData, MODELS, DATASETS
from data_loader import MTEBDataLoader


def _embed_if_missing(
    model: str,
    dataset: str,
    doc_texts: list,
    query_texts: list,
    cache_dir: Path,
) -> None:
    """Generate and cache embeddings if .npy files don't exist."""
    from embedder import Embedder

    cfg = MODELS[model]
    embedder = Embedder(
        model_name=cfg.name,
        cache_dir=cache_dir,
        query_prefix=cfg.query_prefix,
        doc_prefix=cfg.doc_prefix,
    )
    corpus_path = cache_dir / f"{model}_{dataset}_corpus.npy"
    query_path = cache_dir / f"{model}_{dataset}_queries.npy"

    if not corpus_path.exists():
        print(f"Embedding corpus for {model}/{dataset} ({len(doc_texts)} docs)...")
        embedder.embed_corpus(dataset, doc_texts)

    if not query_path.exists():
        print(f"Embedding queries for {model}/{dataset} ({len(query_texts)} queries)...")
        embedder.embed_queries(dataset, query_texts)

    embedder.unload_model()


def load_data(
    model: str,
    dataset: str,
    cache_dir: Path = Path("cache/embeddings"),
    dataset_cache_dir: Path = Path("cache/datasets"),
) -> EmbeddingData:
    """Load dataset, embeddings, and filter queries to those with positive qrels.

    Auto-embeds if cached .npy files are missing.

    Args:
        model: Model short name (e.g. "mxbai-embed-large-v1").
        dataset: Dataset name (e.g. "fiqa").
        cache_dir: Directory containing cached .npy embeddings.
        dataset_cache_dir: Directory containing cached dataset pickles.

    Returns:
        EmbeddingData with filtered queries and positive-only qrels.
    """
    # Check for subsample config
    ds_cfg = DATASETS.get(dataset, {})
    subsample = ds_cfg.get("subsample", None)

    loader = MTEBDataLoader(dataset_cache_dir)
    corpus, queries_dict, qrels = loader.load_dataset(dataset, subsample=subsample)
    doc_ids, doc_texts, all_query_ids, query_texts = loader.get_texts_for_embedding(corpus, queries_dict)

    # Auto-embed if cache files are missing
    corpus_path = cache_dir / f"{model}_{dataset}_corpus.npy"
    query_path = cache_dir / f"{model}_{dataset}_queries.npy"
    if not corpus_path.exists() or not query_path.exists():
        _embed_if_missing(model, dataset, doc_texts, query_texts, cache_dir)

    corpus_emb = np.load(corpus_path)
    all_query_emb = np.load(query_path)
    assert corpus_emb.shape[0] == len(doc_ids), (
        f"Corpus mismatch: {corpus_emb.shape[0]} vs {len(doc_ids)}"
    )
    assert all_query_emb.shape[0] == len(all_query_ids), (
        f"Query mismatch: {all_query_emb.shape[0]} vs {len(all_query_ids)}"
    )

    # Keep only queries with positive relevance judgments
    qrels_pos = {
        qid: {did: s for did, s in rels.items() if s > 0}
        for qid, rels in qrels.items()
    }
    qrels_pos = {qid: rels for qid, rels in qrels_pos.items() if rels}

    keep_mask = [qid in qrels_pos for qid in all_query_ids]
    query_ids = [qid for qid, keep in zip(all_query_ids, keep_mask) if keep]
    query_emb = all_query_emb[keep_mask]

    print(
        f"Loaded {dataset}: {len(doc_ids)} docs ({corpus_emb.shape}), "
        f"{len(query_ids)}/{len(all_query_ids)} queries with positive qrels"
    )

    return EmbeddingData(
        doc_ids=doc_ids,
        query_ids=query_ids,
        corpus_emb=corpus_emb,
        query_emb=query_emb,
        qrels=qrels_pos,
    )
