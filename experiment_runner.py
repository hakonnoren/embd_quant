"""Orchestrates quantization experiments."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import (
    MODELS,
    DATASETS,
    QUANTIZATION_SCHEMES,
    K_VALUES,
    RESCORE_MULTIPLIER,
    CACHE_DIR,
)
from data_loader import MTEBDataLoader
from embedder import Embedder
from quantization import QuantizationHandler
from search import BruteForceSearch
from metrics import RetrievalMetrics


@dataclass
class ExperimentResult:
    """Results from a single experiment configuration."""

    model: str
    dataset: str
    quantization: str
    matryoshka_dim: int
    metrics: Dict[str, float]
    latency_seconds: float
    memory_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "quantization": self.quantization,
            "matryoshka_dim": self.matryoshka_dim,
            "metrics": self.metrics,
            "latency_seconds": self.latency_seconds,
            "memory_bytes": self.memory_bytes,
        }


class ExperimentRunner:
    """Run quantization experiments across models, datasets, and configurations."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.data_loader = MTEBDataLoader(cache_dir / "datasets")
        self.results: List[ExperimentResult] = []

    def run_single_experiment(
        self,
        model_key: str,
        dataset_name: str,
        quantization: str,
        matryoshka_dim: int,
        corpus_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        qrels: Dict,
        doc_ids: List[str],
        query_ids: List[str],
        calibration_embeddings: Optional[np.ndarray] = None,
    ) -> ExperimentResult:
        """Run a single experiment configuration."""

        quantizer = QuantizationHandler()
        doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

        # Apply Matryoshka truncation if needed
        if matryoshka_dim < corpus_embeddings.shape[1]:
            corpus_emb = quantizer.truncate_matryoshka(corpus_embeddings, matryoshka_dim)
            query_emb = quantizer.truncate_matryoshka(query_embeddings, matryoshka_dim)
            if calibration_embeddings is not None:
                calibration_emb = quantizer.truncate_matryoshka(
                    calibration_embeddings, matryoshka_dim
                )
            else:
                calibration_emb = None
        else:
            corpus_emb = corpus_embeddings.copy()
            query_emb = query_embeddings.copy()
            calibration_emb = calibration_embeddings

        n_vectors = corpus_emb.shape[0]
        dim = matryoshka_dim

        # Apply quantization and run search
        if quantization == "float32":
            indices, scores, latency = BruteForceSearch.search_float(
                query_emb, corpus_emb, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "float32")

        elif quantization == "int8":
            search_corpus = quantizer.quantize_to_int8(corpus_emb, calibration_emb)
            search_query = quantizer.quantize_to_int8(query_emb)
            indices, scores, latency = BruteForceSearch.search_int8(
                search_query, search_corpus, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "int8")

        elif quantization == "binary":
            search_corpus = quantizer.quantize_to_binary(corpus_emb)
            search_query = quantizer.quantize_to_binary(query_emb)
            indices, scores, latency = BruteForceSearch.search_binary(
                search_query, search_corpus, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "binary")

        elif quantization == "binary_rescore":
            binary_corpus = quantizer.quantize_to_binary(corpus_emb)
            binary_query = quantizer.quantize_to_binary(query_emb)

            indices, scores, latency = BruteForceSearch.search_binary_with_rescore(
                binary_query,
                binary_corpus,
                query_emb,
                corpus_emb,
                k=max(K_VALUES),
                rescore_multiplier=RESCORE_MULTIPLIER,
            )
            # Memory is primarily the binary index (float corpus loaded on demand)
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "binary")

        else:
            raise ValueError(f"Unknown quantization: {quantization}")

        # Compute metrics
        metrics = RetrievalMetrics.compute_all_metrics(
            indices, qrels, doc_id_to_idx, query_ids, K_VALUES
        )

        return ExperimentResult(
            model=model_key,
            dataset=dataset_name,
            quantization=quantization,
            matryoshka_dim=matryoshka_dim,
            metrics=metrics,
            latency_seconds=latency,
            memory_bytes=memory,
        )

    def run_all_experiments(
        self,
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        quantizations: Optional[List[str]] = None,
        include_matryoshka_combos: bool = True,
    ) -> List[ExperimentResult]:
        """Run experiments across all specified configurations."""

        models = models or list(MODELS.keys())
        datasets = datasets or list(DATASETS.keys())
        quantizations = quantizations or QUANTIZATION_SCHEMES

        for model_key in models:
            model_config = MODELS[model_key]
            embedder = Embedder(
                model_config.name,
                self.cache_dir / "embeddings",
                model_config.query_prefix,
                model_config.doc_prefix,
            )

            for dataset_name in datasets:
                print(f"\n{'=' * 60}")
                print(f"Model: {model_key} | Dataset: {dataset_name}")
                print(f"{'=' * 60}")

                # Load dataset
                corpus, queries, qrels = self.data_loader.load_dataset(dataset_name)
                doc_ids, doc_texts, query_ids, query_texts = (
                    self.data_loader.get_texts_for_embedding(corpus, queries)
                )

                # Generate embeddings
                corpus_embeddings = embedder.embed_corpus(dataset_name, doc_texts)
                query_embeddings = embedder.embed_queries(dataset_name, query_texts)

                # Use corpus as calibration for int8
                calibration_embeddings = corpus_embeddings

                # Determine Matryoshka dimensions to test
                if include_matryoshka_combos and model_config.supports_matryoshka:
                    dims_to_test = model_config.matryoshka_dims
                else:
                    dims_to_test = [model_config.dim]

                for matryoshka_dim in dims_to_test:
                    for quant in quantizations:
                        print(f"  Testing: dim={matryoshka_dim}, quant={quant}...", end=" ")

                        result = self.run_single_experiment(
                            model_key=model_key,
                            dataset_name=dataset_name,
                            quantization=quant,
                            matryoshka_dim=matryoshka_dim,
                            corpus_embeddings=corpus_embeddings,
                            query_embeddings=query_embeddings,
                            qrels=qrels,
                            doc_ids=doc_ids,
                            query_ids=query_ids,
                            calibration_embeddings=calibration_embeddings,
                        )

                        self.results.append(result)

                        # Print summary
                        print(
                            f"R@10: {result.metrics.get('recall@10', 0):.4f} | "
                            f"NDCG@10: {result.metrics.get('ndcg@10', 0):.4f} | "
                            f"Latency: {result.latency_seconds:.3f}s | "
                            f"Mem: {result.memory_bytes / 1024 / 1024:.2f}MB"
                        )

        return self.results
