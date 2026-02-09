"""Orchestrates quantization experiments."""
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path

from config import (
    MODELS,
    DATASETS,
    QUANTIZATION_SCHEMES,
    K_VALUES,
    RESCORE_MULTIPLIER,
    CACHE_DIR,
    ROTATION_SEED,
)
from data_loader import MTEBDataLoader
from embedder import Embedder
from quantization import QuantizationHandler
from search_optimized import BruteForceSearch
from metrics import RetrievalMetrics
from rotation import create_rotation


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
    rotation: str = "none"  # Rotation method used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "quantization": self.quantization,
            "matryoshka_dim": self.matryoshka_dim,
            "metrics": self.metrics,
            "latency_seconds": self.latency_seconds,
            "memory_bytes": self.memory_bytes,
            "rotation": self.rotation,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentResult":
        return ExperimentResult(
            model=d["model"],
            dataset=d["dataset"],
            quantization=d["quantization"],
            matryoshka_dim=d["matryoshka_dim"],
            metrics=d["metrics"],
            latency_seconds=d["latency_seconds"],
            memory_bytes=d["memory_bytes"],
            rotation=d.get("rotation", "none"),
        )

    def key(self) -> Tuple[str, str, str, int, str]:
        """Unique key for this experiment configuration."""
        return (self.model, self.dataset, self.quantization, self.matryoshka_dim, self.rotation)


class ExperimentRunner:
    """Run quantization experiments across models, datasets, and configurations."""

    def __init__(self, cache_dir: Path = CACHE_DIR, batch_size: int = 32, output_dir: Path = None, experiment_id: str = None):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        self.data_loader = MTEBDataLoader(cache_dir / "datasets")
        self.results: List[ExperimentResult] = []
        self.completed: Set[Tuple[str, str, str, int, str]] = set()

        # Load existing results if output_dir specified
        if output_dir:
            self._load_existing_results()

    def _load_existing_results(self):
        """Load existing results from disk."""
        results_file = self.output_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                results_list = data.get("results", data)  # Handle both formats
                if isinstance(results_list, list):
                    for r in results_list:
                        result = ExperimentResult.from_dict(r)
                        self.results.append(result)
                        self.completed.add(result.key())
            print(f"Loaded {len(self.results)} existing results from {results_file}")

    def _save_results(self):
        """Save results to disk."""
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.output_dir / "results.json"
        data = {
            "experiment_id": self.experiment_id,
            "results": [r.to_dict() for r in self.results],
        }
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

    def _is_completed(self, model: str, dataset: str, quantization: str, dim: int, rotation: str = "none") -> bool:
        """Check if experiment already completed."""
        return (model, dataset, quantization, dim, rotation) in self.completed

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
        rotation: str = "none",
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

        # Create rotation object if needed (for quantized representations)
        rotator = None
        if rotation != "none" and quantization in ["int8", "int8_asym", "binary", "binary_asym", "binary_rescore", "binary_median", "binary_median_asym", "quaternary_asym", "lloyd_max_gauss"]:
            # Check if dimension is compatible with Hadamard (must be power of 2)
            if rotation == "hadamard" and (dim & (dim - 1)) != 0:
                print(f"    (Hadamard requires power-of-2 dim, using QR for dim={dim})")
                rotation = "qr"
            rotator = create_rotation(dim, method=rotation, seed=ROTATION_SEED)

        # Apply quantization and run search
        if quantization == "float32":
            indices, _, latency = BruteForceSearch.search_float(
                query_emb, corpus_emb, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "float32")

        elif quantization == "int8":
            # Apply rotation before int8 quantization if specified
            if rotator is not None:
                corpus_for_quant = rotator.rotate(corpus_emb)
                query_for_quant = rotator.rotate(query_emb)
                calibration_for_quant = rotator.rotate(calibration_emb) if calibration_emb is not None else None
            else:
                corpus_for_quant = corpus_emb
                query_for_quant = query_emb
                calibration_for_quant = calibration_emb

            search_corpus = quantizer.quantize_to_int8(corpus_for_quant, calibration_for_quant)
            search_query = quantizer.quantize_to_int8(query_for_quant)
            indices, _, latency = BruteForceSearch.search_int8(
                search_query, search_corpus, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "int8")

        elif quantization == "binary":
            # Apply rotation before binary quantization if specified
            if rotator is not None:
                corpus_for_binary = rotator.rotate(corpus_emb)
                query_for_binary = rotator.rotate(query_emb)
            else:
                corpus_for_binary = corpus_emb
                query_for_binary = query_emb

            search_corpus = quantizer.quantize_to_binary(corpus_for_binary)
            search_query = quantizer.quantize_to_binary(query_for_binary)
            indices, _, latency = BruteForceSearch.search_binary(
                search_query, search_corpus, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "binary")

        elif quantization == "int8_asym":
            # Asymmetric: only corpus is quantized to int8, query stays float32
            if rotator is not None:
                corpus_for_quant = rotator.rotate(corpus_emb)
                query_for_search = rotator.rotate(query_emb)
                calibration_for_quant = rotator.rotate(calibration_emb) if calibration_emb is not None else None
            else:
                corpus_for_quant = corpus_emb
                query_for_search = query_emb
                calibration_for_quant = calibration_emb

            search_corpus = quantizer.quantize_to_int8(corpus_for_quant, calibration_for_quant)
            indices, _, latency = BruteForceSearch.search_int8_asymmetric(
                query_for_search, search_corpus, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "int8")

        elif quantization == "binary_asym":
            # Asymmetric: only corpus is quantized to binary, query stays float32
            if rotator is not None:
                corpus_for_binary = rotator.rotate(corpus_emb)
                query_for_search = rotator.rotate(query_emb)
            else:
                corpus_for_binary = corpus_emb
                query_for_search = query_emb

            binary_corpus = quantizer.quantize_to_binary(corpus_for_binary)
            indices, _, latency = BruteForceSearch.search_binary_asymmetric(
                query_for_search, binary_corpus, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "binary")

        elif quantization == "binary_median":
            # Binary with per-dimension median thresholds (symmetric Hamming)
            if rotator is not None:
                corpus_for_binary = rotator.rotate(corpus_emb)
                query_for_binary = rotator.rotate(query_emb)
            else:
                corpus_for_binary = corpus_emb
                query_for_binary = query_emb

            # Compute median thresholds from corpus, then center both sides
            thresholds = quantizer.compute_binary_thresholds(corpus_for_binary)
            corpus_centered = quantizer.center_for_binary(corpus_for_binary, thresholds)
            query_centered = quantizer.center_for_binary(query_for_binary, thresholds)

            search_corpus = quantizer.quantize_to_binary(corpus_centered)
            search_query = quantizer.quantize_to_binary(query_centered)
            indices, _, latency = BruteForceSearch.search_binary(
                search_query, search_corpus, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "binary_median")

        elif quantization == "binary_median_asym":
            # Binary median with asymmetric search (float query, binary corpus)
            if rotator is not None:
                corpus_for_binary = rotator.rotate(corpus_emb)
                query_for_search = rotator.rotate(query_emb)
            else:
                corpus_for_binary = corpus_emb
                query_for_search = query_emb

            # Compute median thresholds from corpus, center both sides
            thresholds = quantizer.compute_binary_thresholds(corpus_for_binary)
            corpus_centered = quantizer.center_for_binary(corpus_for_binary, thresholds)
            query_centered = quantizer.center_for_binary(query_for_search, thresholds)

            binary_corpus = quantizer.quantize_to_binary(corpus_centered)
            indices, _, latency = BruteForceSearch.search_binary_asymmetric(
                query_centered, binary_corpus, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "binary_median")

        elif quantization == "quaternary_asym":
            # 2-bit quartile quantization with asymmetric search
            if rotator is not None:
                corpus_for_quant = rotator.rotate(corpus_emb)
                query_for_search = rotator.rotate(query_emb)
            else:
                corpus_for_quant = corpus_emb
                query_for_search = query_emb

            cal = quantizer.calibrate_quaternary(corpus_for_quant)
            codes = quantizer.quantize_to_quaternary(corpus_for_quant, cal["boundaries"])
            reconstructed = quantizer.reconstruct_quaternary(codes, cal["centroids"])
            indices, _, latency = BruteForceSearch.search_2bit_asymmetric(
                query_for_search, reconstructed, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "quaternary")

        elif quantization == "lloyd_max_gauss":
            # 2-bit Gaussian-optimal Lloyd-Max quantization
            if rotator is not None:
                corpus_for_quant = rotator.rotate(corpus_emb)
                query_for_search = rotator.rotate(query_emb)
            else:
                corpus_for_quant = corpus_emb
                query_for_search = query_emb

            cal = quantizer.calibrate_lloyd_max(corpus_for_quant)
            codes = quantizer.quantize_to_lloyd_max(corpus_for_quant, cal["medians"], cal["stds"])
            # Efficient scoring: q_scaled = q * sigma, score = q_scaled @ levels[codes].T
            recon_z = quantizer.reconstruct_lloyd_max(codes, cal["stds"])
            q_scaled = query_for_search * cal["stds"]
            indices, _, latency = BruteForceSearch.search_2bit_asymmetric(
                q_scaled, recon_z, max(K_VALUES)
            )
            memory = QuantizationHandler.compute_memory_bytes(n_vectors, dim, "lloyd_max")

        elif quantization == "binary_rescore":
            # Apply rotation before binary quantization if specified
            if rotator is not None:
                corpus_for_binary = rotator.rotate(corpus_emb)
                query_for_binary = rotator.rotate(query_emb)
            else:
                corpus_for_binary = corpus_emb
                query_for_binary = query_emb

            binary_corpus = quantizer.quantize_to_binary(corpus_for_binary)
            binary_query = quantizer.quantize_to_binary(query_for_binary)

            indices, _, latency = BruteForceSearch.search_binary_with_rescore(
                binary_query,
                binary_corpus,
                query_emb,  # Rescore with original (unrotated) embeddings
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
            rotation=rotation,
        )

    def run_all_experiments(
        self,
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        quantizations: Optional[List[str]] = None,
        include_matryoshka_combos: bool = True,
        rotations: Optional[List[str]] = None,
    ) -> List[ExperimentResult]:
        """Run experiments across all specified configurations."""

        models = models or list(MODELS.keys())
        datasets = datasets or list(DATASETS.keys())
        quantizations = quantizations or QUANTIZATION_SCHEMES
        rotations = rotations or ["none"]

        for model_key in models:
            model_config = MODELS[model_key]
            embedder = Embedder(
                model_config.name,
                self.cache_dir / "embeddings",
                model_config.query_prefix,
                model_config.doc_prefix,
                batch_size=self.batch_size,
            )

            for dataset_name in datasets:
                print(f"\n{'=' * 60}")
                print(f"Model: {model_key} | Dataset: {dataset_name}")
                print(f"{'=' * 60}")

                # Load dataset (with subsampling for large corpora)
                subsample = DATASETS.get(dataset_name, {}).get("subsample")
                corpus, queries, qrels = self.data_loader.load_dataset(
                    dataset_name, subsample=subsample
                )
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
                        # Determine which rotations to test for this quantization
                        # Rotation applies to quantized types (int8, binary, binary_rescore)
                        # Float32 doesn't benefit from rotation
                        if quant in ["int8", "int8_asym", "binary", "binary_asym", "binary_rescore", "binary_median", "binary_median_asym", "quaternary_asym", "lloyd_max_gauss"]:
                            rotations_to_test = rotations
                        else:
                            rotations_to_test = ["none"]

                        for rot in rotations_to_test:
                            # Skip if already completed
                            if self._is_completed(model_key, dataset_name, quant, matryoshka_dim, rot):
                                rot_str = f", rot={rot}" if rot != "none" else ""
                                print(f"  Skipping: dim={matryoshka_dim}, quant={quant}{rot_str} (already done)")
                                continue

                            rot_str = f", rot={rot}" if rot != "none" else ""
                            print(f"  Testing: dim={matryoshka_dim}, quant={quant}{rot_str}...", end=" ")

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
                                rotation=rot,
                            )

                            self.results.append(result)
                            self.completed.add(result.key())

                            # Print summary
                            print(
                                f"R@10: {result.metrics.get('recall@10', 0):.4f} | "
                                f"NDCG@10: {result.metrics.get('ndcg@10', 0):.4f} | "
                                f"Latency: {result.latency_seconds:.3f}s | "
                                f"Mem: {result.memory_bytes / 1024 / 1024:.2f}MB"
                            )

                            # Save incrementally
                            self._save_results()

            # Unload model after all datasets to free GPU memory
            embedder.unload_model()

        return self.results
