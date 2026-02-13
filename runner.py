"""Unified experiment runner with persistence and skip-if-done.

Dispatches to search.py for retrieval and rescore.py for rescoring.
Groups experiments by (model, dataset) to load data once.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from config import MODELS, ExperimentConfig, EmbeddingData
from data import load_data
from metrics import RetrievalMetrics
from search import build_index, initial_search, index_memory_bytes, truncate, binarize
from rescore import dispatch_rescore, rescore_funnel


@dataclass
class ExperimentResult:
    """Results from a single experiment."""

    model: str
    dataset: str
    method: str
    truncate_dim: int
    oversample: int
    funnel_factor: int
    retrieval: str
    rescore: str
    funnel: bool
    metrics: Dict[str, float]

    def key(self) -> Tuple:
        return (self.model, self.dataset, self.method,
                self.truncate_dim, self.oversample, self.funnel_factor)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "method": self.method,
            "truncate_dim": self.truncate_dim,
            "oversample": self.oversample,
            "funnel_factor": self.funnel_factor,
            "retrieval": self.retrieval,
            "rescore": self.rescore,
            "funnel": self.funnel,
            "metrics": self.metrics,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentResult":
        return ExperimentResult(
            model=d["model"],
            dataset=d["dataset"],
            method=d["method"],
            truncate_dim=d["truncate_dim"],
            oversample=d.get("oversample", 0),
            funnel_factor=d.get("funnel_factor", 0),
            retrieval=d.get("retrieval", ""),
            rescore=d.get("rescore", "none"),
            funnel=d.get("funnel", False),
            metrics=d["metrics"],
        )


class ExperimentRunner:
    """Run unified retrieval + rescore experiments with persistence."""

    def __init__(self, output_dir: Path, experiment_id: str = ""):
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        self.results: List[ExperimentResult] = []
        self.completed: Set[Tuple] = set()

        if output_dir:
            self._load_existing_results()

    def _load_existing_results(self):
        results_file = self.output_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                results_list = data.get("results", data)
                if isinstance(results_list, list):
                    for r in results_list:
                        result = ExperimentResult.from_dict(r)
                        self.results.append(result)
                        self.completed.add(result.key())
            print(f"Loaded {len(self.results)} existing results from {results_file}")

    def _save_results(self):
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

    def _record(self, result: ExperimentResult):
        self.results.append(result)
        self.completed.add(result.key())
        self._save_results()
        m = result.metrics
        print(f"  NDCG@10={m.get('ndcg@10', 0):.4f} | R@100={m.get('recall@100', 0):.4f} | "
              f"search={m.get('search_sec', 0):.3f}s | rescore={m.get('rescore_sec', 0):.3f}s")

    def run_all(self, configs: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run all experiment configs. Groups by (model, dataset) to load data once."""

        # Group configs by (model, dataset)
        groups: Dict[Tuple[str, str], List[ExperimentConfig]] = {}
        for cfg in configs:
            dk = (cfg.model, cfg.dataset)
            groups.setdefault(dk, []).append(cfg)

        for (model, dataset), cfgs in groups.items():
            print(f"\n{'=' * 60}")
            print(f"Model: {model} | Dataset: {dataset}")
            print(f"{'=' * 60}")

            # Load data once for this (model, dataset) pair
            ref_cfg = cfgs[0]
            data = load_data(
                model, dataset,
                cache_dir=ref_cfg.cache_dir,
                dataset_cache_dir=ref_cfg.dataset_cache_dir,
            )
            full_dim = data.corpus_emb.shape[1]

            for cfg in cfgs:
                # Skip dims that exceed the model's full dim
                if cfg.truncate_dim > full_dim:
                    continue

                key = (cfg.model, cfg.dataset, cfg.method_name,
                       cfg.truncate_dim, cfg.oversample, cfg.funnel_factor)
                if key in self.completed:
                    print(f"  Skipping: {cfg.method_name} dim={cfg.truncate_dim} "
                          f"os={cfg.oversample} ff={cfg.funnel_factor} (done)")
                    continue

                print(f"\n── {cfg.method_name} | dim={cfg.truncate_dim} "
                      f"os={cfg.oversample} ff={cfg.funnel_factor} ──")

                result = self._run_single(cfg, data)
                self._record(result)

        return self.results

    def _run_single(self, cfg: ExperimentConfig, data: EmbeddingData) -> ExperimentResult:
        """Run a single experiment config."""
        max_k = max(cfg.k_values)
        doc_id_to_idx = {did: i for i, did in enumerate(data.doc_ids)}

        # Build retrieval index
        index = build_index(data.corpus_emb, cfg.retrieval, cfg.truncate_dim)

        # Run initial retrieval
        scores, candidate_ids, search_sec = initial_search(
            index, data.query_emb, max_k, cfg.oversample,
            cfg.retrieval, cfg.truncate_dim,
        )

        # Rescore (if needed)
        rescore_sec = 0.0
        rescore_mem_bytes = 0

        if cfg.rescore == "none":
            final_indices = candidate_ids[:, :max_k]
        elif cfg.funnel and cfg.retrieval == "binary":
            # Funnel: binary residual prune → rescore
            full_dim = data.corpus_emb.shape[1]
            has_residual = cfg.truncate_dim < full_dim

            if has_residual:
                corpus_residual = data.corpus_emb[:, cfg.truncate_dim:]
                query_residual = data.query_emb[:, cfg.truncate_dim:]
                residual_dim = corpus_residual.shape[1]
                residual_corpus_bin = binarize(corpus_residual)
                residual_query_bin = binarize(query_residual)
            else:
                residual_corpus_bin = None
                residual_query_bin = None
                residual_dim = 0

            t1 = time.time()
            final_indices, rescore_mem_bytes = rescore_funnel(
                candidate_ids, -scores, cfg.truncate_dim,
                residual_corpus_bin, residual_query_bin, residual_dim,
                cfg.rescore, data.query_emb, data.corpus_emb, max_k,
                funnel_factor=cfg.funnel_factor,
            )
            rescore_sec = time.time() - t1
        else:
            # Direct rescore
            t1 = time.time()
            final_indices, rescore_mem_bytes = dispatch_rescore(
                cfg.rescore, candidate_ids, data.query_emb, data.corpus_emb, max_k,
            )
            rescore_sec = time.time() - t1

        # Compute metrics
        metrics = RetrievalMetrics.compute_all_metrics(
            final_indices, data.qrels, doc_id_to_idx, data.query_ids, cfg.k_values,
        )

        idx_mem = index_memory_bytes(index)
        metrics["search_sec"] = search_sec
        metrics["rescore_sec"] = rescore_sec
        metrics["index_mem_mb"] = idx_mem / (1024 ** 2)
        metrics["rescore_vec_mem_mb"] = rescore_mem_bytes / (1024 ** 2)

        return ExperimentResult(
            model=cfg.model,
            dataset=cfg.dataset,
            method=cfg.method_name,
            truncate_dim=cfg.truncate_dim,
            oversample=cfg.oversample,
            funnel_factor=cfg.funnel_factor,
            retrieval=cfg.retrieval,
            rescore=cfg.rescore,
            funnel=cfg.funnel,
            metrics=metrics,
        )
