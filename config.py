"""Configuration constants and experiment definitions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "cache"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR

# ── Models ────────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    name: str
    dim: int
    matryoshka_dims: List[int]
    query_prefix: str = ""
    doc_prefix: str = ""
    supports_matryoshka: bool = True


MODELS = {
    "mxbai-embed-large-v1": ModelConfig(
        name="mixedbread-ai/mxbai-embed-large-v1",
        dim=1024,
        matryoshka_dims=[1024, 512, 256, 128, 64, 32, 16],
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        supports_matryoshka=True,
    ),
    "nomic-embed-text-v1.5": ModelConfig(
        name="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        matryoshka_dims=[768, 512, 256, 128, 64, 32, 16],
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        supports_matryoshka=True,
    ),
    "all-MiniLM-L6-v2": ModelConfig(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        matryoshka_dims=[384],
        query_prefix="",
        doc_prefix="",
        supports_matryoshka=False,
    ),
}

DEFAULT_MODELS = ["mxbai-embed-large-v1", "nomic-embed-text-v1.5"]

# ── Datasets ──────────────────────────────────────────────────────────────────

DATASETS = {
    "nfcorpus": {"corpus_size": 3633, "query_size": 323},
    "scifact": {"corpus_size": 5183, "query_size": 300},
    "fiqa": {"corpus_size": 57_638, "query_size": 648},
    "quora": {"corpus_size": 522_931, "query_size": 10000},
    "arguana": {"corpus_size": 8674, "query_size": 1406},
    "trec-covid": {"corpus_size": 171_332, "query_size": 50},
    "msmarco": {"corpus_size": 8841823, "query_size": 6980, "subsample": 1_000_000},
    "dbpedia": {"corpus_size": 4635922, "query_size": 400, "subsample": 1_000_000},
}

DEFAULT_DATASETS = ["fiqa","nfcorpus","scifact"]

# ── Sweep parameters ─────────────────────────────────────────────────────────

TRUNCATE_DIMS = [1024, 512, 256, 128, 64, 32, 16]
OVERSAMPLE = 8
DEFAULT_FUNNEL_FACTORS = [1]

# ── Retrieval and rescore methods ─────────────────────────────────────────────

RETRIEVAL_METHODS = ["float32", "int4","binary"]
RESCORE_METHODS = ["none", "float32", "int8", "int4", "binary", "binary_median", "lloyd_max"]

DEFAULT_RETRIEVAL = RETRIEVAL_METHODS
DEFAULT_RESCORE = ["none", "float32", "int8", "int4", "binary"]

# Quality hierarchy for pruning nonsensical combos.
# Retrieval ranks: how good is retrieval alone (Hamming binary < brute int4 < brute float32).
# Rescore ranks: fidelity of the rescore scoring function.
# Rescore must have strictly higher rank than retrieval rank to be worthwhile.
RETRIEVAL_QUALITY_RANK = {
    "binary": 0,
    "int4": 2,
    "float32": 4,
}

RESCORE_QUALITY_RANK = {
    "binary": 1,
    "binary_median": 1,
    "int4": 2,
    "lloyd_max": 2,
    "int8": 3,
    "float32": 4,
}


def is_valid_combo(retrieval: str, rescore: str) -> bool:
    """Check if a retrieval→rescore combination is sensible.

    Rules:
      - rescore="none" is always valid (retrieval-only).
      - Rescore must be strictly higher quality than retrieval
        (no point rescoring with the same or worse method).
      - float32 retrieval only makes sense with rescore="none"
        (it's already the best).
    """
    if rescore == "none":
        return True
    r_rank = RETRIEVAL_QUALITY_RANK.get(retrieval, -1)
    s_rank = RESCORE_QUALITY_RANK.get(rescore, -1)
    return s_rank > r_rank

# ── Evaluation parameters ─────────────────────────────────────────────────────

K_VALUES = [10, 100]

# ── Method display config (for visualization) ────────────────────────────────

# Colors assigned dynamically; these are defaults for common methods
METHOD_COLORS = {
    "float32": "#2ecc71",         # green
    "binary": "#e74c3c",          # red
    "int4": "#9b59b6",            # purple
    "binary→float32": "#3498db",  # blue
    "binary→int8": "#e67e22",     # orange
    "binary→int4": "#1abc9c",     # teal
    "binary→binary": "#f39c12",   # yellow
    "binary→binary_median": "#d35400",  # burnt orange
    "binary→lloyd_max": "#2c3e50",      # dark navy
    "int4→float32": "#16a085",    # dark teal
    "int4→int8": "#c0392b",       # dark red
    "binary→funnel→float32": "#2980b9", # steel blue
    "binary→funnel→binary": "#8e44ad",  # deep purple
}

METHOD_MARKERS = {
    "float32": "o",
    "binary": "^",
    "int4": "D",
    "binary→float32": "s",
    "binary→int8": "P",
    "binary→int4": "X",
    "binary→binary": "v",
    "binary→binary_median": "<",
    "binary→lloyd_max": ">",
    "binary→funnel→float32": "p",
    "binary→funnel→binary": "h",
}

# Fallback palette for methods not in the explicit maps
FALLBACK_COLORS = [
    "#16a085", "#27ae60", "#2980b9", "#8e44ad", "#f39c12",
    "#d35400", "#c0392b", "#7f8c8d", "#2c3e50", "#1abc9c",
]

FALLBACK_MARKERS = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "p", "h", "*"]

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Config for a single experiment run.

    One experiment = one (model, dataset, dim, retrieval, rescore) combo.
    """
    model: str
    dataset: str
    truncate_dim: int
    retrieval: str = "binary"
    rescore: str = "none"
    funnel: bool = False
    oversample: int = OVERSAMPLE
    funnel_factor: int = 1
    k_values: List[int] = field(default_factory=lambda: [10, 100])
    cache_dir: Path = Path("cache/embeddings")
    dataset_cache_dir: Path = Path("cache/datasets")

    @property
    def method_name(self) -> str:
        """Human-readable method string, e.g. 'binary→int8', 'binary→funnel→float32'."""
        if self.rescore == "none":
            return self.retrieval
        if self.funnel:
            return f"{self.retrieval}→funnel→{self.rescore}"
        return f"{self.retrieval}→{self.rescore}"


@dataclass
class EmbeddingData:
    """All data needed for search & evaluation."""
    doc_ids: List[str]
    query_ids: List[str]
    corpus_emb: np.ndarray
    query_emb: np.ndarray
    qrels: Dict[str, Dict[str, int]]
