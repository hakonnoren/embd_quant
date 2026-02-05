"""Configuration constants and experiment definitions."""
from dataclasses import dataclass
from typing import List
from pathlib import Path

# Paths
CACHE_DIR = Path(__file__).parent / "cache"
RESULTS_DIR = Path(__file__).parent / "results"


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
        matryoshka_dims=[1024, 512, 256, 128, 64],
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        supports_matryoshka=True,
    ),
    "nomic-embed-text-v1.5": ModelConfig(
        name="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        matryoshka_dims=[768, 512, 256, 128, 64],
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        supports_matryoshka=True,
    ),
    "all-MiniLM-L6-v2": ModelConfig(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        matryoshka_dims=[384],  # No Matryoshka support
        query_prefix="",
        doc_prefix="",
        supports_matryoshka=False,
    ),
}

# Datasets with approximate sizes
# Names must match HuggingFace mteb/{name} format
DATASETS = {
    "nfcorpus": {"corpus_size": 3633, "query_size": 323},
    "scifact": {"corpus_size": 5183, "query_size": 300},
    "fiqa": {"corpus_size": 57638, "query_size": 648},
}

# Quantization schemes
QUANTIZATION_SCHEMES = ["float32", "int8", "binary", "binary_rescore"]

# Evaluation parameters
K_VALUES = [10, 100]
RESCORE_MULTIPLIER = 4  # For binary retrieval rescoring
