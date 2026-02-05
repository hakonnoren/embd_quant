"""Embedding generation with caching and model abstraction."""
import numpy as np
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    """Generates and caches embeddings for corpus and queries."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[SentenceTransformer] = None

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("Using CUDA")
            elif torch.backends.mps.is_available():
                self.model = self.model.to("mps")
                print("Using MPS (Apple Silicon)")
            else:
                print("Using CPU")
        return self.model

    def _get_cache_path(self, dataset_name: str, embedding_type: str) -> Path:
        """Get cache file path for embeddings."""
        model_short = self.model_name.split("/")[-1]
        return self.cache_dir / f"{model_short}_{dataset_name}_{embedding_type}.npy"

    def embed_corpus(
        self, dataset_name: str, texts: List[str], force_recompute: bool = False
    ) -> np.ndarray:
        """Embed corpus texts with caching."""
        cache_path = self._get_cache_path(dataset_name, "corpus")

        if cache_path.exists() and not force_recompute:
            print(f"Loading cached corpus embeddings from {cache_path}")
            return np.load(cache_path)

        model = self._load_model()

        # Apply document prefix if specified
        if self.doc_prefix:
            texts = [self.doc_prefix + t for t in texts]

        print(f"Embedding {len(texts)} corpus documents...")
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        np.save(cache_path, embeddings)
        print(f"Cached corpus embeddings to {cache_path}")

        # Clear CUDA cache after large embedding jobs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embeddings

    def embed_queries(
        self, dataset_name: str, texts: List[str], force_recompute: bool = False
    ) -> np.ndarray:
        """Embed query texts with caching (applies query prefix)."""
        cache_path = self._get_cache_path(dataset_name, "queries")

        if cache_path.exists() and not force_recompute:
            print(f"Loading cached query embeddings from {cache_path}")
            return np.load(cache_path)

        model = self._load_model()

        # Apply query prefix if specified
        if self.query_prefix:
            texts = [self.query_prefix + t for t in texts]

        print(f"Embedding {len(texts)} queries...")
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        np.save(cache_path, embeddings)
        print(f"Cached query embeddings to {cache_path}")
        return embeddings

    def unload_model(self):
        """Unload model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model unloaded, GPU cache cleared")
