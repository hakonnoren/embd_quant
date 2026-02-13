"""Index building and initial retrieval for all retrieval methods.

Supports float32 (FAISS IndexFlatIP), binary (FAISS IndexBinaryFlat),
and int4 (NumPy brute-force with reconstructed quaternary vectors).
"""

import time
from typing import Tuple

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from quantization import QuantizationHandler


# ── Truncation & binarization ─────────────────────────────────────────────────

def truncate(emb: np.ndarray, dim: int) -> np.ndarray:
    """Truncate embeddings to first `dim` dimensions."""
    if dim >= emb.shape[1]:
        return emb
    return emb[:, :dim]


def binarize(emb: np.ndarray) -> np.ndarray:
    """Pack float embeddings to binary uint8 (sign-based)."""
    return np.packbits((emb > 0).astype(np.uint8), axis=1)


# ── Index wrappers ────────────────────────────────────────────────────────────

class FloatIndex:
    """FAISS IndexFlatIP wrapper."""

    def __init__(self, corpus_emb: np.ndarray):
        self.d = corpus_emb.shape[1]
        self.ntotal = corpus_emb.shape[0]
        self.type = "float32"
        corpus = np.ascontiguousarray(corpus_emb.astype(np.float32))
        if HAS_FAISS:
            self._index = faiss.IndexFlatIP(self.d)
            self._index.add(corpus)
        else:
            self._corpus = corpus

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = min(k, self.ntotal)
        queries = np.ascontiguousarray(queries.astype(np.float32))
        if HAS_FAISS:
            scores, indices = self._index.search(queries, k)
        else:
            sims = queries @ self._corpus.T
            indices = np.argpartition(-sims, k, axis=1)[:, :k]
            for i in range(len(indices)):
                idx = indices[i]
                indices[i] = idx[np.argsort(-sims[i, idx])]
            scores = np.take_along_axis(sims, indices, axis=1)
        return scores, indices


class BinaryIndex:
    """FAISS IndexBinaryFlat wrapper (Hamming distance)."""

    def __init__(self, corpus_bin: np.ndarray, num_bits: int):
        self.d = num_bits
        self.ntotal = corpus_bin.shape[0]
        self.type = "binary"
        corpus = np.ascontiguousarray(corpus_bin.astype(np.uint8))
        if HAS_FAISS:
            self._index = faiss.IndexBinaryFlat(num_bits)
            self._index.add(corpus)
        else:
            self._corpus = corpus

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = min(k, self.ntotal)
        queries = np.ascontiguousarray(queries.astype(np.uint8))
        if HAS_FAISS:
            distances, indices = self._index.search(queries, k)
            scores = -distances.astype(np.float32)
        else:
            n_q = queries.shape[0]
            distances = np.zeros((n_q, self.ntotal), dtype=np.int32)
            for i in range(n_q):
                xor = np.bitwise_xor(queries[i], self._corpus)
                distances[i] = np.unpackbits(xor, axis=1).sum(axis=1)
            indices = np.argpartition(distances, k, axis=1)[:, :k]
            for i in range(len(indices)):
                idx = indices[i]
                indices[i] = idx[np.argsort(distances[i, idx])]
            scores = -np.take_along_axis(distances, indices, axis=1).astype(np.float32)
        return scores, indices


class Int4Index:
    """NumPy brute-force search with reconstructed quaternary (int4) vectors."""

    def __init__(self, corpus_emb: np.ndarray):
        self.ntotal = corpus_emb.shape[0]
        self.d = corpus_emb.shape[1]
        self.type = "int4"
        quantizer = QuantizationHandler()
        cal = quantizer.calibrate_quaternary(corpus_emb)
        codes = quantizer.quantize_to_quaternary(corpus_emb, cal["boundaries"])
        self._reconstructed = quantizer.reconstruct_quaternary(codes, cal["centroids"])
        self._calibration = cal

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = min(k, self.ntotal)
        queries = queries.astype(np.float32)
        sims = queries @ self._reconstructed.T
        indices = np.argpartition(-sims, k, axis=1)[:, :k]
        for i in range(len(indices)):
            idx = indices[i]
            indices[i] = idx[np.argsort(-sims[i, idx])]
        scores = np.take_along_axis(sims, indices, axis=1)
        return scores, indices


# ── Index construction ────────────────────────────────────────────────────────

def build_index(corpus_emb: np.ndarray, retrieval_method: str, truncate_dim: int):
    """Build a retrieval index.

    Args:
        corpus_emb: Full float32 corpus embeddings.
        retrieval_method: One of "float32", "binary", "int4".
        truncate_dim: Truncate to this dimension before indexing.

    Returns:
        Index object with .search(queries, k) -> (scores, indices).
    """
    corpus_trunc = truncate(corpus_emb, truncate_dim)

    if retrieval_method == "float32":
        return FloatIndex(corpus_trunc)
    elif retrieval_method == "binary":
        corpus_bin = binarize(corpus_trunc)
        return BinaryIndex(corpus_bin, corpus_trunc.shape[1])
    elif retrieval_method == "int4":
        return Int4Index(corpus_trunc)
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}")


def initial_search(index, query_emb: np.ndarray, k: int, oversample: int,
                   retrieval_method: str, truncate_dim: int):
    """Run initial retrieval.

    Returns:
        (scores, candidate_ids, search_sec)
    """
    q = truncate(query_emb, truncate_dim)
    if retrieval_method == "binary":
        q = binarize(q)

    t0 = time.time()
    scores, candidate_ids = index.search(q, k * oversample)
    search_sec = time.time() - t0

    return scores, candidate_ids, search_sec


def index_memory_bytes(index) -> int:
    """Estimate memory usage of an index in bytes."""
    if index.type == "float32":
        return index.ntotal * index.d * 4
    elif index.type == "binary":
        return index.ntotal * (index.d // 8)
    elif index.type == "int4":
        return index.ntotal * max(1, index.d // 4)
    return 0
