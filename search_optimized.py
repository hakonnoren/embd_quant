"""Optimized kNN search using FAISS when available, NumPy fallback."""

import numpy as np
from typing import Tuple
import time

# Check if FAISS is available
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class BruteForceSearch:
    """Brute-force kNN search with FAISS optimization."""

    @staticmethod
    def search_float(
        query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search using float32 embeddings."""
        start = time.perf_counter()
        k = min(k, corpus_embeddings.shape[0])

        if HAS_FAISS:
            dim = corpus_embeddings.shape[1]
            corpus = np.ascontiguousarray(corpus_embeddings.astype(np.float32))
            queries = np.ascontiguousarray(query_embeddings.astype(np.float32))

            index = faiss.IndexFlatIP(dim)
            index.add(corpus)
            scores, indices = index.search(queries, k)
        else:
            similarities = query_embeddings @ corpus_embeddings.T
            indices = np.argpartition(-similarities, k, axis=1)[:, :k]
            for i in range(len(indices)):
                idx = indices[i]
                indices[i] = idx[np.argsort(-similarities[i, idx])]
            scores = np.take_along_axis(similarities, indices, axis=1)

        latency = time.perf_counter() - start
        return indices, scores, latency

    @staticmethod
    def search_int8(
        query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search using int8 embeddings (converted to float for computation)."""
        q = query_embeddings.astype(np.float32)
        c = corpus_embeddings.astype(np.float32)
        return BruteForceSearch.search_float(q, c, k)

    @staticmethod
    def search_binary(
        query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search using binary embeddings with Hamming distance."""
        start = time.perf_counter()
        k = min(k, corpus_embeddings.shape[0])

        if HAS_FAISS:
            dim_bits = corpus_embeddings.shape[1] * 8
            corpus = np.ascontiguousarray(corpus_embeddings.astype(np.uint8))
            queries = np.ascontiguousarray(query_embeddings.astype(np.uint8))

            index = faiss.IndexBinaryFlat(dim_bits)
            index.add(corpus)
            distances, indices = index.search(queries, k)
            scores = -distances.astype(np.float32)
        else:
            n_queries = query_embeddings.shape[0]
            n_corpus = corpus_embeddings.shape[0]
            distances = np.zeros((n_queries, n_corpus), dtype=np.int32)

            for i in range(n_queries):
                xor_result = np.bitwise_xor(query_embeddings[i], corpus_embeddings)
                distances[i] = np.sum(np.unpackbits(xor_result, axis=1), axis=1)

            indices = np.argpartition(distances, k, axis=1)[:, :k]
            for i in range(len(indices)):
                idx = indices[i]
                indices[i] = idx[np.argsort(distances[i, idx])]
            scores = -np.take_along_axis(distances, indices, axis=1).astype(np.float32)

        latency = time.perf_counter() - start
        return indices, scores, latency

    @staticmethod
    def search_int8_asymmetric(
        float_query: np.ndarray, int8_corpus: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Asymmetric search: float32 queries against int8 corpus."""
        c = int8_corpus.astype(np.float32)
        return BruteForceSearch.search_float(float_query, c, k)

    @staticmethod
    def search_binary_asymmetric(
        float_query: np.ndarray, binary_corpus: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Asymmetric search: float32 queries against binary corpus.

        Unpacks binary to ±1 and computes dot product.
        FAISS IndexBinaryFlat only supports Hamming, so this is pure NumPy.
        """
        start = time.perf_counter()
        k = min(k, binary_corpus.shape[0])

        # Unpack binary corpus to ±1 floats
        unpacked = np.unpackbits(binary_corpus, axis=1).astype(np.float32)
        unpacked = 2.0 * unpacked - 1.0  # {0,1} -> {-1,+1}

        # Truncate unpacked to match query dim (unpackbits may pad)
        query_dim = float_query.shape[1]
        unpacked = unpacked[:, :query_dim]

        similarities = float_query @ unpacked.T
        indices = np.argpartition(-similarities, k, axis=1)[:, :k]
        for i in range(len(indices)):
            idx = indices[i]
            indices[i] = idx[np.argsort(-similarities[i, idx])]
        scores = np.take_along_axis(similarities, indices, axis=1)

        latency = time.perf_counter() - start
        return indices, scores, latency

    @staticmethod
    def search_2bit_asymmetric(
        float_query: np.ndarray, reconstructed_corpus: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Asymmetric search: float32 queries against reconstructed 2-bit corpus.

        The caller is responsible for reconstructing the corpus from codes
        (quaternary centroids or Lloyd-Max levels). This method just does
        the dot-product search on the resulting float32 matrix.
        """
        return BruteForceSearch.search_float(float_query, reconstructed_corpus, k)

    @staticmethod
    def search_binary_with_rescore(
        binary_query: np.ndarray,
        binary_corpus: np.ndarray,
        float_query: np.ndarray,
        float_corpus: np.ndarray,
        k: int,
        rescore_multiplier: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Two-stage: binary retrieval + float32 rescoring."""
        start = time.perf_counter()

        # Stage 1: Binary retrieval for candidates
        candidates_k = min(rescore_multiplier * k, binary_corpus.shape[0])
        candidate_indices, _, _ = BruteForceSearch.search_binary(
            binary_query, binary_corpus, candidates_k
        )

        # Stage 2: Rescore with float32
        n_queries = float_query.shape[0]
        final_indices = np.zeros((n_queries, k), dtype=np.int64)
        final_scores = np.zeros((n_queries, k), dtype=np.float32)

        for i in range(n_queries):
            cand_idx = candidate_indices[i]
            sims = float_query[i] @ float_corpus[cand_idx].T
            top_k = np.argsort(-sims)[:k]
            final_indices[i] = cand_idx[top_k]
            final_scores[i] = sims[top_k]

        latency = time.perf_counter() - start
        return final_indices, final_scores, latency
