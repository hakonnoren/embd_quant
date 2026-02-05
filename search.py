"""Brute-force kNN search implementations for different precisions."""
import numpy as np
from typing import Tuple
import time


class BruteForceSearch:
    """Brute-force nearest neighbor search for various precision types."""

    @staticmethod
    def search_float(
        query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Search using float32 embeddings with cosine similarity.

        For normalized embeddings, cosine similarity equals dot product.

        Args:
            query_embeddings: (n_queries, dim) query vectors
            corpus_embeddings: (n_corpus, dim) corpus vectors
            k: Number of top results to return

        Returns:
            indices: (n_queries, k) indices of top-k results
            scores: (n_queries, k) similarity scores
            latency: total search time in seconds
        """
        start = time.perf_counter()

        # Dot product for normalized vectors = cosine similarity
        similarities = query_embeddings @ corpus_embeddings.T

        # Get top-k indices
        n_corpus = corpus_embeddings.shape[0]
        k = min(k, n_corpus)

        if k >= n_corpus:
            indices = np.argsort(-similarities, axis=1)[:, :k]
        else:
            # Partial sort is faster for large corpus
            indices = np.argpartition(-similarities, k, axis=1)[:, :k]
            # Sort the top-k
            for i in range(len(indices)):
                top_k_sims = similarities[i, indices[i]]
                sorted_idx = np.argsort(-top_k_sims)
                indices[i] = indices[i, sorted_idx]

        scores = np.array([similarities[i, indices[i]] for i in range(len(indices))])

        latency = time.perf_counter() - start
        return indices, scores, latency

    @staticmethod
    def search_int8(
        query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Search using int8 embeddings with dot product approximation.

        Int8 embeddings maintain relative ordering for dot product.
        """
        start = time.perf_counter()

        # Convert to float for computation, but int8 storage saves memory
        q = query_embeddings.astype(np.float32)
        c = corpus_embeddings.astype(np.float32)

        similarities = q @ c.T

        n_corpus = c.shape[0]
        k = min(k, n_corpus)

        if k >= n_corpus:
            indices = np.argsort(-similarities, axis=1)[:, :k]
        else:
            indices = np.argpartition(-similarities, k, axis=1)[:, :k]
            for i in range(len(indices)):
                top_k_sims = similarities[i, indices[i]]
                sorted_idx = np.argsort(-top_k_sims)
                indices[i] = indices[i, sorted_idx]

        scores = np.array([similarities[i, indices[i]] for i in range(len(indices))])

        latency = time.perf_counter() - start
        return indices, scores, latency

    @staticmethod
    def search_binary(
        query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Search using binary embeddings with Hamming distance.

        Binary embeddings are packed uint8 from ubinary quantization.
        Lower Hamming distance = more similar.

        Args:
            query_embeddings: (n_queries, dim/8) packed binary query vectors
            corpus_embeddings: (n_corpus, dim/8) packed binary corpus vectors
            k: Number of top results to return

        Returns:
            indices: (n_queries, k) indices of top-k results
            scores: (n_queries, k) negative Hamming distances (higher = more similar)
            latency: total search time in seconds
        """
        start = time.perf_counter()

        n_queries = query_embeddings.shape[0]
        n_corpus = corpus_embeddings.shape[0]
        k = min(k, n_corpus)

        # Compute Hamming distances using XOR + popcount
        # For packed uint8: XOR to find differing bits, then count
        distances = np.zeros((n_queries, n_corpus), dtype=np.int32)

        for i in range(n_queries):
            # XOR to find differing bits
            xor_result = np.bitwise_xor(query_embeddings[i], corpus_embeddings)
            # Count set bits using unpackbits
            distances[i] = np.sum(np.unpackbits(xor_result, axis=1), axis=1)

        # Get top-k by minimum distance
        if k >= n_corpus:
            indices = np.argsort(distances, axis=1)[:, :k]
        else:
            indices = np.argpartition(distances, k, axis=1)[:, :k]
            for i in range(len(indices)):
                top_k_dist = distances[i, indices[i]]
                sorted_idx = np.argsort(top_k_dist)
                indices[i] = indices[i, sorted_idx]

        # Convert distances to scores (negate so higher is better)
        scores = np.array(
            [-distances[i, indices[i]].astype(np.float32) for i in range(len(indices))]
        )

        latency = time.perf_counter() - start
        return indices, scores, latency

    @staticmethod
    def search_binary_with_rescore(
        binary_query: np.ndarray,
        binary_corpus: np.ndarray,
        float_query: np.ndarray,
        float_corpus: np.ndarray,
        k: int,
        rescore_multiplier: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Two-stage search: binary retrieval + float32 rescoring.

        1. Retrieve rescore_multiplier * k candidates using binary search
        2. Rescore candidates using float32 query vs float32 corpus

        Args:
            binary_query: Packed binary query embeddings
            binary_corpus: Packed binary corpus embeddings
            float_query: Float32 query embeddings (for rescoring)
            float_corpus: Float32 corpus embeddings (for rescoring)
            k: Number of final results
            rescore_multiplier: How many more candidates to retrieve for rescoring

        Returns:
            indices: (n_queries, k) final top-k indices
            scores: (n_queries, k) float32 similarity scores
            latency: total search time in seconds
        """
        start = time.perf_counter()

        # Stage 1: Binary retrieval for more candidates
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
            cand_embeddings = float_corpus[cand_idx]

            # Compute float similarities for candidates
            sims = float_query[i] @ cand_embeddings.T

            # Get top-k from candidates
            actual_k = min(k, len(sims))
            top_k_local = np.argsort(-sims)[:actual_k]
            final_indices[i, :actual_k] = cand_idx[top_k_local]
            final_scores[i, :actual_k] = sims[top_k_local]

        latency = time.perf_counter() - start
        return final_indices, final_scores, latency
