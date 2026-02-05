"""Optimized kNN search implementations with FAISS and PyTorch backends.

Usage:
    from search_optimized import OptimizedSearch, benchmark_backends

    # Auto-select best available backend
    searcher = OptimizedSearch(backend="auto")
    indices, scores, latency = searcher.search(queries, corpus, k=10)

    # Or specify backend
    searcher = OptimizedSearch(backend="faiss")
"""

import numpy as np
from typing import Tuple, Optional
import time
import warnings


# ============================================================================
# FAISS Backend
# ============================================================================


def search_faiss(
    queries: np.ndarray, corpus: np.ndarray, k: int, use_gpu: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    FAISS brute-force search (IndexFlatIP for inner product).

    Advantages:
    - Highly optimized BLAS operations
    - SIMD vectorization (AVX2/AVX-512)
    - GPU support
    - Memory-efficient
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu)")

    start = time.perf_counter()

    dim = corpus.shape[1]
    k = min(k, corpus.shape[0])

    # Ensure float32 and contiguous
    corpus = np.ascontiguousarray(corpus.astype(np.float32))
    queries = np.ascontiguousarray(queries.astype(np.float32))

    # Create index (IndexFlatIP = inner product, good for normalized vectors)
    index = faiss.IndexFlatIP(dim)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            warnings.warn(f"GPU not available, falling back to CPU: {e}")

    index.add(corpus)
    scores, indices = index.search(queries, k)

    latency = time.perf_counter() - start
    return indices, scores, latency


def search_faiss_binary(
    queries: np.ndarray, corpus: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    FAISS binary search with Hamming distance.

    Much faster than naive Python implementation:
    - Uses popcount instructions
    - SIMD optimized
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")

    start = time.perf_counter()

    dim_bits = corpus.shape[1] * 8  # packed bytes to bits
    k = min(k, corpus.shape[0])

    # Ensure uint8 and contiguous
    corpus = np.ascontiguousarray(corpus.astype(np.uint8))
    queries = np.ascontiguousarray(queries.astype(np.uint8))

    index = faiss.IndexBinaryFlat(dim_bits)
    index.add(corpus)
    distances, indices = index.search(queries, k)

    # Convert Hamming distances to scores (negate so higher = better)
    scores = -distances.astype(np.float32)

    latency = time.perf_counter() - start
    return indices, scores, latency


# ============================================================================
# PyTorch Backend
# ============================================================================


def search_torch(
    queries: np.ndarray, corpus: np.ndarray, k: int, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    PyTorch backend for GPU-accelerated search.

    Advantages:
    - Easy GPU support (CUDA, MPS)
    - Optimized matmul (cuBLAS on GPU)
    - torch.topk is faster than argpartition
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not installed. Install with: pip install torch")

    start = time.perf_counter()

    k = min(k, corpus.shape[0])

    # Move to device
    queries_t = torch.from_numpy(queries).to(device)
    corpus_t = torch.from_numpy(corpus).to(device)

    # Compute similarities
    similarities = torch.mm(queries_t, corpus_t.T)

    # Get top-k (torch.topk is optimized)
    scores_t, indices_t = torch.topk(similarities, k, dim=1, largest=True, sorted=True)

    # Move back to CPU/numpy
    indices = indices_t.cpu().numpy()
    scores = scores_t.cpu().numpy()

    latency = time.perf_counter() - start
    return indices, scores, latency


# ============================================================================
# Unified Interface
# ============================================================================


class OptimizedSearch:
    """
    Unified interface for optimized kNN search.

    Example:
        searcher = OptimizedSearch(backend="auto")
        indices, scores, latency = searcher.search(queries, corpus, k=10)
    """

    BACKENDS = {
        "faiss": lambda q, c, k: search_faiss(q, c, k, use_gpu=False),
        "faiss_gpu": lambda q, c, k: search_faiss(q, c, k, use_gpu=True),
        "faiss_binary": search_faiss_binary,
        "torch": lambda q, c, k: search_torch(q, c, k, device="cpu"),
        "torch_gpu": lambda q, c, k: search_torch(q, c, k, device="cuda"),
        "torch_mps": lambda q, c, k: search_torch(q, c, k, device="mps"),
    }

    def __init__(self, backend: str = "auto"):
        """
        Initialize with specified backend.

        Args:
            backend: One of 'auto', 'faiss', 'faiss_gpu', 'faiss_binary',
                    'torch', 'torch_gpu', 'torch_mps'
        """
        self.backend_name = backend
        if backend == "auto":
            self.backend_name = self._detect_best_backend()

        if self.backend_name not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(self.BACKENDS.keys())}")

    def _detect_best_backend(self) -> str:
        """Auto-detect best available backend."""
        # Try FAISS GPU first
        try:
            import faiss
            if faiss.get_num_gpus() > 0:
                return "faiss_gpu"
            return "faiss"
        except ImportError:
            pass

        # Try PyTorch GPU
        try:
            import torch
            if torch.cuda.is_available():
                return "torch_gpu"
            if torch.backends.mps.is_available():
                return "torch_mps"
            return "torch"
        except ImportError:
            pass

        raise ImportError("No backend available. Install faiss-cpu or torch.")

    def search(
        self,
        queries: np.ndarray,
        corpus: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform kNN search.

        Args:
            queries: (n_queries, dim) query embeddings
            corpus: (n_corpus, dim) corpus embeddings
            k: Number of nearest neighbors

        Returns:
            indices: (n_queries, k) indices of top-k results
            scores: (n_queries, k) similarity scores
            latency: Search time in seconds
        """
        search_fn = self.BACKENDS[self.backend_name]
        return search_fn(queries, corpus, k)

    @classmethod
    def available_backends(cls) -> list:
        """List available backends (those with dependencies installed)."""
        available = []

        try:
            import faiss
            available.append("faiss")
            available.append("faiss_binary")
            if faiss.get_num_gpus() > 0:
                available.append("faiss_gpu")
        except ImportError:
            pass

        try:
            import torch
            available.append("torch")
            if torch.cuda.is_available():
                available.append("torch_gpu")
            if torch.backends.mps.is_available():
                available.append("torch_mps")
        except ImportError:
            pass

        return available


# ============================================================================
# Benchmark utility
# ============================================================================


def benchmark_backends(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int = 10,
    n_runs: int = 3,
    backends: Optional[list] = None
) -> dict:
    """
    Benchmark available backends.

    Args:
        queries: Query embeddings
        corpus: Corpus embeddings
        k: Number of neighbors
        n_runs: Number of runs per backend
        backends: List of backends to test (default: all available)

    Returns:
        Dict with backend -> {latency_mean, latency_std, throughput}
    """
    if backends is None:
        backends = OptimizedSearch.available_backends()

    results = {}

    print(f"Benchmarking {len(backends)} backends...")
    print(f"Queries: {queries.shape}, Corpus: {corpus.shape}, k={k}")
    print("-" * 60)

    for backend in backends:
        try:
            searcher = OptimizedSearch(backend=backend)
            latencies = []

            for _ in range(n_runs):
                _, _, latency = searcher.search(queries, corpus, k)
                latencies.append(latency)

            mean_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            throughput = len(queries) / mean_latency

            results[backend] = {
                "latency_mean": mean_latency,
                "latency_std": std_latency,
                "throughput_qps": throughput,
            }

            print(f"{backend:20s}: {mean_latency*1000:8.2f}ms ± {std_latency*1000:.2f}ms ({throughput:.0f} qps)")

        except Exception as e:
            print(f"{backend:20s}: FAILED - {e}")
            results[backend] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Quick test
    print("Available backends:", OptimizedSearch.available_backends())

    # Generate test data
    np.random.seed(42)
    n_queries, n_corpus, dim = 100, 10000, 768
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    corpus = np.random.randn(n_corpus, dim).astype(np.float32)

    # Normalize
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

    # Benchmark
    benchmark_backends(queries, corpus, k=10, n_runs=3)
