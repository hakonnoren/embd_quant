"""
New quantization methods to test hypotheses from reasoning_context.md.

Implements:
  Q2 - Per-dimension weighted asymmetric binary search
  Q3 - 2-bit (ternary & quaternary) quantization with asymmetric search
  Q5 - Diagnostic: why median hurts at full dim
  Q6 - Mixed-precision Matryoshka (float32 early dims + binary late dims)
"""

import numpy as np
from typing import Tuple, Optional, Dict
import time


# =============================================================================
# Q2: Per-Dimension Weighted Asymmetric Binary
# =============================================================================

def compute_optimal_weights(corpus_embeddings: np.ndarray, thresholds: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute optimal per-dimension weights for weighted asymmetric binary search.
    
    Theory: In binary_asym, the score is q^T sign(d). The quantization error per dim is:
        error_i = d_i - sign(d_i) * E[|d_i|]
    
    The optimal reconstruction of d_i from sign(d_i) that minimizes E[(d_i - w_i * sign(d_i))^2]
    is w_i = E[|d_i|] (the mean absolute value per dimension).
    
    With median centering: w_i = E[|d_i - median_i|]
    
    This gives: score = q^T (w * sign(d - threshold))
    
    Args:
        corpus_embeddings: (n, dim) corpus vectors
        thresholds: optional per-dim thresholds (e.g. median). If None, uses 0.
    
    Returns:
        weights: (dim,) optimal per-dimension weights
    """
    if thresholds is not None:
        centered = corpus_embeddings - thresholds
    else:
        centered = corpus_embeddings
    
    # w_i = E[|d_i|] — the mean absolute deviation from threshold
    weights = np.mean(np.abs(centered), axis=0).astype(np.float32)
    return weights


def search_binary_weighted_asym(
    float_query: np.ndarray,
    binary_corpus: np.ndarray,
    weights: np.ndarray,
    k: int,
    query_dim: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Weighted asymmetric binary search: score = q^T (w * sign(d))
    
    Instead of unpacking binary to {-1, +1}, we unpack to {-w_i, +w_i}.
    This is the MMSE (minimum mean squared error) reconstruction of d from sign(d).
    
    Args:
        float_query: (n_queries, dim) float32 query embeddings
        binary_corpus: (n_corpus, dim/8) packed binary corpus
        weights: (dim,) per-dimension weights
        k: number of results
        query_dim: actual dimension (unpackbits may pad)
    """
    start = time.perf_counter()
    k = min(k, binary_corpus.shape[0])
    
    if query_dim is None:
        query_dim = float_query.shape[1]
    
    # Unpack binary corpus to {-1, +1}
    unpacked = np.unpackbits(binary_corpus, axis=1)[:, :query_dim].astype(np.float32)
    unpacked = 2.0 * unpacked - 1.0  # {0,1} -> {-1,+1}
    
    # Apply weights: each dim scaled by w_i
    unpacked_weighted = unpacked * weights[:query_dim]
    
    # Score = q^T (w * sign(d))
    similarities = float_query @ unpacked_weighted.T
    
    indices = np.argpartition(-similarities, k, axis=1)[:, :k]
    for i in range(len(indices)):
        idx = indices[i]
        indices[i] = idx[np.argsort(-similarities[i, idx])]
    scores = np.take_along_axis(similarities, indices, axis=1)
    
    latency = time.perf_counter() - start
    return indices, scores, latency


# =============================================================================
# Q3: 2-bit Quantization (Ternary and Quaternary)
# =============================================================================

def quantize_ternary(
    embeddings: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ternary quantization: values -> {-1, 0, +1} based on thresholds.
    
    Theory: The optimal threshold t for ternary quantization of N(0,σ²) data
    minimizes E[(x - Q(x))²]. For Gaussian, this is t ≈ 0.6745σ (the MAD).
    More precisely, for asymmetric search, we want to maximize ranking correlation.
    
    The dead zone [-t, +t] maps to 0, meaning "uncertain" dimensions don't vote.
    This is like a soft attention mask on dimensions.
    
    Storage: 2 bits/dim (log2(3) ≈ 1.58, but we use 2 bits for simplicity).
    Packed as uint8: 4 values per byte.
    
    Args:
        embeddings: (n, dim) vectors
        thresholds: (dim,) centering thresholds (e.g. median). Default 0.
        t: (dim,) or scalar dead-zone half-width. Default: 0.675 * std per dim.
    
    Returns:
        quantized: (n, dim) int8 values in {-1, 0, +1}
        thresholds: (dim,) centering thresholds used
        t: (dim,) dead-zone widths used
    """
    if thresholds is None:
        thresholds = np.zeros(embeddings.shape[1], dtype=np.float32)
    
    centered = embeddings - thresholds
    
    if t is None:
        # Optimal for Gaussian: t = 0.675 * sigma (so P(|x| < t) ≈ 0.5)
        # This means half the dimensions are "dead" (0), which maximizes info
        t = 0.675 * np.std(centered, axis=0)
    
    quantized = np.zeros_like(centered, dtype=np.int8)
    quantized[centered > t] = 1
    quantized[centered < -t] = -1
    # Values in [-t, t] stay 0
    
    return quantized, thresholds, t


def quantize_quaternary(
    embeddings: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quaternary (4-level) quantization: {-1.5, -0.5, +0.5, +1.5} or percentile-based.
    
    Uses quartile boundaries: 25th, 50th, 75th percentile.
    Maps to codes {0, 1, 2, 3} stored as 2 bits.
    Reconstruction values: per-bin centroids (Lloyd-Max style).
    
    Storage: exactly 2 bits/dim.
    
    Args:
        embeddings: (n, dim) vectors
        thresholds: (3, dim) — percentile boundaries [p25, p50, p75]
    
    Returns:
        codes: (n, dim) uint8 codes in {0, 1, 2, 3}
        centroids: (4, dim) reconstruction values per code per dim
    """
    dim = embeddings.shape[1]
    
    if thresholds is None:
        thresholds = np.percentile(embeddings, [25, 50, 75], axis=0)  # (3, dim)
    
    p25, p50, p75 = thresholds[0], thresholds[1], thresholds[2]
    
    codes = np.zeros_like(embeddings, dtype=np.uint8)
    codes[embeddings >= p25] = 1
    codes[embeddings >= p50] = 2
    codes[embeddings >= p75] = 3
    
    # Compute centroids per bin per dim (Lloyd-Max style)
    centroids = np.zeros((4, dim), dtype=np.float32)
    for c in range(4):
        mask = codes == c
        for d in range(dim):
            vals = embeddings[mask[:, d], d]
            if len(vals) > 0:
                centroids[c, d] = vals.mean()
            else:
                # Fallback: evenly spaced
                centroids[c, d] = (c - 1.5)
    
    return codes, centroids


def pack_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack 2-bit codes into uint8 (4 values per byte).
    
    Args:
        codes: (n, dim) uint8 codes in {0, 1, 2, 3}
    
    Returns:
        packed: (n, ceil(dim/4)) uint8 array
    """
    n, dim = codes.shape
    # Pad dim to multiple of 4
    pad = (4 - dim % 4) % 4
    if pad > 0:
        codes = np.pad(codes, ((0, 0), (0, pad)), constant_values=0)
    
    packed_dim = codes.shape[1] // 4
    packed = np.zeros((n, packed_dim), dtype=np.uint8)
    
    for i in range(4):
        packed |= (codes[:, i::4].astype(np.uint8) << (6 - 2 * i))
    
    return packed


def unpack_2bit(packed: np.ndarray, original_dim: int) -> np.ndarray:
    """Unpack 2-bit codes from uint8.
    
    Args:
        packed: (n, packed_dim) uint8 array
        original_dim: original dimension before packing
    
    Returns:
        codes: (n, original_dim) uint8 codes in {0, 1, 2, 3}
    """
    n = packed.shape[0]
    padded_dim = packed.shape[1] * 4
    codes = np.zeros((n, padded_dim), dtype=np.uint8)
    
    for i in range(4):
        codes[:, i::4] = (packed >> (6 - 2 * i)) & 0x03
    
    return codes[:, :original_dim]


def search_ternary_asym(
    float_query: np.ndarray,
    ternary_corpus: np.ndarray,
    k: int,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Asymmetric search: float32 query vs ternary corpus.
    
    Score = q^T (w * ternary(d))  where ternary ∈ {-1, 0, +1}
    
    The key insight: dimensions quantized to 0 contribute NOTHING to the score.
    This is like automatic feature selection — uncertain dimensions are masked.
    
    Args:
        float_query: (n_queries, dim) float32
        ternary_corpus: (n_corpus, dim) int8 in {-1, 0, +1}
        k: number of results
        weights: (dim,) optional per-dim weights
    """
    start = time.perf_counter()
    k = min(k, ternary_corpus.shape[0])
    
    corpus_float = ternary_corpus.astype(np.float32)
    if weights is not None:
        corpus_float *= weights
    
    similarities = float_query @ corpus_float.T
    
    indices = np.argpartition(-similarities, k, axis=1)[:, :k]
    for i in range(len(indices)):
        idx = indices[i]
        indices[i] = idx[np.argsort(-similarities[i, idx])]
    scores = np.take_along_axis(similarities, indices, axis=1)
    
    latency = time.perf_counter() - start
    return indices, scores, latency


def search_quaternary_asym(
    float_query: np.ndarray,
    corpus_codes: np.ndarray,
    centroids: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Asymmetric search: float32 query vs quaternary corpus.
    
    Reconstruct corpus from codes + centroids, then dot product with float query.
    Score = q^T reconstruct(codes)  where reconstruct maps {0,1,2,3} -> centroids.
    
    Args:
        float_query: (n_queries, dim) float32
        corpus_codes: (n_corpus, dim) uint8 codes in {0, 1, 2, 3}
        centroids: (4, dim) reconstruction values
        k: number of results
    """
    start = time.perf_counter()
    k = min(k, corpus_codes.shape[0])
    
    # Reconstruct corpus from codes (vectorized lookup)
    # centroids[codes[i,j], j] for each (i, j)
    dim = float_query.shape[1]
    reconstructed = centroids[corpus_codes, np.arange(dim)]  # (n_corpus, dim)
    
    similarities = float_query @ reconstructed.T
    
    indices = np.argpartition(-similarities, k, axis=1)[:, :k]
    for i in range(len(indices)):
        idx = indices[i]
        indices[i] = idx[np.argsort(-similarities[i, idx])]
    scores = np.take_along_axis(similarities, indices, axis=1)
    
    latency = time.perf_counter() - start
    return indices, scores, latency


# =============================================================================
# Q5: Diagnostics — Why median hurts at full dim
# =============================================================================

def diagnose_median_effect(corpus_embeddings: np.ndarray, label: str = "") -> Dict:
    """
    Analyze per-dimension statistics to understand when median centering helps.
    
    Hypothesis: At full dim, L2-normalized embeddings are already nearly zero-centered
    (by the geometry of high-dimensional spheres). Median ≈ mean ≈ 0, so centering
    adds noise without shifting thresholds meaningfully. At truncated dims, the
    remaining dimensions have stronger bias (non-zero mean) because Matryoshka training
    makes early dimensions more "opinionated".
    
    Args:
        corpus_embeddings: (n, dim) normalized embeddings
        label: descriptive label
    
    Returns:
        Dictionary of diagnostic statistics
    """
    n, dim = corpus_embeddings.shape
    
    means = np.mean(corpus_embeddings, axis=0)
    medians = np.median(corpus_embeddings, axis=0)
    stds = np.std(corpus_embeddings, axis=0)
    
    # How far is median from zero (relative to std)?
    median_z_scores = np.abs(medians) / (stds + 1e-10)
    
    # How balanced is the sign split at threshold=0?
    frac_positive_at_zero = np.mean(corpus_embeddings > 0, axis=0)
    sign_imbalance_at_zero = np.abs(frac_positive_at_zero - 0.5)
    
    # Same at median threshold
    centered = corpus_embeddings - medians
    frac_positive_at_median = np.mean(centered > 0, axis=0)
    sign_imbalance_at_median = np.abs(frac_positive_at_median - 0.5)
    
    # Entropy of binary codes (bits per dim)
    def binary_entropy(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    entropy_at_zero = binary_entropy(frac_positive_at_zero)
    entropy_at_median = binary_entropy(frac_positive_at_median)
    
    # Median estimation noise: bootstrap std of median
    rng = np.random.default_rng(42)
    n_bootstrap = 50
    bootstrap_medians = np.zeros((n_bootstrap, dim))
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bootstrap_medians[b] = np.median(corpus_embeddings[idx], axis=0)
    median_noise = np.std(bootstrap_medians, axis=0)
    
    stats = {
        "label": label,
        "dim": dim,
        "n_vectors": n,
        "mean_abs_mean": float(np.mean(np.abs(means))),
        "mean_abs_median": float(np.mean(np.abs(medians))),
        "mean_std": float(np.mean(stds)),
        "mean_median_z_score": float(np.mean(median_z_scores)),
        "mean_sign_imbalance_at_zero": float(np.mean(sign_imbalance_at_zero)),
        "mean_sign_imbalance_at_median": float(np.mean(sign_imbalance_at_median)),
        "mean_entropy_at_zero": float(np.mean(entropy_at_zero)),
        "mean_entropy_at_median": float(np.mean(entropy_at_median)),
        "mean_median_noise": float(np.mean(median_noise)),
        "median_noise_vs_shift": float(np.mean(median_noise / (np.abs(medians) + 1e-10))),
        # Per-dim arrays for plotting
        "_medians": medians,
        "_stds": stds,
        "_sign_imbalance_zero": sign_imbalance_at_zero,
        "_sign_imbalance_median": sign_imbalance_at_median,
        "_entropy_zero": entropy_at_zero,
        "_entropy_median": entropy_at_median,
        "_median_noise": median_noise,
    }
    
    return stats


# =============================================================================
# Q6: Mixed-Precision Matryoshka
# =============================================================================

def search_mixed_precision_matryoshka(
    float_query: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int,
    float_dims: int = 64,
    binary_dims: int = 448,
    float_weight: float = 1.0,
    binary_weight: float = 1.0,
    use_median: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Mixed-precision Matryoshka: float32 for first dims, binary for rest.
    
    Theory: In Matryoshka models, dim 1-64 carry ~60% of the signal (measured by
    variance contribution). Keeping them at float32 preserves this signal perfectly.
    The remaining dims carry less info per dim, so binary quantization loses less.
    
    Total storage: float_dims * 4 + binary_dims / 8 bytes per vector.
    E.g., 64 * 4 + 448 / 8 = 256 + 56 = 312 bytes for 512d 
    vs 512 * 4 = 2048 bytes float32 (6.6x compression)
    vs 512 / 8 = 64 bytes pure binary (4.9x bigger than binary)
    
    Score = float_weight * (q_float^T d_float) + binary_weight * (q_binary^T sign(d_binary))
    
    Args:
        float_query: (n_queries, full_dim) float32 queries
        corpus_embeddings: (n_corpus, full_dim) float32 corpus
        k: number of results
        float_dims: number of leading dims to keep as float32
        binary_dims: number of trailing dims to binarize
        float_weight: weight for float32 component
        binary_weight: weight for binary component  
        use_median: whether to use median thresholds for binary part
    
    Returns:
        indices, scores, latency, memory_info
    """
    start = time.perf_counter()
    
    full_dim = float_query.shape[1]
    total_dims = float_dims + binary_dims
    assert total_dims <= full_dim, f"float_dims + binary_dims = {total_dims} > full_dim = {full_dim}"
    
    n_corpus = corpus_embeddings.shape[0]
    k = min(k, n_corpus)
    
    # Split embeddings
    q_float_part = float_query[:, :float_dims]
    c_float_part = corpus_embeddings[:, :float_dims]
    
    q_binary_part = float_query[:, float_dims:total_dims]
    c_binary_part = corpus_embeddings[:, float_dims:total_dims]
    
    # Float32 component: standard dot product
    float_scores = q_float_part @ c_float_part.T  # (n_queries, n_corpus)
    
    # Binary component: asymmetric (float query, binary corpus)
    if use_median:
        med = np.median(c_binary_part, axis=0)
        c_binary_centered = c_binary_part - med
        q_binary_centered = q_binary_part - med
    else:
        c_binary_centered = c_binary_part
        q_binary_centered = q_binary_part
    
    binary_corpus_packed = np.packbits((c_binary_centered > 0).astype(np.uint8), axis=1)
    unpacked = np.unpackbits(binary_corpus_packed, axis=1)[:, :binary_dims].astype(np.float32)
    unpacked = 2.0 * unpacked - 1.0
    
    binary_scores = q_binary_centered @ unpacked.T  # (n_queries, n_corpus)
    
    # Combine scores with weights
    # Normalize by sqrt of dimension to make components comparable
    combined = (float_weight * float_scores / np.sqrt(float_dims) + 
                binary_weight * binary_scores / np.sqrt(binary_dims))
    
    indices = np.argpartition(-combined, k, axis=1)[:, :k]
    for i in range(len(indices)):
        idx = indices[i]
        indices[i] = idx[np.argsort(-combined[i, idx])]
    scores = np.take_along_axis(combined, indices, axis=1)
    
    # Memory calculation
    float_mem = n_corpus * float_dims * 4
    binary_mem = n_corpus * (binary_dims // 8)
    total_mem = float_mem + binary_mem
    
    memory_info = {
        "float_bytes": float_mem,
        "binary_bytes": binary_mem,
        "total_bytes": total_mem,
        "bytes_per_vector": total_mem / n_corpus,
        "compression_vs_float32": (n_corpus * full_dim * 4) / total_mem,
    }
    
    latency = time.perf_counter() - start
    return indices, scores, latency, memory_info


# =============================================================================
# Q4: Optimal thresholds via ranking correlation maximization
# =============================================================================

def find_optimal_thresholds_per_dim(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    n_candidates: int = 20,
) -> np.ndarray:
    """
    Find per-dim thresholds that maximize ranking correlation in asymmetric binary search.
    
    For each dimension i, try n_candidates threshold values and pick the one that 
    maximizes the Spearman correlation between binary_asym scores and float32 scores
    on a validation sample.
    
    This is the "oracle" approach — in practice you'd use the corpus only, but here
    we evaluate against true similarities to verify whether median is near-optimal.
    
    Approach: 
    - For each dim, test thresholds at percentiles 5, 10, ..., 95
    - For each threshold vector, compute binary_asym scores on validation pairs
    - Pick threshold with highest rank correlation to float32 scores
    
    This is expensive, so we do it on a subsample.
    
    Args:
        corpus_embeddings: (n_corpus, dim) float32
        query_embeddings: (n_queries, dim) float32
        n_candidates: number of threshold candidates per dim
        
    Returns:
        optimal_thresholds: (dim,) per-dim thresholds
    """
    from scipy.stats import spearmanr
    
    n_corpus, dim = corpus_embeddings.shape
    n_queries = query_embeddings.shape[0]
    
    # Subsample for speed
    max_corpus = min(500, n_corpus)
    max_queries = min(50, n_queries)
    rng = np.random.default_rng(42)
    c_idx = rng.choice(n_corpus, max_corpus, replace=False)
    q_idx = rng.choice(n_queries, max_queries, replace=False)
    
    corpus_sub = corpus_embeddings[c_idx]
    query_sub = query_embeddings[q_idx]
    
    # True similarities
    true_sims = query_sub @ corpus_sub.T  # (max_queries, max_corpus)
    true_flat = true_sims.flatten()
    
    # Test percentile thresholds
    percentiles = np.linspace(5, 95, n_candidates)
    candidate_thresholds = np.percentile(corpus_sub, percentiles, axis=0)  # (n_candidates, dim)
    
    # Start with median and optimize greedily per dimension
    best_thresholds = np.median(corpus_sub, axis=0)
    
    # Evaluate median baseline
    centered_c = corpus_sub - best_thresholds
    centered_q = query_sub - best_thresholds
    binary_c = np.packbits((centered_c > 0).astype(np.uint8), axis=1)
    unpacked = np.unpackbits(binary_c, axis=1)[:, :dim].astype(np.float32)
    unpacked = 2.0 * unpacked - 1.0
    binary_sims = centered_q @ unpacked.T
    base_corr, _ = spearmanr(true_flat, binary_sims.flatten())
    
    print(f"  Median baseline Spearman correlation: {base_corr:.6f}")
    
    # Greedy per-dim optimization (coordinate descent, one pass)
    for d_idx in range(dim):
        best_corr = -1
        best_t = best_thresholds[d_idx]
        
        for t_val in candidate_thresholds[:, d_idx]:
            test_thresholds = best_thresholds.copy()
            test_thresholds[d_idx] = t_val
            
            centered_c = corpus_sub - test_thresholds
            centered_q = query_sub - test_thresholds
            binary_c = np.packbits((centered_c > 0).astype(np.uint8), axis=1)
            unpacked = np.unpackbits(binary_c, axis=1)[:, :dim].astype(np.float32)
            unpacked = 2.0 * unpacked - 1.0
            binary_sims = centered_q @ unpacked.T
            
            corr, _ = spearmanr(true_flat, binary_sims.flatten())
            if corr > best_corr:
                best_corr = corr
                best_t = t_val
        
        best_thresholds[d_idx] = best_t
    
    # Final correlation
    centered_c = corpus_sub - best_thresholds
    centered_q = query_sub - best_thresholds
    binary_c = np.packbits((centered_c > 0).astype(np.uint8), axis=1)
    unpacked = np.unpackbits(binary_c, axis=1)[:, :dim].astype(np.float32)
    unpacked = 2.0 * unpacked - 1.0
    binary_sims = centered_q @ unpacked.T
    final_corr, _ = spearmanr(true_flat, binary_sims.flatten())
    print(f"  Optimized Spearman correlation: {final_corr:.6f} (delta: {final_corr - base_corr:+.6f})")
    
    return best_thresholds
