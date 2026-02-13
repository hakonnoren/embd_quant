"""Rescoring strategies for candidate reranking.

All rescore functions take candidate_ids from initial retrieval and rerank
them using a higher-fidelity scoring method.

Supported methods:
  - float32:       float dot-product rescore
  - int8:          float query × int8 corpus (asymmetric)
  - int4:          quaternary reconstructed dot-product
  - binary:        float query × binary {-1,+1} doc
  - binary_median: median-centered binary (calibrated thresholds)
  - lloyd_max:     Lloyd-Max Gaussian 2-bit quantizer

All functions return (final_indices, rescore_vec_mem_bytes).

Funnel variants add a binary residual pruning stage before the final rescore.
"""

import numpy as np
from typing import Tuple

from quantization import QuantizationHandler
from search import truncate, binarize


# ── Helper ────────────────────────────────────────────────────────────────────

def _topk_per_query(scores: np.ndarray, candidate_ids: np.ndarray, k: int) -> np.ndarray:
    """Select top-k candidates per query based on scores."""
    n_q = scores.shape[0]
    k = min(k, scores.shape[1])
    final = np.zeros((n_q, k), dtype=np.int64)
    top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
    for i in range(n_q):
        idx = top_k_idx[i]
        top_k_idx[i] = idx[np.argsort(-scores[i, idx])]
    final = np.take_along_axis(candidate_ids, top_k_idx, axis=1)
    return final


# ── Direct rescore methods ────────────────────────────────────────────────────

def rescore_float32(
    candidate_ids: np.ndarray,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, int]:
    """Rescore candidates using float32 dot-product."""
    n_q = candidate_ids.shape[0]
    n_cand = candidate_ids.shape[1]
    mem_bytes = n_cand * corpus_emb.shape[1] * 4

    final = np.zeros((n_q, k), dtype=np.int64)
    for i in range(n_q):
        cands = candidate_ids[i]
        scores = query_emb[i] @ corpus_emb[cands].T
        top_k = np.argsort(-scores)[:k]
        final[i] = cands[top_k]
    return final, mem_bytes


def rescore_int8(
    candidate_ids: np.ndarray,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, int]:
    """Rescore with int8 quantized corpus, float32 queries (asymmetric).

    Only the corpus is quantized to int8; queries stay float32 to preserve
    precision. This matches the asymmetric approach used by all rescore methods.
    """
    quantizer = QuantizationHandler()
    corpus_int8 = quantizer.quantize_to_int8(corpus_emb, calibration_embeddings=corpus_emb)

    n_q = candidate_ids.shape[0]
    n_cand = candidate_ids.shape[1]
    mem_bytes = n_cand * corpus_emb.shape[1]  # 1 byte per dim

    final = np.zeros((n_q, k), dtype=np.int64)
    for i in range(n_q):
        cands = candidate_ids[i]
        scores = query_emb[i] @ corpus_int8[cands].astype(np.float32).T
        top_k = np.argsort(-scores)[:k]
        final[i] = cands[top_k]
    return final, mem_bytes


def rescore_int4(
    candidate_ids: np.ndarray,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, int]:
    """Rescore with quaternary (int4) reconstructed embeddings.

    Asymmetric: float32 queries × reconstructed int4 corpus.
    """
    quantizer = QuantizationHandler()
    cal = quantizer.calibrate_quaternary(corpus_emb)
    codes = quantizer.quantize_to_quaternary(corpus_emb, cal["boundaries"])
    reconstructed = quantizer.reconstruct_quaternary(codes, cal["centroids"])

    n_q = candidate_ids.shape[0]
    n_cand = candidate_ids.shape[1]
    mem_bytes = n_cand * max(1, corpus_emb.shape[1] // 4)  # 2 bits per dim

    final = np.zeros((n_q, k), dtype=np.int64)
    for i in range(n_q):
        cands = candidate_ids[i]
        scores = query_emb[i] @ reconstructed[cands].T
        top_k = np.argsort(-scores)[:k]
        final[i] = cands[top_k]
    return final, mem_bytes


def rescore_binary(
    candidate_ids: np.ndarray,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, int]:
    """Rescore with binary: float query × binary {-1,+1} doc."""
    corpus_bin = binarize(corpus_emb)
    n_q = candidate_ids.shape[0]
    n_cand = candidate_ids.shape[1]
    mem_bytes = n_cand * corpus_bin.shape[1]  # packed binary bytes

    query_dim = query_emb.shape[1]
    final = np.zeros((n_q, k), dtype=np.int64)
    for i in range(n_q):
        cands = candidate_ids[i]
        doc_bits = np.unpackbits(corpus_bin[cands], axis=-1).astype(np.float32)
        doc_signs = 2.0 * doc_bits[:, :query_dim] - 1.0  # {0,1} -> {-1,+1}
        scores = query_emb[i] @ doc_signs.T
        top_k = np.argsort(-scores)[:k]
        final[i] = cands[top_k]
    return final, mem_bytes


def rescore_binary_median(
    candidate_ids: np.ndarray,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, int]:
    """Rescore with median-centered binary: center at per-dim median, then binarize.

    Asymmetric: float query × binary {-1,+1} doc (after median centering).
    """
    quantizer = QuantizationHandler()
    thresholds = quantizer.compute_binary_thresholds(corpus_emb)
    centered_corpus = quantizer.center_for_binary(corpus_emb, thresholds)
    centered_query = quantizer.center_for_binary(query_emb, thresholds)
    corpus_bin = binarize(centered_corpus)

    n_q = candidate_ids.shape[0]
    n_cand = candidate_ids.shape[1]
    mem_bytes = n_cand * corpus_bin.shape[1]

    query_dim = centered_query.shape[1]
    final = np.zeros((n_q, k), dtype=np.int64)
    for i in range(n_q):
        cands = candidate_ids[i]
        doc_bits = np.unpackbits(corpus_bin[cands], axis=-1).astype(np.float32)
        doc_signs = 2.0 * doc_bits[:, :query_dim] - 1.0
        scores = centered_query[i] @ doc_signs.T
        top_k = np.argsort(-scores)[:k]
        final[i] = cands[top_k]
    return final, mem_bytes


def rescore_lloyd_max(
    candidate_ids: np.ndarray,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, int]:
    """Rescore with Lloyd-Max Gaussian 2-bit quantizer.

    Asymmetric: float query × reconstructed Lloyd-Max corpus.
    """
    quantizer = QuantizationHandler()
    cal = quantizer.calibrate_lloyd_max(corpus_emb)
    codes = quantizer.quantize_to_lloyd_max(corpus_emb, cal["medians"], cal["stds"])
    reconstructed = quantizer.reconstruct_lloyd_max(codes, cal["stds"])
    # Scale back from standardized space
    reconstructed = reconstructed * cal["stds"]

    n_q = candidate_ids.shape[0]
    n_cand = candidate_ids.shape[1]
    mem_bytes = n_cand * max(1, corpus_emb.shape[1] // 4)  # 2 bits per dim

    final = np.zeros((n_q, k), dtype=np.int64)
    for i in range(n_q):
        cands = candidate_ids[i]
        scores = query_emb[i] @ reconstructed[cands].T
        top_k = np.argsort(-scores)[:k]
        final[i] = cands[top_k]
    return final, mem_bytes


# ── Funnel (binary residual prune + rescore) ──────────────────────────────────

def _binary_residual_prune(
    candidate_ids: np.ndarray,
    hamming_scores: np.ndarray,
    trunc_dim: int,
    residual_corpus_bin: np.ndarray,
    residual_query_bin: np.ndarray,
    residual_dim: int,
    mid_k: int,
) -> Tuple[np.ndarray, int]:
    """Binary residual pruning: combine truncated + residual binary IP.

    Prunes from oversample*k candidates to mid_k candidates.
    Returns (pruned_candidate_ids, stage1_mem_bytes).
    """
    n_q = candidate_ids.shape[0]
    n_cand = candidate_ids.shape[1]
    stage1_bytes = n_cand * residual_corpus_bin.shape[1]

    trunc_ip = trunc_dim - 2 * hamming_scores.astype(np.float32)

    pruned = np.zeros((n_q, mid_k), dtype=np.int64)
    for i in range(n_q):
        cands = candidate_ids[i]
        q_bits = residual_query_bin[i]
        doc_bits = residual_corpus_bin[cands]
        xor = np.bitwise_xor(q_bits, doc_bits)
        hamming_residual = np.unpackbits(xor, axis=-1).sum(axis=-1)
        residual_ip = residual_dim - 2 * hamming_residual.astype(np.float32)
        combined = trunc_ip[i] + residual_ip
        top_mid = np.argsort(-combined)[:mid_k]
        pruned[i] = cands[top_mid]

    return pruned, stage1_bytes


def rescore_funnel(
    candidate_ids: np.ndarray,
    hamming_scores: np.ndarray,
    trunc_dim: int,
    residual_corpus_bin: np.ndarray,
    residual_query_bin: np.ndarray,
    residual_dim: int,
    rescore_method: str,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
    funnel_factor: int = 2,
) -> Tuple[np.ndarray, int]:
    """Two-stage funnel: binary residual prune → rescore with any method.

    Stage 1: Binary residual prune from oversample*k → funnel_factor*k.
    Stage 2: Rescore the pruned set with the specified method.

    If residual data is None (e.g. at full dim), skip stage 1 and just rescore.
    """
    mid_k = k * funnel_factor

    if residual_corpus_bin is not None and residual_query_bin is not None:
        pruned_ids, stage1_bytes = _binary_residual_prune(
            candidate_ids, hamming_scores, trunc_dim,
            residual_corpus_bin, residual_query_bin, residual_dim, mid_k,
        )
    else:
        # No residual (full dim) — just take top mid_k from hamming
        n_q = candidate_ids.shape[0]
        pruned_ids = candidate_ids[:, :mid_k]
        stage1_bytes = 0

    # Stage 2: rescore the pruned set
    final, stage2_bytes = dispatch_rescore(
        rescore_method, pruned_ids, query_emb, corpus_emb, k,
    )
    return final, stage1_bytes + stage2_bytes


# ── Dispatcher ────────────────────────────────────────────────────────────────

RESCORE_FUNCTIONS = {
    "float32": rescore_float32,
    "int8": rescore_int8,
    "int4": rescore_int4,
    "binary": rescore_binary,
    "binary_median": rescore_binary_median,
    "lloyd_max": rescore_lloyd_max,
}


def dispatch_rescore(
    method: str,
    candidate_ids: np.ndarray,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, int]:
    """Dispatch to the appropriate rescore function."""
    if method not in RESCORE_FUNCTIONS:
        raise ValueError(f"Unknown rescore method: {method}. "
                         f"Options: {list(RESCORE_FUNCTIONS.keys())}")
    return RESCORE_FUNCTIONS[method](candidate_ids, query_emb, corpus_emb, k)
