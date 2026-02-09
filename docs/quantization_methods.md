# Asymmetric Quantization Methods for Embedding Search

A practical guide to the quantization methods evaluated in this project. All methods follow the **asymmetric** paradigm: queries stay in float32, only corpus vectors are quantized. This maximizes retrieval quality because the few queries are cheap to keep at full precision while the millions of corpus vectors dominate storage.

Every method described here is **streaming-friendly**: calibration statistics are computed once from a representative corpus sample, and each new document can then be quantized independently without access to other documents.

---

## Table of Contents

1. [Shared Concepts](#shared-concepts)
2. [Binary Asymmetric (1-bit)](#1-binary-asymmetric-1-bit)
3. [Binary Median Asymmetric (1-bit)](#2-binary-median-asymmetric-1-bit)
4. [Quaternary Asymmetric (2-bit)](#3-quaternary-asymmetric-2-bit)
5. [Lloyd-Max Gaussian (2-bit)](#4-lloyd-max-gaussian-2-bit--best-2-bit-method)
6. [Int8 Asymmetric (8-bit)](#5-int8-asymmetric-8-bit)
7. [Matryoshka Truncation](#6-matryoshka-truncation)
8. [Storage and Compression Summary](#storage-and-compression-summary)
9. [Which Method to Use](#which-method-to-use)

---

## Shared Concepts

### Asymmetric scoring

All methods share the same scoring pattern:

```
score(q, d) = q_float32 · reconstruct(d_quantized)
```

The query vector `q` is never quantized. Only the corpus document `d` is compressed. At search time, we reconstruct an approximation of `d` from its compressed representation and compute an inner product with the float query. This is "asymmetric" because the two sides of the dot product are at different precisions.

### Calibration vs indexing vs search

Every method has three phases:

| Phase | When | What | Stored |
|-------|------|------|--------|
| **Calibrate** | Once, on representative corpus | Compute per-dimension statistics (medians, stds, thresholds, etc.) | Small metadata vector(s) of size `d` |
| **Index** | Per document, streaming | Quantize document using calibration stats | Compressed codes per doc |
| **Search** | Per query | Score float query against quantized corpus | Nothing new |

### Matryoshka compatibility

For models that support Matryoshka embeddings (mxbai, nomic), you can truncate to the first `k` dimensions before quantization. This stacks multiplicatively with bit-width compression:

```
total_compression = (original_dims / truncated_dims) × (32 / bits_per_dim)
```

For example: 1024d → 256d with 2-bit = 4× from truncation × 16× from quantization = **64× total compression**.

---

## 1. Binary Asymmetric (1-bit)

The simplest method. Each dimension is reduced to a single sign bit.

### How it works

**Calibrate:** Nothing needed.

**Index:** For each document vector `d`, store `sign(d_i)` packed into bits:

```python
def index_binary(corpus_emb):
    """Pack sign bits into uint8 arrays."""
    return np.packbits((corpus_emb > 0).astype(np.uint8), axis=1)
```

**Search:** Unpack to {−1, +1} and compute a standard dot product:

```python
def search_binary_asym(query_emb, binary_corpus, dim, k=10):
    """Asymmetric binary search: q^T sign(d)."""
    unpacked = np.unpackbits(binary_corpus, axis=1)[:, :dim].astype(np.float32)
    signs = 2.0 * unpacked - 1.0  # {0,1} → {-1,+1}
    scores = query_emb @ signs.T
    return top_k(scores, k)
```

### Score interpretation

The score is:

$$s(q, d) = \sum_{i=1}^{d} q_i \cdot \text{sign}(d_i)$$

This is equivalent to: "sum the query dimensions where the document is positive, minus those where it's negative." Large positive query dimensions aligned with positive document dimensions boost the score.

### Storage

| Dimensions | Bytes/vector | Compression vs float32 |
|-----------|-------------|----------------------|
| 1024 | 128 | 32× |
| 256 | 32 | 128× |
| 64 | 8 | 512× |

### When to use

Simplest baseline. Use `binary_med_asym` instead in almost all cases — it's strictly better at truncated dimensions.

---

## 2. Binary Median Asymmetric (1-bit)

Shifts the binarization threshold from zero to the corpus median per dimension. This maximizes per-bit entropy by ensuring each bit is equally likely to be 0 or 1.

### How it works

**Calibrate:** Compute per-dimension medians from the corpus:

```python
def calibrate_binary_median(corpus_emb):
    """One-pass calibration: compute per-dim medians."""
    medians = np.median(corpus_emb, axis=0)  # shape: (d,)
    return medians
```

**Index:** Subtract medians, then binarize:

```python
def index_binary_median(corpus_emb, medians):
    """Binarize centered embeddings."""
    centered = corpus_emb - medians
    return np.packbits((centered > 0).astype(np.uint8), axis=1)
```

**Search:** Center the query by the same medians, then score against the binary corpus:

```python
def search_binary_median_asym(query_emb, binary_corpus, medians, dim, k=10):
    """Score: (q - m)^T sign(d - m)."""
    q_centered = query_emb - medians
    unpacked = np.unpackbits(binary_corpus, axis=1)[:, :dim].astype(np.float32)
    signs = 2.0 * unpacked - 1.0
    scores = q_centered @ signs.T
    return top_k(scores, k)
```

### Score interpretation

$$s(q, d) = (q - m)^T \text{sign}(d - m) = \sum_{i=1}^{d} (q_i - m_i) \cdot \text{sign}(d_i - m_i)$$

Centering both sides by the corpus median ensures:
- Each bit carries maximum information (balanced 0/1 distribution)
- Dimensions where the corpus has a systematic offset (common in Matryoshka-truncated embeddings) are properly handled

### Why center the query too?

It may seem like centering only the document should suffice (since $q^T m$ is constant across documents). But centering the query is critical: the term $-m^T \text{sign}(d - m)$ is document-dependent and carries useful ranking information. It penalizes documents whose deviation pattern from the corpus center doesn't match what the query needs. Our experiments confirmed this — removing query centering degrades NDCG@10 by 0.01–0.24 depending on dimensionality.

### Calibration metadata

Only a single float32 vector of medians: **`d × 4` bytes** (e.g. 4 KB for 1024 dims).

### When to use

**Best 1-bit method overall.** Dominates the Pareto front at low byte budgets (8–64 bytes/vector). Use when storage is severely constrained.

---

## 3. Quaternary Asymmetric (2-bit)

Assigns each dimension to one of 4 levels based on quartile boundaries (25th, 50th, 75th percentile). Each code maps to the mean of its quartile bucket (the centroid), which is the MSE-optimal reconstruction for that bucket.

### How it works

**Calibrate:** Compute quartile boundaries and centroids:

```python
def calibrate_quaternary(corpus_emb):
    """Compute quartile boundaries and bucket centroids per dimension."""
    boundaries = np.percentile(corpus_emb, [25, 50, 75], axis=0)  # (3, d)

    # Assign every value to a bucket
    codes = np.zeros_like(corpus_emb, dtype=np.uint8)
    codes[corpus_emb >= boundaries[0]] = 1
    codes[corpus_emb >= boundaries[1]] = 2
    codes[corpus_emb >= boundaries[2]] = 3

    # Centroid = mean value within each bucket, per dimension
    dim = corpus_emb.shape[1]
    centroids = np.zeros((4, dim), dtype=np.float32)
    for c in range(4):
        mask = (codes == c)
        for j in range(dim):
            centroids[c, j] = corpus_emb[mask[:, j], j].mean()

    return boundaries, centroids
```

**Index:** Quantize each document to 2-bit codes:

```python
def index_quaternary(doc_emb, boundaries):
    """Quantize a single document to 2-bit codes."""
    codes = np.zeros(doc_emb.shape[-1], dtype=np.uint8)
    codes[doc_emb >= boundaries[0]] = 1
    codes[doc_emb >= boundaries[1]] = 2
    codes[doc_emb >= boundaries[2]] = 3
    return codes  # pack into d/4 bytes with np.packbits or bitwise ops
```

**Search:** Reconstruct via centroid lookup, then dot product:

```python
def search_quaternary_asym(query_emb, all_codes, centroids, k=10):
    """Reconstruct corpus from codes and score with float query."""
    dim = centroids.shape[1]
    # Vectorized reconstruction: centroids[code_value, dim_index]
    reconstructed = centroids[all_codes, np.arange(dim)]  # (n_docs, dim)
    scores = query_emb @ reconstructed.T
    return top_k(scores, k)
```

### Score interpretation

$$s(q, d) = \sum_{i=1}^{d} q_i \cdot c_{i, \text{code}_i(d)}$$

where $c_{i,k}$ is the centroid for bucket $k$ in dimension $i$. This is a lookup-table based inner product — identical in structure to how Product Quantization scores work, but with per-scalar (not per-subspace) codebooks.

### Calibration metadata

- 3 boundary values per dimension: `3 × d × 4` bytes
- 4 centroid values per dimension: `4 × d × 4` bytes
- Total: **`28d` bytes** (e.g. 28 KB for 1024 dims)

### When to use

A solid 2-bit method. In our experiments, Lloyd-Max Gaussian (below) consistently outperforms it, so prefer that instead.

---

## 4. Lloyd-Max Gaussian (2-bit) — Best 2-bit Method

Uses the theoretically optimal 4-level scalar quantizer for Gaussian data. Instead of placing boundaries at data quartiles (which splits the data into equal-count buckets), it places them to **minimize mean squared reconstruction error** under a Gaussian assumption.

### The key insight

For a standard normal $z \sim \mathcal{N}(0, 1)$, the MSE-optimal 4-level quantizer has:

| | Boundary | Reconstruction level |
|---|---|---|
| Bin 0: $z < -0.9816$ | −0.9816 | −1.5104 |
| Bin 1: $-0.9816 \le z < 0$ | 0 | −0.4528 |
| Bin 2: $0 \le z < 0.9816$ | +0.9816 | +0.4528 |
| Bin 3: $z \ge 0.9816$ | | +1.5104 |

These are fixed constants — no iterative optimization needed.

For real embeddings, we standardize per dimension ($z_i = (d_i - m_i) / \sigma_i$), apply the fixed quantizer, and reconstruct in the standardized space.

### How it works

**Calibrate:** Compute per-dimension median and standard deviation:

```python
def calibrate_lloyd_max(corpus_emb):
    """One-pass calibration: per-dim median and std."""
    medians = np.median(corpus_emb, axis=0)  # (d,)
    stds = np.std(corpus_emb, axis=0)        # (d,)
    return medians, stds
```

**Index:** Standardize, then quantize to 2-bit codes:

```python
# Fixed constants (Gaussian-optimal)
LLOYD_BOUNDS = np.array([-0.9816, 0.0, 0.9816])
LLOYD_LEVELS = np.array([-1.5104, -0.4528, 0.4528, 1.5104], dtype=np.float32)

def index_lloyd_max(doc_emb, medians, stds):
    """Quantize a document to 2-bit Lloyd-Max codes."""
    z = (doc_emb - medians) / np.clip(stds, 1e-10, None)
    codes = np.digitize(z, LLOYD_BOUNDS)  # → {0, 1, 2, 3}
    return codes.astype(np.uint8)
```

**Search — the efficient scoring trick:**

Since reconstruction in original space is $\hat{d}_i = m_i + \sigma_i \cdot L[\text{code}_i]$, the score becomes:

$$s(q, d) = q^T \hat{d} = \underbrace{q^T m}_{\text{constant for query}} + \sum_i (q_i \sigma_i) \cdot L[\text{code}_i]$$

The first term is identical for all documents and doesn't affect ranking. So we only compute:

$$\text{ranking\_score}(q, d) = \sum_i q'_i \cdot L[\text{code}_i] \quad \text{where } q'_i = q_i \cdot \sigma_i$$

This means: **transform the query once** (element-wise multiply by σ), then score against the fixed reconstruction levels. No per-document denormalization needed.

```python
def search_lloyd_max(query_emb, all_codes, medians, stds, k=10):
    """
    Efficient Lloyd-Max search.
    
    1. Fold per-dim scales into the query (once per query)
    2. Reconstruct codes to fixed levels
    3. Matrix multiply
    """
    # Step 1: scale-adjusted query (d,)
    q_scaled = query_emb * stds    # q'_i = q_i · σ_i
    
    # Step 2: reconstruct all docs in standardized space
    recon_z = LLOYD_LEVELS[all_codes]  # (n_docs, dim)
    
    # Step 3: dot product
    scores = q_scaled @ recon_z.T
    return top_k(scores, k)
```

### Why this beats quartile-based quantization

| Aspect | Quartile (our quaternary) | Lloyd-Max |
|---|---|---|
| Boundary placement | Equal-count (25/50/75 percentile) | MSE-optimal |
| Reconstruction levels | Bucket means (data-driven) | Fixed Gaussian-optimal constants |
| Tail handling | Same bucket size everywhere | Wider outer buckets capture tails better |
| Theory | Heuristic | Provably MSE-optimal for Gaussian |

The improvement is largest at truncated dimensions where the embedding distribution is most Gaussian-like.

### Calibration metadata

Only 2 float32 vectors: **`2 × d × 4 = 8d` bytes** (e.g. 8 KB for 1024 dims). Simpler than quaternary (which needs boundaries + centroids).

### Empirical results (NDCG@10, mxbai/scifact)

| Method | d=1024 | d=256 | d=128 | d=64 |
|---|---|---|---|---|
| binary_med_asym (1b) | 0.726 | 0.653 | 0.569 | 0.454 |
| quaternary_asym (2b) | 0.741 | 0.664 | 0.602 | 0.477 |
| **lloyd_max_gauss (2b)** | **0.747** | **0.686** | **0.631** | **0.505** |
| float32 (32b) | 0.739 | 0.693 | 0.671 | 0.599 |

Lloyd-Max exceeds float32 at full dimensionality (a slight regularization effect) and consistently closes more of the gap at every truncation level.

### When to use

**Recommended default for 2-bit quantization.** Strictly better than quaternary, simpler calibration, and the Gaussian assumption holds well for L2-normalized embeddings.

---

## 5. Int8 Asymmetric (8-bit)

Standard linear quantization to 8-bit integers. Maps the per-dimension range [min, max] to [0, 255].

### How it works

**Calibrate:** Compute per-dimension min and max (or use range from `sentence-transformers`):

```python
def calibrate_int8(corpus_emb):
    """Per-dim range for linear quantization."""
    mins = corpus_emb.min(axis=0)
    maxs = corpus_emb.max(axis=0)
    ranges = (maxs - mins).clip(1e-10)
    return mins, ranges

def index_int8(doc_emb, mins, ranges):
    """Linear quantize to uint8."""
    normalized = (doc_emb - mins) / ranges  # [0, 1]
    return (normalized * 255).clip(0, 255).astype(np.uint8)
```

**Search:** The simplest approach uses the raw int8 values directly:

```python
def search_int8_asym(query_emb, int8_corpus, k=10):
    """Float query × int8 corpus."""
    scores = query_emb @ int8_corpus.astype(np.float32).T
    return top_k(scores, k)
```

Note: since the linear transform $\hat{d}_i = \text{min}_i + (\text{range}_i / 255) \cdot c_i$ is monotonic across documents only when you properly account for the offsets, a more principled approach folds `min` and `range` into the query (similar to the Lloyd-Max trick). In practice, the ranking-only version works well enough.

### Storage

| Dimensions | Bytes/vector | Compression vs float32 |
|-----------|-------------|----------------------|
| 1024 | 1024 | 4× |
| 256 | 256 | 16× |

### When to use

When you need high fidelity and can afford 4× compression. Note that our 2-bit Lloyd-Max often approaches or matches int8 quality at 4× less storage — so int8 is becoming harder to justify except when absolute maximum quality is needed.

---

## 6. Matryoshka Truncation

Not a quantization method per se, but a dimension-reduction technique that stacks with any of the above.

Models trained with Matryoshka Representation Learning (MRL) produce embeddings where the first $k$ dimensions form a valid lower-dimensional embedding. You simply slice and re-normalize:

```python
def truncate_and_normalize(emb, dim):
    """Matryoshka truncation + L2 renormalization."""
    truncated = emb[:, :dim].copy()
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / np.where(norms == 0, 1.0, norms)
```

This is applied **before** quantization. The pipeline becomes:

```
embed → truncate to k dims → L2-normalize → quantize (binary/2-bit/int8)
```

### Available dimensions (from our models)

| Model | Full dim | Matryoshka dims |
|-------|---------|----------------|
| mxbai-embed-large-v1 | 1024 | 512, 256, 128, 64 |
| nomic-embed-text-v1.5 | 768 | 512, 256, 128, 64 |
| all-MiniLM-L6-v2 | 384 | *(not supported)* |

---

## Storage and Compression Summary

Bytes per vector for a 1024-dimensional embedding:

| Method | bits/dim | d=1024 | d=512 | d=256 | d=128 | d=64 |
|--------|---------|--------|-------|-------|-------|------|
| float32 | 32 | 4096 B | 2048 B | 1024 B | 512 B | 256 B |
| int8 | 8 | 1024 B | 512 B | 256 B | 128 B | 64 B |
| lloyd_max / quaternary | 2 | 256 B | 128 B | 64 B | 32 B | 16 B |
| binary (any variant) | 1 | 128 B | 64 B | 32 B | 16 B | 8 B |

Compression ratio relative to float32 at full dim (4096 bytes):

| Method + dim | Bytes | Compression | NDCG@10* |
|---|---|---|---|
| binary_med_asym d=64 | 8 | **512×** | 0.454 (61%) |
| binary_med_asym d=128 | 16 | 256× | 0.569 (77%) |
| binary_med_asym d=256 | 32 | 128× | 0.653 (88%) |
| lloyd_max d=256 | 64 | 64× | 0.686 (93%) |
| binary_med_asym d=1024 | 128 | 32× | 0.726 (98%) |
| lloyd_max d=1024 | 256 | 16× | 0.747 (101%) |
| int8 d=1024 | 1024 | 4× | ~0.73 (99%) |
| float32 d=1024 | 4096 | 1× | 0.739 (100%) |

*\*NDCG@10 on mxbai/scifact*

---

## Which Method to Use

### Decision tree

```
Is storage extremely tight (< 32 bytes/vector)?
├─ YES → binary_med_asym at highest dim that fits
│         (e.g., d=256 → 32 bytes, 88% of float32 quality)
│
└─ NO → Can you afford ~64-256 bytes/vector?
         ├─ YES → lloyd_max_gauss at highest dim that fits
         │         (e.g., d=256 → 64 bytes, 93% quality)
         │         (e.g., d=1024 → 256 bytes, 101% quality)
         │
         └─ NO → int8 or float32 if storage is not an issue
```

### Quick recommendations

| Scenario | Method | Typical config | Quality |
|---|---|---|---|
| Extreme compression, mobile/edge | binary_med_asym | d=128, 16 bytes | ~77% |
| Balanced quality/storage | lloyd_max_gauss | d=256, 64 bytes | ~93% |
| Maximum quality, moderate storage | lloyd_max_gauss | full dim, 256 bytes | ~100%+ |
| Maximum quality, storage no concern | float32 | full dim | 100% |

### Methods we tested that are NOT recommended

| Method | Why not |
|---|---|
| Plain `binary_asym` (t=0) | `binary_med_asym` is strictly better at truncated dims |
| Per-dimension weighting | No improvement for L2-normalized embeddings (±0.003 NDCG) |
| Mixed-precision Matryoshka | Pure binary at higher dim always dominates on Pareto front |
| Ternary (dead-zone) | Lloyd-Max is better at same 2-bit budget |
| Quaternary (quartile) | Lloyd-Max is better — same bits, smarter boundaries |
| Sign-magnitude 2-bit | Worse than Lloyd-Max at truncated dims |
| Removing query centering from median | Hurts by 0.01–0.24 NDCG — the "bias term" is useful |

---

## Full Pipeline Example

End-to-end: embed → truncate → calibrate → index → search.

```python
import numpy as np

# ── Constants ──
LLOYD_BOUNDS = np.array([-0.9816, 0.0, 0.9816])
LLOYD_LEVELS = np.array([-1.5104, -0.4528, 0.4528, 1.5104], dtype=np.float32)


def truncate_and_normalize(emb, dim):
    t = emb[:, :dim].copy()
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.where(norms == 0, 1.0, norms)


# ── 1. CALIBRATE (once, from representative corpus sample) ──
def calibrate(corpus_emb, dim, method="lloyd_max"):
    corpus = truncate_and_normalize(corpus_emb, dim)
    
    if method == "binary_med":
        return {"medians": np.median(corpus, axis=0)}
    
    elif method == "lloyd_max":
        return {
            "medians": np.median(corpus, axis=0),
            "stds": np.std(corpus, axis=0).clip(1e-10),
        }


# ── 2. INDEX (per document, streaming) ──
def index_document(doc_emb, dim, cal, method="lloyd_max"):
    doc = truncate_and_normalize(doc_emb.reshape(1, -1), dim).squeeze()
    
    if method == "binary_med":
        centered = doc - cal["medians"]
        return np.packbits((centered > 0).astype(np.uint8))
    
    elif method == "lloyd_max":
        z = (doc - cal["medians"]) / cal["stds"]
        codes = np.digitize(z, LLOYD_BOUNDS).astype(np.uint8)
        return codes  # pack 4 codes per byte for storage


# ── 3. SEARCH (per query, against full index) ──
def search(query_emb, index_codes, dim, cal, method="lloyd_max", k=10):
    query = truncate_and_normalize(query_emb.reshape(1, -1), dim).squeeze()
    
    if method == "binary_med":
        q_centered = query - cal["medians"]
        unpacked = np.unpackbits(index_codes, axis=1)[:, :dim].astype(np.float32)
        signs = 2.0 * unpacked - 1.0
        scores = q_centered @ signs.T
    
    elif method == "lloyd_max":
        q_scaled = query * cal["stds"]         # fold σ into query
        recon_z = LLOYD_LEVELS[index_codes]     # (n_docs, dim) from codes
        scores = q_scaled @ recon_z.T
    
    # Return top-k indices
    top_idx = np.argpartition(-scores, k)[:k]
    return top_idx[np.argsort(-scores[top_idx])]
```
