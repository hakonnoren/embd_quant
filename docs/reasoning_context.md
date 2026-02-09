# Context for Reasoning: Improving Embedding Quantization for Vector Search

## Goal

We are building a practical evaluation framework for embedding quantization techniques used in vector search (Vespa). We want to find methods that maximize retrieval quality (NDCG@10, Recall@10) while minimizing storage (bytes per vector). All methods must be **streaming-friendly**: after a one-time calibration pass over the corpus, each new vector must be quantizable independently.

We want you to:
1. Identify relevant research that could improve our results
2. Use mathematical reasoning to propose concrete improvements
3. Explain *why* each improvement should work, with equations

---

## Setup

### Models
- **mxbai-embed-large-v1** (1024d, Matryoshka-trained, power-of-2 so Hadamard works)
- **nomic-embed-text-v1.5** (768d, Matryoshka-trained, not power-of-2)
- **all-MiniLM-L6-v2** (384d, no Matryoshka)

### Datasets
- **SciFact** (~5K docs, 300 queries, scientific claim verification)
- **NFCorpus** (~3.6K docs, 323 queries, nutrition/medical, multi-label with graded relevance)

### Evaluation
- **NDCG@10**: Normalized Discounted Cumulative Gain (ranking quality with graded relevance)
- **Recall@10**: Fraction of relevant docs found in top 10
- Ground truth: float32 full-dimensional inner product

---

## Methods Implemented

### 1. Float32 (baseline)
Standard inner product search. 4 bytes/dim.

### 2. Int8 (symmetric)
Per-dimension linear quantization to [-128, 127]. Ranges from calibration data (corpus min/max per dim).
```
starts = min(corpus, axis=0)
steps = (max - min) / 255
int8_val = (x - starts) / steps - 128
```
Search: cast both query and corpus to float32, inner product. 1 byte/dim.

### 3. Int8 Asymmetric
Only corpus quantized to int8. Query stays float32. Same inner product search.
Rationale: queries are few, so keeping them float32 is cheap. Only corpus suffers quantization error.

### 4. Binary (symmetric)
`sign(x) > 0`, packed as uint8. Search: Hamming distance (XOR + popcount). 1 bit/dim = 1/8 byte/dim.
Uses sentence_transformers: `np.packbits(embeddings > 0)`

### 5. Binary Asymmetric
Only corpus binarized. Query stays float32. Score: `float_query @ binary_corpus_unpacked.T` where unpacked maps {0,1} to {-1,+1}.
FAISS `IndexBinaryFlat` does NOT support this — it only does Hamming. So this is pure NumPy.

### 6. Binary + Rescore
Two-stage: (1) Hamming distance for top-4k candidates, (2) rescore with float32 inner product.
Requires float32 corpus in memory for stage 2.

### 7. Binary Median (symmetric)
Instead of `sign(x) > 0`, use `sign(x - median(x))` per dimension.
Equivalent to: subtract per-dim corpus median, then standard binary quantization.
Both query and corpus are centered by the same median vector before binarization.
```python
thresholds = np.median(corpus, axis=0)  # shape (dim,)
corpus_centered = corpus - thresholds
query_centered = query - thresholds
binary_corpus = packbits(corpus_centered > 0)
binary_query = packbits(query_centered > 0)
# Then Hamming distance as usual
```

### 8. Binary Median Asymmetric
Corpus binarized at median threshold. Query centered by median but stays float32.
```python
thresholds = np.median(corpus, axis=0)
binary_corpus = packbits((corpus - thresholds) > 0)
query_centered = query - thresholds
score = query_centered @ unpack_to_pm1(binary_corpus).T
```

### 9. Rotation (Hadamard / QR)
Applied before quantization to redistribute information across dimensions.
- **Hadamard**: R = (H * D)^n where H is Walsh-Hadamard, D is random ±1 diagonal. O(d log d). Requires power-of-2 dim. We use n=3 rounds.
- **QR**: Random orthogonal matrix via QR decomposition of Gaussian matrix. O(d^2). Works for any dim.

Rotation is applied to BOTH query and corpus with the same matrix for consistency.

### 10. Matryoshka Truncation
For models trained with Matryoshka representation learning, truncating to the first `d` dimensions and re-normalizing gives a valid lower-dimensional embedding. Tested at dims: 64, 128, 256, 512, full.

---

## Key Results

### Full-dimension performance (rotation=none)

| Model | Dataset | float32 | int8 | int8_asym | binary | binary_asym | binary_rescore | binary_median | binary_median_asym |
|-------|---------|---------|------|-----------|--------|-------------|----------------|---------------|--------------------|
| mxbai (1024d) | scifact | 0.7389 | 0.7277 | 0.7348 | 0.7042 | 0.7292 | 0.7389 | 0.7040 | 0.7259 |
| mxbai (1024d) | nfcorpus | 0.3868 | 0.3703 | 0.3873 | 0.3555 | 0.3728 | 0.3869 | 0.3559 | 0.3559 |
| nomic (768d) | scifact | 0.7033 | 0.6815 | 0.7064 | 0.6369 | 0.6764 | 0.7024 | 0.6707 | 0.6927 |
| nomic (768d) | nfcorpus | 0.3452 | 0.3291 | 0.3459 | 0.3053 | 0.3296 | 0.3461 | 0.3209 | 0.3227 |
| MiniLM (384d) | scifact | 0.6451 | 0.6340 | 0.6469 | 0.5894 | 0.6228 | 0.6451 | 0.5717 | 0.6138 |
| MiniLM (384d) | nfcorpus | 0.3160 | 0.3066 | 0.3154 | 0.2744 | 0.2943 | 0.3148 | 0.2507 | 0.2666 |

### Matryoshka dimension sweep: NDCG@10 (mxbai, scifact)

| Dim | float32 | int8 | int8_asym | binary | binary_asym | binary_median | binary_median_asym |
|-----|---------|------|-----------|--------|-------------|---------------|--------------------|
| 1024 | 0.7389 | 0.7277 | 0.7348 | 0.7042 | 0.7292 | 0.7040 | 0.7259 |
| 512 | 0.7277 | 0.7133 | 0.7249 | 0.6562 | 0.6754 | 0.6753 | 0.7031 |
| 256 | 0.6932 | 0.6700 | 0.6893 | 0.5539 | 0.6240 | 0.5989 | 0.6534 |
| 128 | 0.6714 | 0.6355 | 0.6677 | 0.4056 | 0.5210 | 0.4768 | 0.5688 |
| 64 | 0.5985 | 0.5669 | 0.5996 | 0.2323 | 0.3493 | 0.3340 | 0.4536 |

### Matryoshka dimension sweep: NDCG@10 (nomic, scifact)

| Dim | float32 | int8 | int8_asym | binary | binary_asym | binary_median | binary_median_asym |
|-----|---------|------|-----------|--------|-------------|---------------|--------------------|
| 768 | 0.7033 | 0.6815 | 0.7064 | 0.6369 | 0.6764 | 0.6707 | 0.6927 |
| 512 | 0.6976 | 0.6690 | 0.7031 | 0.5804 | 0.6528 | 0.6416 | 0.6641 |
| 256 | 0.6810 | 0.6394 | 0.6860 | 0.4747 | 0.5848 | 0.5537 | 0.6092 |
| 128 | 0.6445 | 0.5962 | 0.6430 | 0.3253 | 0.4247 | 0.4358 | 0.5233 |
| 64 | 0.5307 | 0.4969 | 0.5140 | 0.1719 | 0.2321 | 0.2602 | 0.3803 |

### Matryoshka dimension sweep: NDCG@10 (mxbai, nfcorpus)

| Dim | float32 | int8 | int8_asym | binary | binary_asym | binary_median | binary_median_asym |
|-----|---------|------|-----------|--------|-------------|---------------|--------------------|
| 1024 | 0.3868 | 0.3703 | 0.3873 | 0.3555 | 0.3728 | 0.3559 | 0.3559 |
| 512 | 0.3825 | 0.3604 | 0.3806 | 0.3243 | 0.3478 | 0.3339 | 0.3500 |
| 256 | 0.3624 | 0.3409 | 0.3628 | 0.2537 | 0.3042 | 0.2938 | 0.3191 |
| 128 | 0.3188 | 0.3081 | 0.3161 | 0.1794 | 0.2253 | 0.2244 | 0.2635 |
| 64 | 0.2557 | 0.2474 | 0.2545 | 0.1012 | 0.1363 | 0.1379 | 0.1940 |

### Matryoshka dimension sweep: NDCG@10 (nomic, nfcorpus)

| Dim | float32 | int8 | int8_asym | binary | binary_asym | binary_median | binary_median_asym |
|-----|---------|------|-----------|--------|-------------|---------------|--------------------|
| 768 | 0.3452 | 0.3291 | 0.3459 | 0.3053 | 0.3296 | 0.3209 | 0.3227 |
| 512 | 0.3321 | 0.3234 | 0.3339 | 0.2764 | 0.3041 | 0.3104 | 0.3156 |
| 256 | 0.3226 | 0.3162 | 0.3242 | 0.2208 | 0.2663 | 0.2671 | 0.2968 |
| 128 | 0.3050 | 0.2912 | 0.3072 | 0.1551 | 0.1991 | 0.2116 | 0.2558 |
| 64 | 0.2585 | 0.2458 | 0.2563 | 0.0922 | 0.1330 | 0.1347 | 0.1919 |

### Rotation effect at full dim

Rotation (3-round Hadamard for mxbai, QR for nomic/MiniLM) has small, inconsistent effects on binary quantization. Typical delta: +/-0.006 NDCG@10. Median thresholds and rotation address the same problem (dimension imbalance), so they don't stack — combining them yields diminishing or negative returns.

---

## Key Findings

### What works
1. **Asymmetric distance is the single biggest win.** Keeping the query in float32 while only quantizing the corpus consistently improves quality. The gain is larger for binary (up to +0.10 NDCG) than int8 (up to +0.03 NDCG).

2. **Median thresholds help dramatically at truncated Matryoshka dimensions.** At full dim, median is neutral or slightly negative. But at dim=64, binary_median_asym gives +0.10 to +0.15 NDCG over binary_asym. The reason: truncated embeddings have non-zero-centered dimensions, so the `> 0` threshold wastes bits.

3. **binary_median_asym sits between binary_asym and int8 on the Pareto front**, especially at truncated dims. At dim=512 it achieves 91-96% of float32 quality at 32x compression.

### What doesn't work
1. **Rotation provides no consistent benefit** for binary quantization. Deltas are noise-level (+/-0.006). Rotation + median thresholds interfere rather than complement.

2. **Median thresholds hurt MiniLM** (384d) at full dim (-0.02 to -0.03 NDCG). Possibly because the median estimates are noisy with small corpus, or MiniLM is already well-centered.

### The gap we want to close
There is a large quality gap between binary (32x compression, 1 bit/dim) and int8 (4x compression, 8 bits/dim). At dim=128:
- binary_median_asym: ~0.56 NDCG (mxbai/scifact)
- int8: ~0.64 NDCG
- int8_asym: ~0.67 NDCG

A 2-bit or 4-bit scheme at 8x-16x compression could fill this gap.

---

## Mathematical Framework

### Binary quantization as approximation

The true similarity is the inner product: `s(q, d) = q^T d`

**Symmetric binary**: `s_bin(q, d) = sign(q)^T sign(d)` counts dimension agreements. This approximates the angle between q and d (by the sign-consistency concentration inequality).

**Asymmetric binary**: `s_asym(q, d) = q^T sign(d)`. Each dimension contributes `q_i * sign(d_i)`. If `q_i` and `d_i` agree in sign, we get `+|q_i|`; if they disagree, `-|q_i|`. So large-magnitude query dimensions dominate — this is good because they carry more signal.

**Median-centered asymmetric**: `s_med(q, d) = (q - m)^T sign(d - m)` where `m` is the per-dim corpus median. This ensures the sign bits split the corpus 50/50, maximizing per-bit entropy. The query centering ensures dimensions near the median (ambiguous signal) contribute less.

### Quantization error analysis

For binary asymmetric, the per-dimension quantization error on the corpus is:
```
d_i ≈ sign(d_i) * E[|d_i|]
error_i = d_i - sign(d_i) * E[|d_i|]
```

The total score error is: `q^T error = sum_i q_i * error_i`

This error is smallest when the corpus dimensions are symmetric around zero (or the threshold) and have similar variance. Rotation aims to equalize variance; median thresholds aim to center each dim.

### Why rotation + median don't stack

Both transformations attempt to make `P(d_i > threshold) = 0.5` for each dimension `i`. Rotation achieves this by mixing dimensions; median achieves this by moving the threshold. If one succeeds, the other has no remaining imbalance to fix, and may introduce new distortions.

---

## Constraints

1. **Streaming-friendly**: After calibration (one pass over corpus), each new vector must be quantizable independently. No global recomputation when a new document arrives.
2. **Practical for Vespa**: Methods should work in a real search engine. Per-dimension metadata (thresholds, scales) is fine. Per-vector metadata beyond the quantized representation is costly.
3. **Asymmetric is preferred**: Queries are few and ephemeral; corpus is large and stored. Keeping queries in float32 is always practical.

---

## Questions for the Reasoning Model

### 1. Research directions
What published methods could improve binary or low-bit quantization for retrieval? Consider:
- Learned quantization (e.g., LSQ, product quantization, additive quantization)
- Optimal thresholds beyond median (e.g., maximum mutual information)
- Multi-bit extensions of binary quantization
- Weighted Hamming distances
- Connections to locality-sensitive hashing theory

### 2. Per-dimension weighting for asymmetric search
Currently binary_asym computes `q^T sign(d)`, treating all bits equally when computing the unpacked representation. If we instead compute `q^T (sigma * sign(d))` where `sigma_i = std(corpus_dim_i)`, we weight high-variance dimensions more.

- Does this have a theoretical justification? (Hint: think about minimizing expected score error)
- What's the optimal weight per dimension?
- Can we combine this with median thresholds?

### 3. 2-bit quantization
We want a 2-bit scheme (16x compression) between binary and int8. Options:
- **Ternary**: thresholds at -t and +t, values {-1, 0, +1}. What's the optimal t?
- **Quaternary**: thresholds at 25th, 50th, 75th percentile, values {-1.5, -0.5, +0.5, +1.5} or learned.
- How to do asymmetric search efficiently with 2-bit corpus?

### 4. Optimal threshold theory
For binary quantization with asymmetric search, is median actually the optimal threshold? Or is there a threshold that maximizes ranking correlation with the true inner product? Derive this.

### 5. Why does median hurt at full dimension?
At full dim for mxbai and MiniLM, binary_median_asym slightly underperforms binary_asym. But at truncated dims it's dramatically better. Why? Is this:
- Noise in median estimation?
- The full-dim embeddings already being zero-centered by design?
- An interaction between L2 normalization and median centering?

### 6. Matryoshka-aware quantization
For Matryoshka models, the first dimensions carry more information. Should quantization be dimension-aware?
- More bits for early dimensions, fewer for later?
- Mixed precision: float32 for dims 1-64, binary for dims 65-512?
- Is there a principled way to allocate bits across dimensions to minimize retrieval error?

### 7. Information-theoretic bounds
Given a d-dimensional normalized embedding and b bits of storage per vector, what is the theoretical maximum retrieval quality (e.g., in terms of inner product preservation or ranking correlation)? How close are our methods to this bound?
