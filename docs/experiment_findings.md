# Experiment Findings: Quantization Methods for Streaming Vector Search

Summary of results from `research_experiments.ipynb` and `hypothesis_tests.ipynb`.

## 1. Statistical Foundations

### Are embedding dimensions Gaussian?

Close enough. Per-dimension excess kurtosis averages ~0.0 (Gaussian = 0), but 50-60% of dimensions reject Shapiro-Wilk at p=0.05. The deviations from Gaussianity are mild — heavier tails and slight skewness — and don't meaningfully degrade Lloyd-Max performance.

### Does MSE predict ranking quality?

Yes. Pearson r = 0.738 between reconstruction MSE and NDCG@10 loss across all method×dimension combinations. MSE is a reliable proxy for method design, though not a perfect predictor (ranking depends on relative score ordering, not absolute error).

### Do dimensions correlate with each other?

Barely. Mean |correlation| < 0.04. PCA captures 10-17% more variance than random projections at 64 dims, but this is small. Each dimension is approximately independent, which validates per-dimension scalar quantization.

### When does median centering help?

Median offset (|median_i| / std_i) predicts the benefit with R² = 0.738. At full dimension (768-1024), L2-normalized embeddings already have near-zero medians — centering adds estimation noise for no gain. At Matryoshka-truncated dimensions (64-256), early dimensions carry strong per-dim bias (medians of 0.05-0.15), and centering recovers 0.05-0.15 bits of entropy per dimension.

The crossover: centering helps when |median_i| >> 1.253 * sigma_i / sqrt(n).

### Does calibration matter?

Yes, significantly. The mean NDCG@10 gap between calibrated and calibration-free 2-bit methods is +0.0264, especially pronounced at lower dimensions. Calibration metadata (medians, stds) is per-dimension and global — negligible memory overhead.


## 2. Method Comparison

### 1-Bit Methods (1 bit/dim)

| Method | Mechanism | Best use case |
|---|---|---|
| binary_asym | sign(d), float query | Baseline, no calibration needed |
| binary_med_asym | sign(d - median), float query centered | Best 1-bit method, dominates low-storage Pareto front |
| binary_weighted_asym | a_i * sign(d_i - m_i) | No improvement over unweighted for L2-normalized embeddings (variance is uniform) |
| ITQ + binary | Learned rotation then binary | Hurts at full dim, marginal help at low dims |

### 2-Bit Methods (2 bits/dim)

| Method | Mechanism | NDCG@10 (mxbai/scifact, full dim) |
|---|---|---|
| ternary_asym | {-1, 0, +1} with dead zone | ~0.72 |
| quaternary_asym | 4 quartile buckets | ~0.73 |
| lloyd_max_gauss | Gaussian-optimal 4-level | 0.7465 |
| sign_magnitude | sign bit + magnitude bit | ~0.73 |
| lloyd_max_empirical | Per-dim k-means, 4 centroids | ~0.745 |
| **residual_1+1** | **Two-stage binary** | **0.7486** |

### 3-Bit Methods

| Method | Mechanism | Key result |
|---|---|---|
| lloyd_max_3bit | Gaussian-optimal 8-level | 97.6% of float32 at d=128 (48 bytes). Enters Pareto front at multiple points |

### Query Centering: Does It Matter?

Tested `(q-m)^T sign(d-m)` vs `q^T sign(d-m)` (removing query centering). Result: **negligible difference**. The bias term `m^T sign(d-m)` is nearly constant across documents for L2-normalized embeddings, so it doesn't affect ranking. Both formulations are valid.


## 3. Residual 1+1 Bit Quantization (Detailed)

The best 2-bit method. Uses two successive 1-bit approximations instead of a single 2-bit codebook.

### Intuition

A single 2-bit quantizer (like Lloyd-Max) assigns each value to one of 4 levels. Residual quantization takes a different approach: approximate coarsely first, then refine the error. Each stage is just binary — simple and well-understood.

### Algorithm

**Stage 1: Coarse binary approximation**

1. Compute per-dimension median: m_i = median(corpus[:, i])
2. Center: c_i = d_i - m_i
3. Store sign bit: b1 = sign(c_i)
4. Compute conditional means for reconstruction:
   - alpha_pos_i = mean(c_i | c_i > 0) — average positive value per dim
   - alpha_neg_i = mean(c_i | c_i < 0) — average negative value per dim
5. Reconstruct: r1_i = alpha_pos_i if b1=+1, else alpha_neg_i

This is 1-bit binary_med_asym with optimal reconstruction weights.

**Stage 2: Refine the residual**

6. Compute residual: e_i = c_i - r1_i (what stage 1 got wrong)
7. Compute residual median: m2_i = median(e[:, i])
8. Center residual: e'_i = e_i - m2_i
9. Store second sign bit: b2 = sign(e'_i)
10. Compute conditional means of residual:
    - beta_pos_i = mean(e'_i | e'_i > 0)
    - beta_neg_i = mean(e'_i | e'_i < 0)
11. Reconstruct residual: r2_i = beta_pos_i if b2=+1, else beta_neg_i

**Full reconstruction:**

d_hat_i = m_i + r1_i + m2_i + r2_i

**Scoring:** q^T @ d_hat (asymmetric — query stays float32).

### Why it works

Consider a dimension where the corpus values are distributed as:

```
corpus values:  -0.8  -0.3  -0.1   0.2   0.5   0.9
median = 0.05
```

**Stage 1** splits at the median:
- Negative group: {-0.85, -0.35, -0.15} → alpha_neg = -0.45
- Positive group: {0.15, 0.45, 0.85}   → alpha_pos = +0.48

After stage 1, the errors are:
- {-0.40, +0.10, +0.30, -0.33, -0.03, +0.37}

**Stage 2** splits these errors at their median, capturing the "large vs small" distinction within each half of the original distribution. This effectively gives 4 reconstruction levels — like Lloyd-Max — but the levels are **data-driven** and **hierarchically adapted** rather than assuming a Gaussian shape.

### Why it beats Lloyd-Max

Lloyd-Max assumes Gaussian per-dimension distributions and uses fixed boundaries (-0.9816sigma, 0, +0.9816sigma). The residual method:

1. **Makes no distributional assumption.** Conditional means are optimal for any distribution.
2. **Adapts to asymmetry.** If positive and negative halves have different spreads, stage 1 captures this. Lloyd-Max with symmetric levels cannot.
3. **Successive refinement.** Each stage minimizes MSE given the previous approximation — this is the principle behind residual vector quantization (RVQ), applied here in the simplest possible form.

### Storage

- 2 bits per dimension per vector (same as Lloyd-Max, quaternary, ternary)
- Calibration: 6 floats per dimension (m, alpha_pos, alpha_neg, m2, beta_pos, beta_neg) — global, not per-vector
- At d=1024: 256 bytes per vector + 24 KB calibration metadata

### Limitations

- Requires calibration on the corpus (6 statistics per dimension)
- More expensive to encode than Lloyd-Max (two passes over the data)
- Scoring requires full reconstruction to float32 before inner product — no bit-manipulation shortcuts


## 4. The Lloyd-Max Anomaly

Lloyd-Max at 2 bits gives NDCG@10 = 0.7465 on mxbai/scifact, exceeding float32's 0.7389. We ran 5 conditions to explain this:

| Condition | What it tests | Result |
|---|---|---|
| C1: float32 baseline | — | 0.7389 |
| C2: float32 with sigma-weighting | Is implicit dim-weighting helping? | ~0.739 (no help) |
| C3: Lloyd-Max + sigma-fold | Current best | 0.7465 |
| C4: Lloyd-Max without sigma-fold | Is sigma-folding the trick? | 0.7465 (identical to C3) |
| C5: float32 + noise | Noise sensitivity | 1% noise: ~0.738; 5% noise: catastrophic |

**Conclusion:** The anomaly is a small-dataset artifact. SciFact has only 5K documents and 300 queries. Quantization acts as a regularizer/denoiser — collapsing nearby values to discrete levels smooths out noise in the embedding dimensions. This does NOT generalize: on NFCorpus (larger, multi-label), Lloyd-Max is significantly worse than float32 (-0.0077, bootstrap CI doesn't cross zero).

**Effective dimensionality** is 992/1024 (mxbai) and 739/768 (nomic), confirming near-uniform variance — there's nothing meaningful for sigma-weighting to exploit.


## 5. Mixed-Precision Matryoshka

For Matryoshka models, early dimensions carry more signal. A mixed-precision scheme keeps early dims at float32 and binarizes the rest:

| Configuration | Bytes/vec | Compression | Trade-off |
|---|---|---|---|
| float32 d=512 | 2048 | 1x | Full quality |
| f32:128 + bin:384 (=512d) | 560 | 3.7x | High quality, moderate compression |
| f32:64 + bin:448 (=512d) | 312 | 6.6x | Good balance |
| binary_med_asym d=512 | 64 | 32x | Maximum compression |

Score = (q_float^T d_float) / sqrt(float_dims) + (q_bin^T sign(d_bin)) / sqrt(binary_dims)

The normalization by sqrt(dims) ensures both components contribute equally. Performance falls between pure float32 and pure binary — a practical middle ground when neither extreme is acceptable.


## 6. Information-Theoretic Gaps

For d-dimensional unit vectors quantized at b bits per dimension, the rate-distortion bound gives minimum MSE ~= sigma^2 * 2^(-2b).

| Method | bits/dim | Measured MSE/dim | Theoretical bound | Gap |
|---|---|---|---|---|
| Binary | 1 | 0.3-0.5 | ~0.25 | 1.2-2x |
| Ternary | 2 | 0.1-0.2 | ~0.06 | 2-3x |
| Int8 | 8 | ~0.001 | ~10^-5 | Large but irrelevant |

The 2-3x gap at 2 bits suggests room for more sophisticated methods (product quantization, residual VQ), but with diminishing returns for practical ranking where Spearman rho matters more than MSE.


## 7. Pareto Front (mxbai/scifact)

| Bytes/vec | Method | NDCG@10 | % of float32 |
|---|---|---|---|
| 8 | binary_med_asym d=64 | 0.4536 | 61.4% |
| 16 | binary_med_asym d=128 | 0.5688 | 77.0% |
| 32 | binary_med_asym d=256 | 0.6534 | 88.4% |
| 48 | lloyd_max_3bit d=128 | 0.6552 | 88.7% |
| 64 | binary_med_asym d=512 | 0.7031 | 95.2% |
| 128 | binary_med_asym d=1024 | 0.7259 | 98.2% |
| 192 | lloyd_max_3bit d=512 | 0.7295 | 98.7% |
| 256 | residual_1+1 d=1024 | 0.7486 | 101.3% |

binary_med_asym dominates the low-storage regime (8-128 bytes). Lloyd-Max 3-bit occupies a useful niche between 1-bit and 8-bit. Residual 1+1 is the only 2-bit method that (slightly) exceeds float32.


## 8. Practical Recommendations for Vespa Streaming Search

| Storage budget | Recommended method | Expected quality |
|---|---|---|
| Minimal (8-16 bytes) | binary_med_asym at d=64-128 | 61-77% of float32 |
| Low (32-64 bytes) | binary_med_asym at d=256-512 | 88-95% of float32 |
| Medium (128 bytes) | binary_med_asym at d=1024 or int8 at d=128 | ~98% of float32 |
| Standard (256 bytes) | residual_1+1 at full dim | ~100% of float32 |
| Full (1024-4096 bytes) | int8 or float32 | 99-100% |

For streaming systems: binary_med_asym has the best drift robustness (only per-dim medians needed, stable for L2-normalized embeddings). Residual_1+1 requires 6 calibration stats per dimension — more drift-sensitive but still global, not per-vector.
