# New Research Directions for Embedding Quantization

This document extends our experimental findings with new research questions, theoretical analysis, ideas for novel methods, and a concrete experimental plan to push the Pareto front of storage vs retrieval quality.

---

## Table of Contents

1. [Where We Stand](#1-where-we-stand)
2. [The Lloyd-Max Anomaly: Why Quantization Beats Float32](#2-the-lloyd-max-anomaly-why-quantization-beats-float32)
3. [Optimality Theorems and Bounds](#3-optimality-theorems-and-bounds)
4. [New Methods to Push the Pareto Front](#4-new-methods-to-push-the-pareto-front)
5. [Feature-Based Method Selection](#5-feature-based-method-selection)
6. [Missing Experimental Coverage](#6-missing-experimental-coverage)
7. [Experimental Plan](#7-experimental-plan)

---

## 1. Where We Stand

### Current Pareto front (mxbai / scifact)

| Bytes/vec | Method | NDCG@10 | % of float32 |
|-----------|--------|---------|---------------|
| 8 | binary_med_asym d=64 | 0.4536 | 61% |
| 16 | binary_med_asym d=128 | 0.5688 | 77% |
| 32 | binary_med_asym d=256 | 0.6534 | 88% |
| 64 | binary_med_asym d=512 | 0.7031 | 95% |
| 128 | binary_asym d=1024 | 0.7292 | 99% |
| 256 | quaternary_asym d=1024 | 0.7407 | 100.2% |
| 256 | **lloyd_max_gauss d=1024** | **0.7465** | **101.0%** |
| 4096 | float32 d=1024 | 0.7389 | 100% |

### Open anomalies
1. **Lloyd-Max exceeds float32** (0.7465 > 0.7389 NDCG) — this needs explanation
2. **Quaternary (empirical centroids) also exceeds float32** on some configs
3. **Ternary optimal dead-zone is 0.3σ**, not the Gaussian-optimal 0.675σ
4. **Int8 has lower Spearman ρ (0.9179) than quaternary (0.9912)** despite 4× more bits
5. **Median centering hurts at full dim** but helps dramatically at truncated dims

---

## 2. The Lloyd-Max Anomaly: Why Quantization Beats Float32

This is the most surprising and important finding. A 2-bit quantizer outperforms 32-bit float on NDCG@10. This "shouldn't" happen if quantization only destroys information.

### Hypothesis A: Implicit variance weighting

Lloyd-Max reconstructs dimension $i$ as:

$$\hat{d}_i = \sigma_i \cdot L_{c_i}$$

where $L_{c_i} \in \{-1.5104, -0.4528, +0.4528, +1.5104\}$ is the reconstruction level for code $c_i$.

The scoring trick folds $\sigma_i$ into the query:

$$s(q, d) = \sum_i (q_i \cdot \sigma_i) \cdot L_{c_i}$$

This is **not** the standard inner product $q^T d$. It is a **variance-weighted** inner product. Dimensions with high $\sigma_i$ get more weight. If high-variance dimensions are more informative for retrieval, this is a *better* similarity measure than raw inner product.

**Test:** Compute $\text{NDCG}(q_{\text{float}} \cdot (\sigma \odot d_{\text{float}}))$ — float32 with $\sigma$-weighting but no quantization. If this also exceeds standard float32, the gain is from weighting, not quantization.

### Hypothesis B: Denoising / regularization

Quantization clusters nearby values together, smoothing out noise. Each quantized dimension takes one of 4 values, which is like projecting $d$ onto the closest point in a discrete codebook. If the embedding dimensions have noise (estimation error from finite training), collapsing to cluster centers can reduce noise without losing much signal — analogous to:

- Dropout improving generalization
- PCA truncation denoising
- Label smoothing in classification

**Test:** Add Gaussian noise to float32 embeddings and check if NDCG degrades. If the embeddings are noisy, this supports the denoising hypothesis.

### Hypothesis C: Rank-optimal ≠ MSE-optimal

NDCG cares about *ranking*, not score values. Quantization introduces MSE distortion but may preserve or even *improve* the ranking if:

- It amplifies score differences between relevant and irrelevant documents
- The quantization boundaries happen to separate relevant from irrelevant clusters

**Test:** Check if score *gaps* between relevant and irrelevant docs are larger after Lloyd-Max quantization.

### Key experiment: disentangling the factors

```
Condition 1: float32 standard inner product        → baseline
Condition 2: float32 σ-weighted inner product       → tests Hypothesis A alone
Condition 3: Lloyd-Max with σ-folding               → tests A + B together (current best)
Condition 4: Lloyd-Max WITHOUT σ-folding             → tests B alone
Condition 5: float32 + Gaussian noise (σ=0.01)      → tests noise sensitivity
```

If Condition 2 ≈ Condition 3 > Condition 1, the gain is from weighting (A).
If Condition 3 > Condition 2 > Condition 1, both weighting and denoising contribute.
If Condition 4 > Condition 1, denoising alone helps.

---

## 3. Optimality Theorems and Bounds

### 3.1 Is Lloyd-Max optimal among 2-bit quantizers?

**Theorem (Lloyd-Max optimality):** For a memoryless Gaussian source with known variance, the 4-level Lloyd-Max quantizer minimizes MSE among all 4-level scalar quantizers.

But our setting differs in three ways:

1. **Objective is ranking, not MSE.** The optimal quantizer for $\max \rho(q^T d,\; q^T \hat{d})$ may differ from the MSE-optimal one.

2. **Embeddings are not exactly Gaussian.** They live on the unit hypersphere $\mathbb{S}^{d-1}$. In high dimensions ($d > 100$), the projected coordinates are approximately Gaussian by the CLT / concentration of measure, but the tails are bounded by $[-1, 1]$ after L2 normalization.

3. **Asymmetric scoring.** The query is unquantized, so the error is one-sided: $q^T(\hat{d} - d)$ rather than $(\hat{q} - q)^T(\hat{d} - d)$.

**Conjecture:** For the asymmetric ranking objective with Gaussian-like per-dimension marginals, Lloyd-Max is *near*-optimal but may not be strictly optimal. The ranking-optimal quantizer would place boundaries to maximize the variance of the reconstructed score $\text{Var}[q^T \hat{d}]$ subject to a fixed number of levels, which differs from minimizing reconstruction MSE.

**Experimental test:** Compare Lloyd-Max to a quantizer optimized directly for Spearman $\rho$:
- Parameterize 3 boundaries and 4 levels per dimension
- Optimize via grid search or gradient-free optimization
- Compare on a held-out set

### 3.2 Is the median optimal for 1-bit?

For 1-bit asymmetric with threshold $t_i$, the score is:

$$s(q, d) = \sum_i (q_i - t_i) \cdot \text{sign}(d_i - t_i)$$

The median maximizes the entropy $H(\text{sign}(d_i - t_i))$ by ensuring $P(d_i > t_i) = 0.5$.

**But maximum entropy ≠ maximum ranking fidelity.** Consider an extreme: if a dimension has bimodal distribution with modes at $-0.8$ and $+0.2$, the median (≈ $-0.3$) would make sign bits unrelated to the actual bimodal structure. A threshold at $0$ would split the two modes cleanly.

**The right criterion** is to maximize the mutual information between $\text{sign}(d_i - t_i)$ and $d_i$, or equivalently, the expected absolute score contribution $E[|q_i - t_i| \cdot |d_i - t_i|]$.

For a symmetric unimodal distribution centered at $\mu_i$, the median equals the mean, and the threshold is:

$$t_i^* = \mu_i \quad \text{(median = mean for symmetric distributions)}$$

This is what we do. For asymmetric distributions, the optimal threshold can differ from both mean and median.

**Experimental test:** Per-dim threshold optimization using the `find_optimal_thresholds_per_dim` function we already have. Compare optimized vs median on all configs.

### 3.3 Rate-distortion gap

For an i.i.d. Gaussian source $\mathcal{N}(0, \sigma^2)$, the rate-distortion function is:

$$D(R) = \sigma^2 \cdot 2^{-2R}$$

At rate $R = 2$ bits/dim: $D_{\text{R-D}} = \sigma^2 / 16 = 0.0625 \sigma^2$.

Lloyd-Max at 4 levels achieves: $D_{\text{LM}} = 0.1175 \sigma^2$.

The **gap factor** is $0.1175 / 0.0625 = 1.88$. This gap arises because:

1. Scalar quantization cannot exploit inter-dimension correlations (which vector quantization can)
2. Fixed-length codes waste bits (entropy coding would use ~1.91 bits on average for the 4-level Gaussian quantizer, leaving 0.09 bits of slack)

**Can we close the gap while staying streaming-friendly?**

- **Entropy coding:** Variable-length codes (Huffman/ANS) could save ~5% storage but break byte-alignment and complicate random access. Not worth it for search.
- **Dimension rotation before quantization:** If dimensions are correlated, rotating to PCA/Hadamard basis decorrelates them, making scalar quantization more effective. Worth testing.
- **Vector quantization (PQ):** Product quantization groups dims into subvectors and quantizes each group jointly. Much closer to R-D bound but requires learned codebooks. Not streaming-friendly unless codebooks are fixed.

### 3.4 When does each method become optimal?

We can derive conditions under which each method dominates:

**Binary (1-bit) is optimal when:**
- Storage is the binding constraint (≤ 1 bit/dim)
- $d_{\text{eff}}$ (effective dimensionality) is high, so each bit averages over many directions
- The distribution is symmetric and unimodal per dimension

**Lloyd-Max (2-bit) is optimal when:**
- Per-dimension marginals are approximately Gaussian
- $\text{kurtosis} \approx 3$ (Gaussian kurtosis)
- Enough corpus vectors ($n \gg 100$) for reliable $\sigma$ estimation

**Int8 (8-bit) is optimal when:**
- Marginals are far from Gaussian (e.g., heavy-tailed, multimodal)
- The embedding has low effective dimensionality (few dimensions carry most signal)
- The additional storage cost is acceptable

**Quaternary (percentile-based) is optimal when:**
- Distribution is non-Gaussian but well-separated into quartiles
- Empirical centroids differ significantly from Gaussian centroids

---

## 4. New Methods to Push the Pareto Front

### 4.1 Residual quantization (1+1 bit)

**Idea:** Two-stage binary quantization. Stage 1 captures the sign, Stage 2 refines the magnitude.

```
Stage 1: b₁ = sign(d_i - median)           → captures direction
Stage 2: residual = d_i - reconstruct(b₁)
          b₂ = sign(residual - median(residual))  → refines magnitude
```

**Why it might beat Lloyd-Max:** Residual VQ adapts to the actual distribution rather than assuming Gaussian. Stage 1 split is at the median (maximum entropy), Stage 2 further subdivides each half.

**Storage:** 2 bits/dim (same as Lloyd-Max).

**Connection to Lloyd-Max:** If the distribution is Gaussian, the two methods converge: Stage 1 threshold = 0, Stage 2 thresholds = ±0.6745σ — which is close to Lloyd-Max's {-0.9816σ, 0, +0.9816σ}. But for non-Gaussian distributions, the residual approach may adapt better.

**Asymmetric scoring:**

$$s(q, d) = q^T \left[ \alpha_1 \cdot \text{sign}(d - m) + \alpha_2 \cdot \text{sign}(\text{residual} - m_r) \right]$$

where $\alpha_1, \alpha_2$ are the conditional mean magnitudes for each stage.

### 4.2 Rotation + Lloyd-Max

We showed rotation doesn't help binary. But the reasoning for *why* suggests it might help Lloyd-Max:

- Rotation (Hadamard) makes per-dimension distributions more Gaussian by the CLT (mixing many independent-ish components)
- Lloyd-Max assumes Gaussian marginals
- So rotation → more Gaussian → Lloyd-Max assumption is more accurate

This could be especially valuable for MiniLM (384d, no Matryoshka), where dimensions may be more structured/non-Gaussian.

### 4.3 3-bit quantization (8 levels)

The gap between 2-bit (256 bytes for 1024d) and 8-bit (1024 bytes) is large. 3 bits (384 bytes, 10.7× compression) fills this gap.

Lloyd-Max 3-bit uses 8 levels. The Gaussian-optimal boundaries and levels are:

| Level | Boundary | Reconstruction |
|-------|----------|---------------|
| 0 | — | −2.1520σ |
| 1 | −1.5104σ | −1.0500σ |
| 2 | −0.5006σ | −0.2451σ |
| 3 | 0 | +0.2451σ |
| 4 | +0.5006σ | +1.0500σ |
| 5 | +1.5104σ | +2.1520σ |

**Packing:** 3 bits don't divide evenly into bytes. Options:
- Pack 8 values into 3 bytes (wasteful at boundaries)
- Use 4 bits and waste 1 bit per dim (practical, 512 bytes for 1024d, still 8× compression)
- Use SIMD-friendly packing for production

### 4.4 Anisotropic quantization (ScaNN-style)

Google's ScaNN paper (Guo et al., 2020) observes that for inner product search, not all reconstruction errors matter equally. Errors **parallel** to the query affect the score, errors **perpendicular** don't:

$$q^T(d - \hat{d}) = q^T e_{\parallel} + q^T e_{\perp} = q^T e_{\parallel}$$

So we should minimize the **query-direction-weighted** distortion:

$$\min_{\hat{d}} \; E_q\left[ (q^T (d - \hat{d}))^2 \right] = \min_{\hat{d}} \; (d - \hat{d})^T \Sigma_q (d - \hat{d})$$

where $\Sigma_q = E[qq^T]$ is the query covariance matrix.

If $\Sigma_q \neq I$ (queries cluster in certain directions), the optimal quantizer should allocate more precision to query-aligned directions.

**Implementation for scalar quantization:**
1. Compute $\Sigma_q$ from the query set
2. Weight each dimension's quantization error by $\Sigma_{q,ii}$
3. Allocate more bits to high-weight dimensions (bit allocation)

**Challenge:** Requires access to queries at calibration time, which we have for evaluation but wouldn't in a pure streaming setting. However, in practice, you often have a sample of historical queries.

### 4.5 Learned binary codes (ITQ)

Iterative Quantization (Gong et al., 2012) finds a rotation matrix $R$ that minimizes the quantization error of binarization:

$$\min_R \| \text{sign}(XR) - XR \|_F^2 \quad \text{s.t.} \quad R^T R = I$$

This is solved by alternating:
1. Binary codes $B = \text{sign}(XR)$
2. Rotation $R = V U^T$ where $B^T X = U \Sigma V^T$ (SVD)

**Why it might help:** Our random Hadamard rotation is generic. ITQ finds the rotation specifically optimized for binary quantization error. At the cost of one SVD computation during calibration (still streaming-friendly for indexing), ITQ could improve binary quality.

### 4.6 Variable bit allocation across dimensions

**The reverse water-filling theorem** says: for independent Gaussian dimensions with variances $\sigma_1^2, \ldots, \sigma_d^2$ and a total bit budget $B$, the optimal per-dimension allocation is:

$$b_i^* = \frac{B}{d} + \frac{1}{2} \log_2 \frac{\sigma_i^2}{\left(\prod_j \sigma_j^2\right)^{1/d}}$$

Dimensions with higher variance get more bits.

For L2-normalized embeddings, per-dim variances are roughly equal (we showed this in Q2), so this predicts uniform allocation. But after Matryoshka truncation or for non-normalized embeddings, this could help.

**More interesting variant:** Weight by $\sigma_i^2 \cdot (\Sigma_q)_{ii}$ — dimensions that are both high-variance in the corpus AND high-variance in the queries should get more bits.

### 4.7 Asymmetric distance computation (ADC) for fast 2-bit search

Currently our 2-bit search does: unpack codes → reconstruct → matrix multiply. For production, we can use **lookup tables**:

```python
# Precompute per-dimension distance tables
# For each dim i and code c ∈ {0,1,2,3}:
#   table[i][c] = q_i * level[c] * sigma_i
# Then score = sum_i table[i][codes[i]]

def adc_search(query, codes, levels, sigmas):
    dim = len(query)
    table = np.outer(query * sigmas, levels)  # (dim, 4) lookup table
    # For each document, sum the table lookups
    scores = table[np.arange(dim), codes.T].sum(axis=0)  # vectorized
    return scores
```

This avoids the full reconstruction and uses integer indexing, which is cache-friendly. With 2-bit codes packed 4 per byte, we can even precompute 256-entry byte-level tables.

---

## 5. Feature-Based Method Selection

### 5.1 What features predict the best method?

Given a new embedding model and corpus, which quantization method should you use? We propose computing cheap per-dimension statistics to predict this:

| Feature | Computation | What it predicts |
|---------|------------|-----------------|
| **Kurtosis** | $\kappa = E[(x-\mu)^4] / \sigma^4$ | Gaussianity. $\kappa \approx 3$ → Gaussian → Lloyd-Max optimal. $\kappa \gg 3$ → heavy tails → empirical Lloyd-Max better. |
| **Sign imbalance** | $\bar{\delta} = \text{mean}_i \|P(d_i > 0) - 0.5\|$ | Whether median centering helps. High $\bar{\delta}$ → centering improves binary. |
| **Effective dimensionality** | $d_{\text{eff}} = (\sum \sigma_i^2)^2 / \sum \sigma_i^4$ | How many dims carry signal. Low $d_{\text{eff}}$ → early truncation works. |
| **Bimodality coefficient** | $BC = (\gamma^2 + 1) / (\kappa + 3(n-1)^2/((n-2)(n-3)))$ | Whether dimensions are bimodal. Bimodal → simple sign split is already optimal. |
| **Intrinsic dimension** | 2-NN estimator (Facco et al.) | Manifold complexity. Low → aggressive compression possible. |
| **Dim correlation** | Mean $|\text{Corr}(d_i, d_j)|$ for $i \neq j$ | Whether rotation/decorrelation helps. High correlation → rotation + quantization wins. |

### 5.2 Decision flowchart

```
Input: bit_budget, embedding stats
│
├── bit_budget = 1 bit/dim
│   ├── sign_imbalance > 0.1 → binary_med_asym
│   └── sign_imbalance ≤ 0.1 → binary_asym (median adds noise)
│
├── bit_budget = 2 bits/dim
│   ├── kurtosis ≈ 3 (Gaussian) → lloyd_max_gauss
│   ├── kurtosis > 4 (heavy tails) → lloyd_max_empirical
│   └── kurtosis < 2.5 (light tails) → quaternary_asym (percentile-based)
│
├── bit_budget = 3-8 bits/dim
│   └── int8_asym (simple, robust)
│
└── Want maximum compression with Matryoshka?
    └── truncate dims first, then quantize (compression stacks multiplicatively)
```

### 5.3 Proposed experiment

For all 9 model×dataset combinations:
1. Compute the 6 features above
2. Run all quantization methods
3. Fit a simple model (decision tree or logistic regression) predicting which method wins
4. Report which features matter most

---

## 6. Missing Experimental Coverage

We have cached embeddings for **3 models × 3 datasets = 9 combinations** but have only tested a subset:

| | SciFact | NFCorpus | FiQA |
|---|:---:|:---:|:---:|
| mxbai-embed-large-v1 | ✅ Extensive | ✅ Partial | ❌ Untested |
| nomic-embed-text-v1.5 | ✅ Extensive | ✅ Partial | ❌ Untested |
| all-MiniLM-L6-v2 | ❌ Untested | ❌ Untested | ❌ Untested |

### Why FiQA matters
- 57K documents (10× larger than SciFact/NFCorpus)
- Tests whether calibration statistics from a larger corpus are more stable
- Tests whether the Lloyd-Max > float32 anomaly persists at scale

### Why MiniLM matters
- 384 dimensions, no Matryoshka training
- Tests whether our methods generalize beyond Matryoshka models
- Smaller dimensionality means each dimension carries more signal — quantization may hurt more

---

## 7. Experimental Plan

### Phase 1: Explain the anomaly (highest priority)
1. **Disentangle σ-weighting from quantization** — the 5-condition experiment from §2
2. **Test score gap amplification** — are score gaps between relevant/irrelevant larger after Lloyd-Max?
3. **Noise sensitivity** — add noise to float32 and measure degradation

### Phase 2: New methods
4. **Residual 1+1 bit quantization** — two-stage binary
5. **Rotation + Lloyd-Max** — Hadamard before Lloyd-Max for more Gaussian dims
6. **3-bit Lloyd-Max** — fill the gap between 2-bit and 8-bit
7. **ITQ rotation for binary** — learned rotation vs random Hadamard

### Phase 3: Full coverage
8. **All 9 model×dataset combinations** with all Pareto-contending methods
9. **FiQA at scale** — 57K docs, how do methods scale?
10. **MiniLM deep dive** — non-Matryoshka model behavior

### Phase 4: Feature-based selection
11. **Compute embedding features** (kurtosis, bimodality, effective dim, etc.) for all 9 configs
12. **Correlate features with method performance** — which stats predict the winner?
13. **Build a simple method selector** from cheap corpus statistics

### Phase 5: Speed benchmarks
14. **ADC lookup table scoring** — benchmark 2-bit ADC vs matrix multiply
15. **Packed binary Hamming vs float-dot** — measure actual wall-clock speedups
16. **End-to-end latency** — calibration time + indexing time + query latency

---

## Summary of New Research Questions

| ID | Question | Type | Priority |
|----|----------|------|----------|
| Q9 | Why does Lloyd-Max exceed float32? σ-weighting or denoising? | Theory + Experiment | ★★★ |
| Q10 | Is Lloyd-Max optimal for ranking (not just MSE)? | Theory | ★★☆ |
| Q11 | Can residual 1+1 bit beat Lloyd-Max? | New Method | ★★★ |
| Q12 | Does rotation help Lloyd-Max (by making dims more Gaussian)? | Experiment | ★★☆ |
| Q13 | Where does 3-bit Lloyd-Max sit on the Pareto front? | New Method | ★★☆ |
| Q14 | Can we predict the best method from corpus statistics? | Feature Analysis | ★★★ |
| Q15 | Does the Lloyd-Max anomaly persist on FiQA (57K docs)? | Scale Test | ★★☆ |
| Q16 | How do non-Matryoshka models (MiniLM) behave? | Generalization | ★★☆ |
| Q17 | Can anisotropic (ScaNN-style) quantization improve further? | New Method | ★★☆ |
| Q18 | Does ITQ (learned rotation) beat random Hadamard for binary? | New Method | ★☆☆ |
| Q19 | Is the int8 anomaly (lower ρ than 2-bit) a bug or a feature? | Investigation | ★★☆ |
| Q20 | What is the effective dimensionality of each model's embeddings? | Feature Analysis | ★☆☆ |
