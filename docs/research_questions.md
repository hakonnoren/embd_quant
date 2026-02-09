# Open Research Questions for Embedding Quantization

## Context

We have a working evaluation framework with 10 quantization methods across 3 models, 3 datasets, and multiple Matryoshka truncation levels. This document captures:
1. What we know and can prove theoretically
2. Open questions we want a reasoning model to investigate
3. Suggested experiments to answer them
4. Ideas for new methods that could push the Pareto front

See also: [quantization_methods.md](quantization_methods.md) for method details, [deployment_considerations.md](deployment_considerations.md) for production concerns.

---

## Part I: Theoretical Foundations

### 1.1 The Asymmetric Scoring Decomposition

All our methods compute:

$$s(q, d) = q^T \hat{d}$$

where $q$ is the float32 query and $\hat{d}$ is the reconstructed corpus vector. The quantization error is $\epsilon = d - \hat{d}$, so:

$$s(q, d) = q^T d - q^T \epsilon$$

The ranking error comes entirely from $q^T \epsilon$. For **symmetric** quantization (both sides quantized), the error would be $(q + \epsilon_q)^T (d + \epsilon_d) = q^T d + q^T \epsilon_d + \epsilon_q^T d + \epsilon_q^T \epsilon_d$, which has three error terms instead of one. This is why asymmetric is strictly better: we eliminate two of three error sources.

**Can we prove**: For any quantizer $Q$ with $b$ bits per dimension, the expected ranking loss under asymmetric scoring is at most half that of symmetric scoring?

### 1.2 Lloyd-Max Optimality

The Lloyd-Max quantizer minimizes $E[(X - \hat{X})^2]$ for scalar random variable $X$ quantized to $K$ levels. For $K=4$ and $X \sim N(0,1)$:

- Boundaries: $\{-0.9816, 0, +0.9816\}$
- Levels: $\{-1.5104, -0.4528, +0.4528, +1.5104\}$
- MSE = 0.1175 (vs 0.1188 for equal-count quartiles, a 1.1% improvement)

**What we can prove:**
- Lloyd-Max is the unique MSE-optimal scalar quantizer for a given distribution and number of levels (under mild regularity conditions)
- For Gaussian sources, the fixed constants above are exact
- The improvement over uniform quantization grows with fewer levels (large at 2-4 levels, negligible at 256)

**What we cannot prove (yet):**
- That MSE-optimal quantization implies ranking-optimal quantization. NDCG depends on the rank ordering of scores, not their absolute values. A quantizer that's worse in MSE could theoretically be better in NDCG if its errors are rank-preserving.
- That the Gaussian assumption is justified for L2-normalized embeddings. We assume it but haven't verified it.

### 1.3 The Binary Median as 1-bit Lloyd-Max

For a symmetric distribution, the 2-level Lloyd-Max quantizer places its boundary at the mean (= median for symmetric distributions). Our `binary_median_asym` does exactly this:

$$\hat{d}_i = \text{sign}(d_i - m_i) \quad \text{where } m_i = \text{median}(\text{corpus}_i)$$

So `binary_median_asym` IS the 1-bit Lloyd-Max quantizer (with reconstruction levels ±1 instead of ±E[|X-m|], but in asymmetric search the query absorbs the scaling).

**Key subtlety — the query centering term:**

$$s(q, d) = (q - m)^T \text{sign}(d - m) = \underbrace{q^T \text{sign}(d-m)}_{\text{standard binary}} - \underbrace{m^T \text{sign}(d-m)}_{\text{bias correction}}$$

The second term is document-dependent (different documents have different sign patterns relative to the median). It acts as a bias correction: documents whose deviation pattern from the corpus center correlates with the median itself get penalized. Removing it (not centering the query) hurts NDCG by 0.01-0.24 in our experiments.

**Question**: Can we derive the exact conditions under which query centering helps? Is it related to the correlation between the median vector $m$ and the sign pattern $\text{sign}(d-m)$?

### 1.4 The Lloyd-Max Scoring Trick

For Lloyd-Max, the reconstructed document is:

$$\hat{d}_i = m_i + \sigma_i \cdot L[\text{code}_i]$$

where $L$ is the fixed level table. The score becomes:

$$s(q, d) = q^T \hat{d} = \underbrace{q^T m}_{\text{constant across docs}} + (q \odot \sigma)^T L[\text{code}]$$

The first term doesn't affect ranking. So we precompute $q' = q \odot \sigma$ once per query, then score against the level-reconstructed codes. This is important for efficiency: no per-document denormalization needed.

**Note**: This trick doesn't work for quaternary because its centroids are data-dependent per-bucket means, not fixed constants. The reconstruction is $\hat{d}_i = c_{i, \text{code}_i}$ where centroids $c$ vary per dimension and per bucket.

---

## Part II: Empirical Results Summary

### 2.1 Full Dimension Results (NDCG@10, rotation=none)

#### SciFact (5K docs, 300 queries)

| Method | bits/dim | mxbai (1024d) | nomic (768d) | miniLM (384d) |
|---|---|---|---|---|
| float32 | 32 | 0.7389 | 0.7032 | 0.6451 |
| int8_asym | 8 | 0.7348 | 0.7064 | 0.6469 |
| lloyd_max_gauss | 2 | **0.7465** | 0.6877 | **0.6576** |
| quaternary_asym | 2 | 0.7407 | 0.6824 | 0.6523 |
| binary_median_asym | 1 | 0.7259 | 0.6927 | 0.6138 |
| binary_asym | 1 | 0.7292 | 0.6764 | 0.6228 |

#### NFCorpus (3.6K docs, 323 queries)

| Method | bits/dim | mxbai (1024d) | nomic (768d) |
|---|---|---|---|
| float32 | 32 | 0.3868 | 0.3452 |
| int8_asym | 8 | 0.3873 | 0.3459 |
| lloyd_max_gauss | 2 | **0.3792** | **0.3415** |
| quaternary_asym | 2 | 0.3732 | 0.3344 |
| binary_median_asym | 1 | 0.3559 | 0.3227 |
| binary_asym | 1 | 0.3728 | 0.3296 |

### 2.2 Matryoshka Truncation (dim=256, NDCG@10)

#### SciFact

| Method | bits/dim | bytes/vec | mxbai | nomic |
|---|---|---|---|---|
| float32 | 32 | 1024 | 0.6932 | 0.6810 |
| int8_asym | 8 | 256 | 0.6893 | 0.6860 |
| lloyd_max_gauss | 2 | 64 | 0.6855 | 0.6472 |
| quaternary_asym | 2 | 64 | 0.6643 | 0.6458 |
| binary_median_asym | 1 | 32 | 0.6534 | 0.6092 |
| binary_asym | 1 | 32 | 0.6240 | 0.5848 |

#### NFCorpus

| Method | bits/dim | bytes/vec | mxbai | nomic |
|---|---|---|---|---|
| float32 | 32 | 1024 | 0.3624 | 0.3226 |
| int8_asym | 8 | 256 | 0.3628 | 0.3242 |
| lloyd_max_gauss | 2 | 64 | 0.3465 | 0.3148 |
| quaternary_asym | 2 | 64 | 0.3329 | 0.3072 |
| binary_median_asym | 1 | 32 | 0.3191 | 0.2968 |
| binary_asym | 1 | 32 | 0.3042 | 0.2663 |

### 2.3 Key Empirical Observations

1. **Lloyd-Max exceeds float32 at full dim on SciFact** (0.7465 vs 0.7389 for mxbai). This suggests a regularization effect — quantization is acting like noise injection that prevents overfitting to embedding noise.

2. **Binary_median_asym dominates binary_asym at truncated dims** but the gap narrows at full dim (and sometimes reverses for NFCorpus at full dim). This suggests median centering matters more when the per-dimension distribution is more skewed (which happens after Matryoshka truncation + renormalization).

3. **Lloyd-Max consistently beats quaternary** at the same 2-bit budget. The gap is 1-3% NDCG, consistent with the theoretical MSE improvement.

4. **NFCorpus is harder** for all quantized methods — the gap to float32 is larger. NFCorpus has graded relevance with many partially-relevant documents, making fine-grained ranking more important.

5. **Rotation (Hadamard/QR) doesn't help** — results are within noise (±1% NDCG). For models that are already Matryoshka-trained, the dimension distribution appears balanced enough that rotation adds nothing.

---

## Part III: Open Questions

### Q1: Is MSE minimization the right objective for ranking?

**The gap**: Lloyd-Max minimizes $E[\|d - \hat{d}\|^2]$ (MSE), but we evaluate NDCG (a ranking metric). These are related but not identical.

The score error for a (query, document) pair is:
$$\Delta s_{q,d} = q^T(d - \hat{d}) = q^T \epsilon_d$$

A ranking error occurs when $\Delta s_{q,d_1} - \Delta s_{q,d_2}$ flips the ordering of two documents. This depends not on the absolute error but on the **correlation structure** of errors across documents.

**Hypothesis**: A quantizer that minimizes $E[(\Delta s_{q,d_1} - \Delta s_{q,d_2})^2]$ (pairwise ranking error) might differ from the MSE-optimal quantizer. Specifically, if quantization errors are correlated across documents (e.g., all documents in a cluster get the same error pattern), ranking within clusters is preserved even if absolute errors are large.

**Suggested experiment**: Compare Lloyd-Max (MSE-optimal) against a modified quantizer that maximizes rank correlation (Kendall's tau) between quantized and float32 scores on a held-out query set. If they give the same result, MSE is sufficient; if not, there's room for a ranking-aware quantizer.

### Q2: What is the per-dimension distribution of L2-normalized embeddings?

**Why it matters**: Lloyd-Max assumes Gaussian. If the actual distribution has heavier tails (e.g., Laplacian) or is asymmetric, different boundaries would be optimal.

For a random unit vector in $\mathbb{R}^d$, each coordinate is marginally distributed as $\text{Beta}(1/2, (d-1)/2)$ projected to $[-1, 1]$, which converges to $N(0, 1/d)$ for large $d$. But learned embeddings are not random — the model imposes structure.

**Suggested experiment**:
1. For each model, extract the full corpus embeddings at each Matryoshka dim
2. Compute per-dimension kurtosis, skewness, and fit Gaussian vs Laplacian vs empirical
3. Check if dimensions with higher kurtosis (heavier tails) benefit more from Lloyd-Max vs quartile
4. If the distribution is consistently non-Gaussian, compute the true Lloyd-Max constants for the empirical distribution

**Prediction**: Early Matryoshka dimensions (high-variance, most informative) may be more Gaussian. Later dimensions (lower-variance, less trained) may be more uniform or heavy-tailed. This could justify different quantization strategies per dimension group.

### Q3: Can we prove when binary_median_asym beats binary_asym?

Empirically, median centering helps most at truncated dims. Can we formalize this?

**Setup**: Let $d_i \sim F_i$ with mean $\mu_i$ and median $m_i$. For zero-centered distributions ($\mu_i = m_i = 0$), centering does nothing. The benefit of centering is proportional to $|m_i|$ — the offset of the distribution from zero.

After Matryoshka truncation to $k$ dims and L2 renormalization:
$$d'_i = \frac{d_i}{\sqrt{\sum_{j=1}^k d_j^2}}$$

The renormalization amplifies dimensions that were originally small (later Matryoshka dims), potentially introducing asymmetry. If the first $k$ dimensions capture most variance, the remaining ones get inflated, shifting their medians away from zero.

**Testable prediction**: The improvement from median centering should correlate with:
$$\text{benefit} \propto \frac{1}{d} \sum_{i=1}^d |m_i|^2$$

i.e., the squared norm of the median vector. Compute this for each model × dim combination and check if it predicts the NDCG gap between `binary_median_asym` and `binary_asym`.

### Q4: Non-uniform bit allocation across dimensions

Current approach: every dimension gets the same number of bits (1, 2, or 8). But Matryoshka-trained models explicitly encode more information in early dimensions.

**The idea**: Allocate more bits to important dimensions, fewer to less important ones. For a total budget of $B$ bits across $d$ dimensions:

$$\min_{b_1, ..., b_d} \sum_{i=1}^d E[\epsilon_i^2] \quad \text{s.t.} \quad \sum_{i=1}^d b_i = B, \quad b_i \in \{0, 1, 2, 4, 8\}$$

The MSE of Lloyd-Max at $b$ bits for a Gaussian with variance $\sigma_i^2$ is approximately:
- 0 bits: $\sigma_i^2$ (no information)
- 1 bit: $0.3634 \cdot \sigma_i^2$ (63.7% error reduction)
- 2 bits: $0.1175 \cdot \sigma_i^2$ (88.3% error reduction)
- 4 bits: $0.009497 \cdot \sigma_i^2$ (99.1% error reduction)
- 8 bits: $1.5 \times 10^{-5} \cdot \sigma_i^2$ (99.998% error reduction)

Optimal allocation: give more bits to dimensions with higher $\sigma_i$ (more variance = more information to preserve). This is the classic water-filling / reverse water-filling problem from rate-distortion theory.

**But**: in asymmetric scoring, the ranking error for dimension $i$ is $q_i \cdot \epsilon_i$, not just $\epsilon_i$. The "importance" of a dimension depends on both its variance $\sigma_i^2$ AND the typical query loading $E[q_i^2]$. For Matryoshka models, early dimensions tend to have both higher variance and higher query loading, so allocating more bits to them is doubly justified.

**Suggested experiment**:
1. Compute per-dim variance of corpus embeddings at each truncation level
2. Compute per-dim mean squared query value $E[q_i^2]$
3. Run a simple heuristic: 4-bit for dims where $\sigma_i \cdot \sqrt{E[q_i^2]} > \tau$, 2-bit for the rest
4. Compare against uniform 2-bit

**Complication**: Non-uniform bit allocation requires variable-length encoding, which complicates memory access patterns. For a practical system, grouping dims into "blocks" (e.g., first 128 dims at 4 bits, rest at 2 bits) is more realistic than per-dim allocation.

### Q5: Product Quantization vs Scalar Quantization

All our methods are **scalar quantizers**: each dimension is quantized independently. Product Quantization (PQ) groups dimensions into subspaces (e.g., 8 dims each) and uses a shared codebook per subspace.

**Why PQ could help**: Scalar quantization ignores correlations between dimensions. If $d_1$ and $d_2$ are correlated, quantizing them jointly with a 2D codebook preserves more information than quantizing each independently.

**Why PQ might not help here**:
1. L2-normalized embeddings from trained models tend to have low inter-dimension correlation (the model learns decorrelated representations)
2. PQ with small codebooks (e.g., 256 codes per subspace) is equivalent to 8 bits per subspace. For 8 dims per subspace, that's 1 bit/dim — same as binary. But PQ can capture correlations.
3. PQ requires storing codebooks, which is negligible for large corpora but adds complexity
4. PQ doesn't have a natural "Matryoshka-compatible" structure — truncating dimensions breaks subspace boundaries

**Key question**: Is there meaningful correlation structure in the embeddings that scalar quantization misses? Compute the covariance matrix $\Sigma$ of corpus embeddings. If $\Sigma$ is nearly diagonal, scalar quantization is near-optimal. If there are significant off-diagonal terms, PQ could help.

**Suggested experiment**: Compute explained variance ratio of PCA components vs original dimensions. If the first $k$ PCA components explain significantly more variance than the first $k$ original dimensions, there's redundancy that PQ could exploit. For Matryoshka models, we expect this gap to be small (Matryoshka training already front-loads variance).

### Q6: Why does Lloyd-Max exceed float32?

At full dimension on SciFact, lloyd_max_gauss (0.7465) exceeds float32 (0.7389) for mxbai. This is surprising — quantization loses information, so how can it improve ranking?

**Hypotheses**:

1. **Regularization**: Quantization acts like noise injection, smoothing out idiosyncratic patterns in corpus vectors that don't generalize to queries. Similar to how dropout improves neural networks.

2. **Implicit dimension weighting**: The standardization step ($z = (d-m)/\sigma$) and the scoring trick ($q' = q \odot \sigma$) effectively weights dimensions by their standard deviation. If some high-variance dimensions contain noise rather than signal, down-weighting them (through quantization's lossy compression) helps.

3. **Statistical fluke**: The improvement is ~1% NDCG, which could be within the variance of the metric on 300 queries.

**Suggested experiment**:
- Bootstrap confidence intervals for NDCG@10 (resample queries with replacement, 1000 iterations)
- If the 95% CI of float32 and lloyd_max overlap, the difference is not significant
- Also test on the larger fiqa dataset (648 queries) for more statistical power

### Q7: Can we derive information-theoretic bounds?

**Rate-distortion theory** gives the minimum distortion (MSE) achievable at a given bit rate for a source distribution. For a Gaussian source with variance $\sigma^2$:

$$D(R) = \sigma^2 \cdot 2^{-2R}$$

where $R$ is bits per sample. At $R=2$, $D = \sigma^2/16 = 0.0625\sigma^2$.

Our Lloyd-Max achieves MSE = $0.1175\sigma^2$ at 2 bits, which is 1.88× the theoretical limit. The gap comes from the constraint to use a fixed set of levels (scalar quantization can't match vector quantization's efficiency).

**But**: Rate-distortion bounds apply to MSE, not to ranking quality. We care about preserving the ordering of $q^T d$ values, not the values themselves.

**Question for reasoning model**: Is there a rate-distortion-style bound for ranking preservation? Something like: "Given $d$ dimensions and $b$ bits per dimension, the expected fraction of pairwise ranking errors is at least $f(d, b)$." This would tell us how close our methods are to fundamental limits.

### Q8: Model-specific quantization behavior

Different models respond differently to quantization:

| Observation | mxbai | nomic | miniLM |
|---|---|---|---|
| Lloyd-Max vs float32 (full dim) | +1.0% | -2.2% | +1.9% |
| Median centering benefit (full dim) | -0.5% | +2.4% | -1.4% |
| Rotation effect | neutral | neutral | neutral |

**Why does nomic dislike Lloyd-Max at full dim but love median centering?**

Nomic is 768d (not power of 2), trained with different objectives. Its per-dimension distributions may be less Gaussian than mxbai's, making the Gaussian Lloyd-Max constants suboptimal. Meanwhile, its dimension medians may be further from zero (larger $|m_i|$), making centering more valuable.

**Suggested diagnostic**:
1. Compute kurtosis per dimension for each model (Gaussian has kurtosis = 3)
2. Compute $\|m\|^2 / d$ (mean squared median offset) for each model
3. Check if kurtosis predicts Lloyd-Max improvement and $\|m\|^2$ predicts median centering benefit

This could yield a **model-selection rule**: given a new embedding model, compute two statistics ($\bar{\kappa}$ = mean kurtosis, $\|m\|^2/d$ = median offset) and choose the quantization method accordingly.

---

## Part IV: Ideas for New Methods

### Idea 1: Adaptive-resolution scalar quantization

Instead of uniform bit allocation, use Matryoshka dimension ordering to guide allocation:

```
dims 1-64:   4 bits/dim (Lloyd-Max 16-level)  → 32 bytes
dims 65-256: 2 bits/dim (Lloyd-Max 4-level)   → 48 bytes
dims 257+:   1 bit/dim  (binary median)       → variable
```

Total for 1024d: 32 + 48 + 96 = 176 bytes (vs 256 for uniform 2-bit, 128 for uniform 1-bit).

**Why it might work**: Matryoshka explicitly trains early dims to be most informative. Giving them more bits preserves more of this information.

**Complication**: Requires implementing 4-bit Lloyd-Max (16 levels). The Gaussian-optimal constants for 16 levels are known but we haven't implemented them.

### Idea 2: Empirical Lloyd-Max (per-dimension iterative optimization)

Instead of assuming Gaussian and using fixed constants, run the Lloyd-Max algorithm (iterative k-means in 1D) per dimension on the calibration data:

```python
# Per dimension j:
for iteration in range(20):
    # Assign points to nearest level
    codes = assign(corpus[:, j], levels)
    # Update levels to mean of assigned points
    levels = [corpus[codes == k, j].mean() for k in range(4)]
    # Update boundaries to midpoints between levels
    bounds = [(levels[k] + levels[k+1]) / 2 for k in range(3)]
```

This is guaranteed to converge and gives the true MSE-optimal quantizer for the empirical distribution.

**Our hypothesis notebook tested this** (`lloyd_max_empir`) and found it gave nearly identical results to Gaussian Lloyd-Max. This suggests the Gaussian assumption is good enough — but it's worth retesting on larger/different datasets.

### Idea 3: Query-aware quantization

Standard quantization minimizes reconstruction MSE $E[\|d - \hat{d}\|^2]$. But for ranking, we should minimize the ranking error $E[(q^T(d_1 - \hat{d}_1) - q^T(d_2 - \hat{d}_2))^2]$.

If we have access to a representative query set, we can optimize quantization levels to minimize ranking loss directly. This is a query-weighted MSE where dimension $i$ gets weight proportional to $E[q_i^2]$:

$$\min \sum_{i=1}^d E[q_i^2] \cdot E[\epsilon_i^2]$$

For Lloyd-Max, this means using weighted MSE in the optimization. Dimensions that queries load on heavily should get more precise quantization.

**Key insight**: This is equivalent to transforming the embedding space by $D = \text{diag}(\sqrt{E[q_i^2]})$ before quantization, then inverse-transforming after reconstruction. The query weighting is absorbed into the per-dimension scale.

**Practical concern**: Requires a representative query sample during calibration. For Vespa, this means using historical queries — available in production but not always at initial deployment.

### Idea 4: 2-bit binary codes with Hamming distance

Current 2-bit methods (Lloyd-Max, quaternary) require float reconstruction at search time. What if we could do fast hardware-accelerated search with 2-bit codes, like Hamming distance for binary?

**Gray coding**: Encode the 4 levels as 2-bit Gray codes: {00, 01, 11, 10}. Then Hamming distance between two 2-bit codes approximates the distance between their values:
- Adjacent levels → Hamming distance 1
- Opposite levels → Hamming distance 2

This allows using bitwise XOR + popcount (hardware-accelerated) for approximate search with 2-bit codes, potentially much faster than float reconstruction.

**Trade-off**: Hamming distance on 2-bit codes is a cruder approximation than full asymmetric reconstruction. But for a first-stage retrieval (like our binary_rescore), it could be fast enough to then rescore with float queries.

### Idea 5: Calibration-free 2-bit quantization

For maximum streaming robustness (zero drift), can we design a 2-bit quantizer that needs no calibration?

For L2-normalized vectors, each dimension is bounded in $[-1, 1]$ and has expected value 0. A "universal" quantizer:

```
boundaries: {-0.5, 0, +0.5}     (or scaled by 1/sqrt(d))
levels:     {-0.75, -0.25, +0.25, +0.75}
```

These don't assume Gaussian, just symmetry around 0. The boundaries at ±0.5 are a reasonable split for a distribution on [-1, 1] that's peaked at 0.

**Prediction**: This will be worse than Lloyd-Max (which adapts to the actual distribution) but better than binary (which discards magnitude entirely). The question is how close to Lloyd-Max it gets.

---

## Part V: Experimental Plan

### Priority experiments (highest value per effort):

1. **Per-dimension distribution analysis**: Compute kurtosis, skewness, median offset for each model × dim. This is pure analysis (no new quantization code) and directly answers Q2 and Q8.

2. **Bootstrap confidence intervals**: Resample queries 1000× and compute CI for each method's NDCG@10. Answers Q6 (is Lloyd-Max > float32 significant?).

3. **Correlation structure**: Compute correlation matrix of corpus embeddings. If near-diagonal, scalar quantization is near-optimal (answers Q5).

4. **Median offset prediction**: Compute $\|m\|^2/d$ per model × dim, plot against the NDCG improvement from median centering. If it's a clean predictor, we have a model-selection rule (answers Q3).

### Secondary experiments (new methods):

5. **4-bit Lloyd-Max**: Implement 16-level Gaussian-optimal quantizer. Test if it closes the gap to int8.

6. **Adaptive bit allocation**: 4-bit for first 64 dims, 2-bit for rest. Compare against uniform 2-bit at the same total byte budget.

7. **Calibration-free 2-bit**: Implement fixed boundaries {-c/sqrt(d), 0, +c/sqrt(d)} and compare against Lloyd-Max. If competitive, it's a simpler deployment option.

### Analysis-only (no new code):

8. **Rate-distortion gap**: Compute the theoretical minimum MSE at 1 and 2 bits/dim for each dimension's empirical distribution. Compare against our achieved MSE. How close are we to the bound?

9. **Ranking vs MSE**: On existing results, compute Kendall's tau between quantized and float32 score vectors. Is rank correlation always monotonic with MSE, or do they diverge?

---

## Part VI: Summary of What We Can State Precisely

### Proven / well-established:

1. Asymmetric scoring has strictly less error than symmetric (one error term vs three)
2. Lloyd-Max is the unique MSE-optimal scalar quantizer for a given distribution
3. For Gaussian distributions, the 4-level Lloyd-Max constants are known exactly
4. Binary median is the 1-bit Lloyd-Max quantizer for symmetric distributions
5. At high bit widths (8+), uniform and Lloyd-Max quantization give nearly identical MSE
6. Rate-distortion theory gives fundamental lower bounds on MSE at any bit rate
7. L2-normalized embedding dimensions converge to Gaussian marginals for large $d$ (CLT on sphere)

### Empirically observed but not proven:

1. Lloyd-Max beats quartile-based quantization at 2 bits (consistent across all tests, expected from theory)
2. Median centering helps more at truncated dimensions (observed, hypothesis about renormalization-induced skew)
3. Rotation doesn't help for Matryoshka-trained models (observed, hypothesis about already-balanced dimensions)
4. Lloyd-Max can exceed float32 on some model/dataset combos (observed, regularization hypothesis)
5. NFCorpus is harder to quantize than SciFact (observed, graded relevance hypothesis)

### Unknown:

1. Whether MSE-optimal implies ranking-optimal (no proof either way)
2. The per-dimension distribution shape for specific models (not yet measured)
3. Whether non-uniform bit allocation across Matryoshka dims helps (not yet tested)
4. Whether inter-dimension correlations are large enough for PQ to beat scalar quantization (not yet measured)
5. Whether there exist calibration-free 2-bit quantizers competitive with Lloyd-Max (not yet tested)
6. Information-theoretic bounds on ranking preservation at a given bit rate (no known results)
