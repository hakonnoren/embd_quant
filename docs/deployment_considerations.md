# Deployment Considerations for Quantization Methods

## 1. Calibration Drift in Streaming Systems

In a streaming vector database like Vespa, documents arrive continuously. Calibration statistics (medians, stds, boundaries) are computed once from an initial corpus sample and then used to quantize all future documents. If the corpus distribution shifts over time, these statistics become stale.

### How each method handles drift

| Method | Calibration stats | Drift sensitivity | Why |
|---|---|---|---|
| `binary_asym` | None (threshold = 0) | **Immune** | No calibration needed at all |
| `binary_median_asym` | Per-dim median (d floats) | **Low** | Median only affects the binarization threshold. A drifted median means slightly unbalanced bits (not exactly 50/50), but the method degrades gracefully |
| `lloyd_max_gauss` | Per-dim median + std (2d floats) | **Moderate** | The reconstruction levels are fixed constants (not data-driven). Only the standardization (centering + scaling) drifts. A stale std means the boundary bins are slightly misplaced |
| `int8_asym` | Per-dim min + max (2d floats) | **Moderate** | New documents with values outside the calibration range get clipped to 0 or 255, losing information in the tails |
| `quaternary_asym` | Quartile boundaries + centroids (7d floats) | **Worst** | Both boundaries AND reconstruction centroids are fully data-driven. The centroids directly enter the score computation, so stale centroids directly corrupt ranking scores |

### Why drift is less scary than it sounds

For L2-normalized embeddings from a fixed model, the per-dimension distribution is largely determined by the model weights, not the corpus content. The distribution of `dim[42]` across English text documents is remarkably stable whether you calibrate on 10K or 10M documents. This is because:

1. L2 normalization constrains values to a hypersphere
2. The model maps semantically diverse inputs to a consistent statistical structure
3. Individual dimensions represent learned features that have stable marginal distributions

A calibration set of ~10K diverse documents should give statistics that hold well for millions of additional documents, as long as the domain doesn't shift dramatically (e.g., switching from English to Chinese).

### Mitigation strategies

- **Periodic recalibration**: Recompute stats from a fresh sample, re-quantize the index. For Lloyd-Max this means updating 2 vectors; for quaternary, 7 vectors.
- **Conservative ranges for int8**: Use percentile-based ranges (1st-99th) instead of min/max to be robust to outliers in future documents.
- **Online monitoring**: Track the fraction of new documents that fall outside calibrated boundaries. If it exceeds a threshold, trigger recalibration.

### Memory footprint of calibration metadata

All stats are **global per-dimension** — stored once, not per document:

| Method | Global metadata | Per-vector storage | Example: d=1024 |
|---|---|---|---|
| `binary_asym` | 0 | d/8 bytes | 0 + 128 B/vec |
| `binary_median_asym` | d × 4 bytes | d/8 bytes | 4 KB + 128 B/vec |
| `lloyd_max_gauss` | 2d × 4 bytes | d/4 bytes | 8 KB + 256 B/vec |
| `int8_asym` | 2d × 4 bytes | d bytes | 8 KB + 1024 B/vec |
| `quaternary_asym` | 7d × 4 bytes | d/4 bytes | 28 KB + 256 B/vec |

For any realistic corpus (>1000 documents), the global metadata is negligible compared to the per-vector storage.

---

## 2. A Unified View: Lloyd-Max at Every Bit Width

### The observation

Our quantization methods at different bit widths use different design principles:

| Bit width | Method | Design principle |
|---|---|---|
| 1-bit | `binary_asym` | Threshold at 0 (arbitrary) |
| 1-bit | `binary_median_asym` | Threshold at corpus median |
| 2-bit | `quaternary_asym` | Equal-count quartile boundaries + empirical centroids |
| 2-bit | `lloyd_max_gauss` | MSE-optimal boundaries + MSE-optimal levels for Gaussian |
| 8-bit | `int8_asym` | Uniform linear spacing over [min, max] |

This is inconsistent. We use MSE-optimal (Lloyd-Max) for 2-bit but not for 1-bit or 8-bit. Can we unify?

### Lloyd-Max at 1 bit: we already do it

The 1-bit Lloyd-Max quantizer for a symmetric distribution places its boundary at the **mean** (or equivalently, median for symmetric distributions). The two reconstruction levels are the conditional means E[x | x > median] and E[x | x < median].

Our `binary_median_asym` is exactly the 1-bit Lloyd-Max quantizer:
- **Boundary**: median (= Lloyd-Max optimal for symmetric distributions)
- **Reconstruction**: ±1 signs (we don't use conditional-mean levels, but in asymmetric search the query handles the weighting implicitly)

So `binary_median_asym` IS the Lloyd-Max principle applied to 1 bit. We just don't call it that.

The `binary_asym` (threshold at 0) is the "naive uniform" version — analogous to how linear int8 is the "naive uniform" version at 8 bits.

### Lloyd-Max at 8 bits: diminishing returns

Could we apply the Lloyd-Max principle to 8-bit quantization? Yes — the Gaussian-optimal 256-level quantizer has known (tabulated) boundaries and reconstruction levels. But the improvement over uniform linear quantization is **negligible**:

| Levels | Uniform SNR (dB) | Lloyd-Max SNR (dB) | Improvement |
|---|---|---|---|
| 4 (2-bit) | ~9.3 | ~9.3 | ~0 dB for uniform range, but **large** when you account for non-uniform density |
| 16 (4-bit) | ~24.6 | ~24.6 | <0.1 dB |
| 256 (8-bit) | ~50 | ~50 | <0.01 dB |

The reason: with 256 levels, both uniform and optimal spacing approximate the distribution well. Each level covers such a tiny slice of the distribution that the placement barely matters. The error is dominated by the fundamental resolution limit (Δ/√12 per dimension), not by suboptimal bin placement.

**More precisely**: for a Gaussian source, the MSE of uniform quantization over ±3σ with 256 levels is approximately σ²·(6/256)²/12 ≈ 4.6×10⁻⁵ σ². The Lloyd-Max optimal is ≈ 4.5×10⁻⁵ σ². The difference is ~2%, which is invisible in NDCG.

### Why 2-bit is the sweet spot for Lloyd-Max

At 2 bits (4 levels), each level covers ~25% of the probability mass. Placing boundaries to minimize MSE vs placing them at equal-count percentiles makes a real difference because:

1. **Tail sensitivity**: The outer bins span a much wider value range. Lloyd-Max places them at ±0.9816σ instead of ±0.6745σ (quartiles), giving the tails more room.
2. **Reconstruction accuracy**: The fixed levels (±0.4528σ, ±1.5104σ) are MSE-optimal, while quartile centroids are only optimal for the empirical distribution (overfitting to calibration data).
3. **Robustness**: Fixed constants don't drift (see Section 1). Empirical centroids do.

### The consistent framework

If we wanted full consistency, the methods would be:

| Bits | "Naive" (uniform/threshold=0) | Lloyd-Max optimal |
|---|---|---|
| 1 | `binary_asym` (t=0) | `binary_median_asym` (t=median) |
| 2 | `quaternary_asym` (equal-count) | `lloyd_max_gauss` (MSE-optimal) |
| 4 | uniform 16-level | Lloyd-Max 16-level (marginal gain) |
| 8 | `int8_asym` (uniform 256-level) | Lloyd-Max 256-level (negligible gain) |

The pattern: **Lloyd-Max matters most at low bit widths** where each level represents a large chunk of the distribution. At 8 bits, the uniform approach is already near-optimal.

### Should we implement Lloyd-Max int8?

**No.** The theoretical improvement is ~2% in MSE, which translates to <0.001 in NDCG — well within noise. The current linear int8 is fine. The complexity of non-uniform quantization (lookup tables for reconstruction, non-trivial encoding) isn't worth it at 256 levels.

The place where smarter quantization pays off is exactly where we've focused: **1-2 bits per dimension**, where the gap between naive and optimal is large.

### What about 4-bit?

4-bit (16 levels) is an interesting middle ground we haven't explored:
- Storage: d/2 bytes per vector (2× of 2-bit, 2× better than int8)
- Lloyd-Max 16-level would give a small but potentially measurable improvement over uniform 16-level
- Could be useful if 2-bit isn't quite good enough but int8 is overkill

This is a potential avenue for future work, though the 2-bit Lloyd-Max results are already very strong (often matching or exceeding int8).

---

## 3. Summary: Recommended Methods for Production

For a streaming vector database deployment:

| Priority | Method | Why |
|---|---|---|
| **Best quality/byte** | `lloyd_max_gauss` | MSE-optimal 2-bit, fixed reconstruction constants, only 2 calibration vectors that barely drift for normalized embeddings |
| **Ultra-compact** | `binary_median_asym` | 1-bit Lloyd-Max equivalent, only 1 calibration vector (median), extremely drift-robust |
| **Maximum quality** | `int8_asym` | 8-bit uniform is already near-optimal, no benefit from Lloyd-Max at this bit width |
| **Zero calibration** | `binary_asym` | No calibration at all, slightly worse than median variant but completely stateless |

Methods to **skip** in production:
- `quaternary_asym` — strictly worse than `lloyd_max_gauss` at the same bit budget, and more drift-sensitive (7 calibration vectors vs 2)
- `binary_rescore` — requires keeping the full float32 corpus in memory, defeating the purpose of compression
- Per-dimension weighting — no measurable improvement for L2-normalized embeddings
