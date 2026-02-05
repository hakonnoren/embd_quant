# Embedding Quantization Experiments: Methodology

This document describes the techniques, metrics, and experimental setup used in this test lab.

## Table of Contents

1. [Overview](#overview)
2. [Embedding Models](#embedding-models)
3. [Quantization Techniques](#quantization-techniques)
4. [Matryoshka Representation Learning](#matryoshka-representation-learning)
5. [Combining Matryoshka + Quantization](#combining-matryoshka--quantization)
6. [Evaluation Datasets](#evaluation-datasets)
7. [Metrics](#metrics)
8. [Search Implementation](#search-implementation)
9. [References](#references)

---

## Overview

Dense embedding models map text to high-dimensional vectors (384-1024 dimensions) for semantic similarity search. While effective, storing and searching these embeddings at scale is expensive:

| Vectors | Dimensions | Float32 Storage |
|---------|------------|-----------------|
| 1M | 1024 | 4 GB |
| 10M | 1024 | 40 GB |
| 100M | 1024 | 400 GB |

This test lab evaluates techniques to reduce storage and improve search speed while maintaining retrieval quality:

1. **Scalar Quantization (int8)**: 4x compression
2. **Binary Quantization**: 32x compression
3. **Matryoshka Dimension Reduction**: Variable compression (2x-16x)
4. **Combined approaches**: Up to 512x compression

---

## Embedding Models

### mixedbread-ai/mxbai-embed-large-v1

- **Dimensions**: 1024
- **Parameters**: ~335M
- **Special training**: Trained with both MRL (Matryoshka) and BQL (Binary Quantization Learning)
- **Query prefix**: `"Represent this sentence for searching relevant passages: "`
- **Matryoshka dims**: 1024, 512, 256, 128, 64

This model is specifically optimized for quantization, making it ideal for testing compression techniques.

### nomic-ai/nomic-embed-text-v1.5

- **Dimensions**: 768
- **Parameters**: ~137M
- **Special training**: Native Matryoshka support
- **Query prefix**: `"search_query: "`
- **Document prefix**: `"search_document: "`
- **Matryoshka dims**: 768, 512, 256, 128, 64

### sentence-transformers/all-MiniLM-L6-v2

- **Dimensions**: 384
- **Parameters**: ~22M
- **Special training**: None (baseline model)
- **Matryoshka support**: No

Serves as a baseline to compare how models without quantization-aware training perform.

---

## Quantization Techniques

### Float32 (Baseline)

Standard 32-bit floating point representation.

```
Storage: 4 bytes per dimension
1024-dim vector = 4096 bytes
```

### Scalar Quantization (int8)

Maps continuous float32 values to discrete int8 values (-128 to 127).

**Process:**
1. Compute min/max values per dimension from a calibration dataset
2. Linearly map float32 range to 256 buckets (int8 range)
3. Store as int8, convert back to float32 for search

```python
# Quantization formula
scale = (max_val - min_val) / 255
quantized = round((value - min_val) / scale) - 128

# Dequantization
value = (quantized + 128) * scale + min_val
```

**Properties:**
- **Compression**: 4x (1 byte vs 4 bytes per dimension)
- **Quality retention**: ~97-99%
- **Search**: Dot product on dequantized values
- **Requires**: Calibration dataset for min/max ranges

### Binary Quantization

Converts float32 values to single bits by thresholding at zero.

**Process:**
1. For each dimension: if value > 0, bit = 1, else bit = 0
2. Pack 8 bits into 1 byte using `np.packbits`

```python
# Quantization
binary = (embeddings > 0).astype(np.uint8)
packed = np.packbits(binary, axis=1)

# Example: 1024 dims → 128 bytes
```

**Properties:**
- **Compression**: 32x (1 bit vs 32 bits per dimension)
- **Quality retention**: ~92% without rescoring, ~96% with rescoring
- **Search**: Hamming distance (XOR + popcount)
- **Speed**: 20-40x faster than float32 (with optimized implementation)

### Binary with Rescoring

Two-stage retrieval to recover quality lost in binary quantization:

1. **Stage 1**: Binary search to retrieve `k × rescore_multiplier` candidates
2. **Stage 2**: Rescore candidates using float32 query embeddings

```python
# Stage 1: Fast binary retrieval (e.g., top 400 for k=100)
candidates = binary_search(binary_query, binary_corpus, k=400)

# Stage 2: Precise rescoring with float32
scores = float_query @ float_corpus[candidates].T
top_k = candidates[argsort(scores)[:100]]
```

**Properties:**
- **Compression**: 32x (binary index in memory)
- **Quality retention**: ~96-100%
- **Trade-off**: Requires float32 corpus on disk for rescoring

---

## Matryoshka Representation Learning

Named after Russian nesting dolls, Matryoshka embeddings are trained so that the first `k` dimensions form a valid, useful embedding.

### How It Works

During training, the model optimizes multiple loss functions simultaneously:

```
Loss = L(full_dim) + L(dim/2) + L(dim/4) + L(dim/8) + ...
```

This ensures the most important information is packed into the first dimensions.

### Using Matryoshka Embeddings

Simply truncate to desired dimension and re-normalize:

```python
def truncate_matryoshka(embeddings, target_dim):
    truncated = embeddings[:, :target_dim]
    # Re-normalize for cosine similarity
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / norms
```

### Dimension vs Quality Trade-off

For mxbai-embed-large-v1 (trained with MRL):

| Dimensions | Compression | Typical Quality |
|------------|-------------|-----------------|
| 1024 | 1x | 100% |
| 512 | 2x | ~99% |
| 256 | 4x | ~97% |
| 128 | 8x | ~93% |
| 64 | 16x | ~85% |

---

## Combining Matryoshka + Quantization

The techniques are orthogonal and can be combined:

### Example: 256-dim + Binary

```
Original: 1024 dims × 4 bytes = 4096 bytes
Matryoshka: 256 dims × 4 bytes = 1024 bytes (4x reduction)
Binary: 256 dims × 1 bit = 32 bytes (128x total reduction)
```

### Compression Matrix

| Matryoshka Dim | Float32 | Int8 | Binary |
|----------------|---------|------|--------|
| 1024 | 1x | 4x | 32x |
| 512 | 2x | 8x | 64x |
| 256 | 4x | 16x | 128x |
| 128 | 8x | 32x | 256x |
| 64 | 16x | 64x | 512x |

### Quality Expectations (mxbai-embed-large-v1)

Based on the model being trained with both MRL and BQL:

| Configuration | Expected Quality |
|---------------|------------------|
| 1024d float32 | 100% (baseline) |
| 1024d binary+rescore | ~96% |
| 512d float32 | ~99% |
| 256d binary+rescore | ~90% |
| 64d binary+rescore | ~80% |

---

## Evaluation Datasets

We use retrieval datasets from the MTEB (Massive Text Embedding Benchmark):

### nfcorpus

- **Domain**: Medical/nutrition
- **Corpus**: 3,633 documents
- **Queries**: 323 test queries
- **Relevance**: Graded (0-2)
- **Challenge**: Domain-specific terminology

### scifact

- **Domain**: Scientific claims
- **Corpus**: 5,183 documents
- **Queries**: 300 test queries
- **Relevance**: Binary
- **Challenge**: Fact verification

### fiqa

- **Domain**: Financial Q&A
- **Corpus**: 57,638 documents
- **Queries**: 648 test queries
- **Relevance**: Binary
- **Challenge**: Larger corpus, domain-specific

---

## Metrics

### Recall@k

Measures how many relevant documents are retrieved in the top-k results.

```
Recall@k = |Retrieved@k ∩ Relevant| / |Relevant|
```

- **Recall@10**: Are the relevant docs in the top 10?
- **Recall@100**: Are they in the top 100? (useful for re-ranking pipelines)

### NDCG@k (Normalized Discounted Cumulative Gain)

Measures ranking quality, giving more weight to relevant documents ranked higher.

```
DCG@k = Σ (relevance_i / log2(i + 1))  for i in 1..k
NDCG@k = DCG@k / IDCG@k  (normalized by ideal ranking)
```

- Accounts for graded relevance (not just binary)
- Penalizes relevant docs ranked lower
- Range: 0 to 1 (higher is better)

### Latency

Wall-clock time for searching all queries against the corpus.

### Memory

Storage required for the corpus embeddings:

```
Memory = n_vectors × dimensions × bytes_per_value
```

### Compression Ratio

```
Compression = baseline_bytes / actual_bytes
```

Where baseline is float32 at full dimension.

---

## Search Implementation

### Float32 / Int8 Search

Brute-force dot product (cosine similarity for normalized vectors):

```python
similarities = query_embeddings @ corpus_embeddings.T
top_k_indices = argpartition(-similarities, k)[:k]
```

### Binary Search (Hamming Distance)

For packed binary vectors, Hamming distance counts differing bits:

```python
# XOR finds differing bits, popcount counts them
xor_result = np.bitwise_xor(query, corpus)
distances = np.sum(np.unpackbits(xor_result, axis=1), axis=1)
```

Lower Hamming distance = more similar.

**Note**: Our Python implementation is naive. Production systems use:
- FAISS with optimized Hamming distance
- SIMD instructions (AVX2/AVX-512)
- GPU acceleration

These achieve 20-40x speedup over float32 search.

---

## References

1. **Embedding Quantization** (Hugging Face Blog)
   https://huggingface.co/blog/embedding-quantization

2. **Matryoshka Representation Learning**
   Kusupati et al., 2022
   https://arxiv.org/abs/2205.13147

3. **Combining Matryoshka with Binary Quantization** (Vespa Blog)
   https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/

4. **MTEB: Massive Text Embedding Benchmark**
   Muennighoff et al., 2023
   https://huggingface.co/spaces/mteb/leaderboard

5. **Sentence Transformers Documentation**
   https://www.sbert.net/

6. **mixedbread-ai/mxbai-embed-large-v1**
   https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
