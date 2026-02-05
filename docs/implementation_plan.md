# Embedding Quantization Test Lab - Implementation Plan

## Overview

Build a modular Python test framework for evaluating embedding quantization techniques and Matryoshka representations using brute-force kNN search on MTEB retrieval datasets.

## Configuration

- **Format**: Python scripts
- **Datasets**: NFCorpus (~3.6K docs), SciFact (~5K docs), FiQA (~57K docs)
- **Models**:
  - `mixedbread-ai/mxbai-embed-large-v1` (1024 dims, MRL+BQL trained)
  - `nomic-ai/nomic-embed-text-v1.5` (768 dims, Matryoshka)
  - `sentence-transformers/all-MiniLM-L6-v2` (384 dims, baseline)
- **Quantization**: float32, int8, binary, binary+rescore, Matryoshka combinations
- **Metrics**: Recall@10/100, NDCG@10, latency, memory usage

## File Structure

```
quantization_tests/
├── config.py                # Configuration and model/dataset definitions
├── data_loader.py           # MTEB dataset loading with caching
├── embedder.py              # Embedding generation with caching
├── quantization.py          # Quantization utilities (int8, binary, Matryoshka)
├── search.py                # Brute-force kNN implementations
├── metrics.py               # Evaluation metrics (Recall, NDCG)
├── experiment_runner.py     # Orchestrates experiments
├── results_reporter.py      # Results aggregation and reporting
├── run_experiments.py       # Main entry point with CLI
├── requirements.txt         # Dependencies
├── cache/                   # Embedding cache (gitignored)
└── results/                 # Output results
```

## Implementation Steps

### 1. Create `config.py`
Define model configurations (name, dimensions, Matryoshka dims, query prefixes), dataset info, and experiment parameters (k values, rescore multiplier).

### 2. Create `data_loader.py`
- Load MTEB datasets from HuggingFace: `mteb/{dataset}/corpus`, `mteb/{dataset}/queries`, `mteb/{dataset}/default`
- Cache loaded data as pickle files
- Extract ordered lists for embedding generation

### 3. Create `embedder.py`
- Wrap SentenceTransformer with caching
- Save/load embeddings as numpy files
- Apply query prefixes for models that need them
- Normalize embeddings for cosine similarity

### 4. Create `quantization.py`
- **Binary**: Use `quantize_embeddings(embeddings, precision="ubinary")` - 32x compression
- **Int8**: Use `quantize_embeddings(embeddings, precision="int8", calibration_embeddings=...)` - 4x compression
- **Matryoshka**: Truncate to first k dims, then re-normalize
- Memory calculation helper

### 5. Create `search.py`
- **Float32**: `query @ corpus.T` dot product, argpartition for top-k
- **Int8**: Same as float32 but cast int8→float32 for computation
- **Binary**: XOR + popcount (unpackbits+sum) for Hamming distance
- **Binary+Rescore**: Two-stage - binary retrieval for 4×k candidates, then float32 rescore

### 6. Create `metrics.py`
- **Recall@k**: `|retrieved ∩ relevant| / min(|relevant|, k)`
- **NDCG@k**: Use sklearn.metrics.ndcg_score
- Convert qrels from doc_id→relevance to index-based format

### 7. Create `experiment_runner.py`
- Loop over models × datasets × quantizations × Matryoshka dims
- Cache embeddings per model/dataset
- Run search and compute metrics for each configuration
- Collect ExperimentResult dataclass objects

### 8. Create `results_reporter.py`
- Convert results to pandas DataFrame
- Compute compression ratios
- Generate summary tables, pivot tables
- Save to CSV, JSON, and Markdown

### 9. Create `run_experiments.py`
- CLI with argparse: `--models`, `--datasets`, `--quantizations`, `--no-matryoshka`, `--output-dir`
- Print progress and per-experiment summaries
- Save final results

### 10. Create `requirements.txt`
```
sentence-transformers>=2.2.0
datasets>=2.14.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0
torch>=2.0.0
tqdm>=4.65.0
tabulate>=0.9.0
```

## Key Technical Details

### Binary Quantization
- Threshold at 0: values > 0 → 1, else → 0
- Pack bits: 1024 dims → 128 bytes (32x compression)
- Hamming distance for search (XOR + popcount)
- Rescore with float32 improves accuracy from ~92% to ~96%

### Int8 Quantization
- Requires calibration data to compute per-dimension min/max
- 4x compression, ~99% accuracy retention
- Use corpus embeddings for calibration

### Matryoshka
- Simply truncate to first k dimensions
- mxbai-embed-large-v1: 1024 → 512, 256, 128, 64
- nomic-embed-text-v1.5: 768 → 512, 256, 128, 64
- Re-normalize after truncation

## Usage

```bash
# Run all experiments
python run_experiments.py

# Quick test on smallest dataset
python run_experiments.py --datasets NFCorpus --no-matryoshka

# Specific model and quantization
python run_experiments.py --models mxbai-embed-large-v1 --quantizations binary binary_rescore
```

## Expected Output

- Console: Progress bars, per-experiment metrics
- `results/results.csv`: Full results table
- `results/results.json`: Programmatic access
- `results/summary.md`: Markdown summary

## Verification

1. Run `python run_experiments.py --datasets NFCorpus --no-matryoshka` for quick test
2. Verify embeddings are cached in `cache/`
3. Verify results files are generated in `results/`
4. Check that float32 baseline gives reasonable recall (>0.5 for NFCorpus)
5. Verify compression ratios: int8=4x, binary=32x
