# Embedding Quantization Test Lab

A Python framework for evaluating embedding quantization techniques and Matryoshka representations using brute-force kNN search on MTEB retrieval datasets.

## Features

- **Multiple quantization schemes**: float32, int8 (4x compression), binary (32x compression)
- **Matryoshka dimension reduction**: Test truncated embeddings (1024→512→256→128→64)
- **Binary rescoring**: Two-stage retrieval with binary search + float32 rescoring
- **Caching**: Embeddings and datasets are cached to avoid recomputation
- **Multiple models**: Test different embedding models optimized for quantization

## Models

| Model | Dimensions | Matryoshka | Notes |
|-------|-----------|------------|-------|
| `mxbai-embed-large-v1` | 1024 | ✅ | Trained with MRL+BQL |
| `nomic-embed-text-v1.5` | 768 | ✅ | Good Matryoshka support |
| `all-MiniLM-L6-v2` | 384 | ❌ | Baseline model |

## Datasets

| Dataset | Corpus | Queries | Source |
|---------|--------|---------|--------|
| `nfcorpus` | 3,633 | 323 | MTEB |
| `scifact` | 5,183 | 300 | MTEB |
| `fiqa` | 57,638 | 648 | MTEB |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run all experiments
python run_experiments.py

# Quick test (smallest dataset, no Matryoshka)
python run_experiments.py --datasets nfcorpus --no-matryoshka

# Specific model and quantization
python run_experiments.py --models mxbai-embed-large-v1 --quantizations binary binary_rescore

# Single dataset with specific model
python run_experiments.py --datasets scifact --models all-MiniLM-L6-v2

# Named experiment
python run_experiments.py --experiment-id baseline_v1 --datasets nfcorpus
```

## Output

Results are saved to `results/`:
- `results.csv` - Full results table
- `results.json` - JSON format for programmatic access
- `summary.md` - Markdown summary

## Metrics

- **Recall@k**: Fraction of relevant documents retrieved in top-k
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Latency**: Search time in seconds
- **Memory**: Index size in MB
- **Compression ratio**: vs float32 baseline

## References

- [Embedding Quantization (Hugging Face)](https://huggingface.co/blog/embedding-quantization)
- [Matryoshka + Binary Quantization (Vespa)](https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)

## License

MIT
