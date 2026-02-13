# Embedding Quantization Test Lab

A Python framework for evaluating embedding quantization techniques and Matryoshka representations using brute-force kNN search on MTEB retrieval datasets, with Vespa Cloud validation.

## Features

- **Multiple quantization schemes**: float32, int8, binary, 2-bit (Lloyd-Max, quaternary)
- **Matryoshka dimension reduction**: Truncated embeddings (1024 -> 512 -> 256 -> 128 -> 64)
- **Rescoring pipelines**: Binary retrieval + float32/int8/binary rescore
- **Residual rescoring**: Two-stage funnel with binary residual pruning
- **Asymmetric scoring**: Float queries x quantized documents for lower error
- **Median binarization**: Per-dimension calibrated thresholds
- **Vespa Cloud deployment**: End-to-end validation on Vespa Cloud
- **Caching**: Embeddings and datasets cached to avoid recomputation

## Project Structure

```
quantization_tests/
  config.py            # Models, datasets, method colors, experiment config
  data.py              # Data loading and embedding preparation
  data_loader.py       # MTEB dataset loader
  embedder.py          # Embedding model wrapper
  quantization.py      # Quantization methods (int8, binary, 2-bit, Lloyd-Max)
  search.py            # Brute-force kNN search (float32, binary hamming)
  rescore.py           # Rescoring strategies (float32, int8, binary, funnel)
  metrics.py           # Recall, NDCG evaluation
  rotation.py          # Rotational quantization (experimental)
  runner.py            # Experiment orchestration
  run.py               # CLI entry point

  analysis/            # Visualization and analysis scripts
    visualize.py       # Plot generation (dim sweeps, Pareto fronts)
    make_latex_table.py # LaTeX table generation with Pareto markers
    analyze_results.py # Results analysis utilities
    mem_table.py       # Memory comparison tables

  vespa/               # Vespa Cloud deployment
    vespa_cloud_deploy.py  # Deploy, feed, evaluate on Vespa Cloud
    binarysearch/          # Vespa application package

  notebooks/           # Jupyter notebooks for exploration
  docs/                # Documentation and presentation materials
  results/             # Experiment results (gitignored)
  cache/               # Cached embeddings (gitignored)
```

## Models

| Model | Dimensions | Matryoshka | Notes |
|-------|-----------|------------|-------|
| `mxbai-embed-large-v1` | 1024 | Yes | Trained with MRL+BQL |
| `nomic-embed-text-v1.5` | 768 | Yes | Good Matryoshka support |
| `all-MiniLM-L6-v2` | 384 | No | Baseline model |

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
# Run experiments (see run.py --help for all options)
python run.py --help

# Example: binary retrieval with float32 rescore on FiQA
python run.py --datasets fiqa --models mxbai-embed-large-v1 \
  --retrieval binary --rescore float32

# Generate plots from results
python analysis/visualize.py results/<experiment_id>/results.json

# Generate LaTeX table with compression ratios and Pareto markers
python analysis/make_latex_table.py results/<experiment_id>/results.json ndcg@100

# Deploy and evaluate on Vespa Cloud
python vespa/vespa_cloud_deploy.py --dataset FiQA --tenant <your-tenant>
```

## Output

Results are saved to `results/<experiment_id>/`:
- `results.json` - Full results with metrics per method/dim combination
- `plots/` - Dimension sweep plots, Pareto front visualizations

## Metrics

- **Recall@k**: Fraction of relevant documents retrieved in top-k
- **NDCG@10/100**: Normalized Discounted Cumulative Gain
- **Compression ratio**: vs float32 baseline (e.g. binary@512dim = 64x)
- **Quality retention**: method NDCG / baseline NDCG (%)
- **Latency**: Vespa search time (ms)

## References

- [Embedding Quantization (Hugging Face)](https://huggingface.co/blog/embedding-quantization)
- [Matryoshka + Binary Quantization (Vespa)](https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)

## License

MIT
