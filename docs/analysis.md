# Quantization Experiment Analysis

Datasets: SciFact (~5K docs), NFCorpus (~3.6K docs)
Models: MxBai-embed-large-v1 (1024d), Nomic-embed-text-v1.5 (768d), all-MiniLM-L6-v2 (384d)

## 1. Asymmetric Distance Consistently Helps

Keeping the query unquantized while only quantizing the corpus is a **clear win across the board**.

### binary → binary_asym (NDCG@10, full dim)

| Model    | Dataset  | Rotation | Binary  | Binary Asym | Change  |
|----------|----------|----------|---------|-------------|---------|
| MxBai    | scifact  | none     | 0.8351  | 0.8414      | +0.8%   |
| MxBai    | scifact  | hadamard | 0.8218  | 0.8356      | +1.7%   |
| MxBai    | nfcorpus | none     | 0.4886  | 0.4956      | +1.4%   |
| MxBai    | nfcorpus | hadamard | 0.4821  | 0.4931      | +2.3%   |
| Nomic    | scifact  | none     | 0.8096  | 0.8183      | +1.1%   |
| Nomic    | nfcorpus | none     | 0.4742  | 0.4861      | +2.5%   |
| MiniLM   | scifact  | none     | 0.8001  | 0.8268      | +3.3%   |
| MiniLM   | nfcorpus | none     | 0.4029  | 0.4380      | **+8.7%** |

Recall@10 gains are even larger: up to **+16-18%** for binary_asym on weaker models / harder datasets.

### int8 → int8_asym (NDCG@10, full dim)

| Model    | Dataset  | Rotation | Int8    | Int8 Asym | Change  |
|----------|----------|----------|---------|-----------|---------|
| MxBai    | scifact  | none     | 0.8365  | 0.8382    | +0.2%   |
| MxBai    | nfcorpus | none     | 0.4963  | 0.5024    | +1.2%   |
| Nomic    | scifact  | none     | 0.8113  | 0.8246    | +1.6%   |
| Nomic    | nfcorpus | none     | 0.4832  | 0.4858    | +0.5%   |
| MiniLM   | scifact  | none     | 0.7959  | 0.8188    | +2.9%   |
| MiniLM   | nfcorpus | none     | 0.4564  | 0.4569    | +0.1%   |

Smaller gains since int8 already has less quantization error, but always positive.

## 2. Rotation is Surprisingly Unhelpful

Rotation (Hadamard/QR) was expected to help binary quantization by distributing information more uniformly across dimensions, but results are mixed:

- **NDCG**: rotation is a coin flip — sometimes +1.7%, sometimes -2.5%. No consistent direction.
- **Recall**: rotation tends to **hurt** for larger models (MxBai, Nomic), typically -2 to -3%.
- **Int8**: rotation has near-zero effect (<1% either way).

The models may already have reasonably distributed embeddings, so rotation adds noise without benefit.

## 3. binary_asym vs binary_rescore — A Near-Tie on NDCG

| Model  | Dataset  | binary_asym | binary_rescore | float32 |
|--------|----------|-------------|----------------|---------|
| MxBai  | scifact  | 0.8414      | 0.8429         | 0.8429  |
| MxBai  | nfcorpus | 0.4956      | 0.5018         | 0.5018  |
| Nomic  | scifact  | 0.8183      | 0.8263         | 0.8241  |
| Nomic  | nfcorpus | 0.4861      | 0.4777         | 0.4785  |
| MiniLM | scifact  | **0.8268**  | 0.8132         | 0.8131  |
| MiniLM | nfcorpus | 0.4380      | **0.4622**     | 0.4577  |

- **binary_rescore** wins on **recall** (sees full float32 corpus during rescoring)
- **binary_asym** is competitive or better on **NDCG** for some combos
- binary_asym is **~20x faster** and needs no float32 corpus copy in memory

## 4. Best Configs Per Model/Dataset (excluding float32)

| Model  | Dataset  | Best Config                 | NDCG@10 | % of float32 |
|--------|----------|-----------------------------|---------|---------------|
| MxBai  | scifact  | binary_rescore, rot=none    | 0.8429  | 100.0%        |
| MxBai  | nfcorpus | binary_rescore, rot=hadamard| 0.5028  | 100.2%        |
| Nomic  | scifact  | binary_rescore, rot=none    | 0.8263  | 100.3%        |
| Nomic  | nfcorpus | binary_asym, rot=none       | 0.4861  | 101.6%        |
| MiniLM | scifact  | binary_asym, rot=none       | 0.8268  | 101.7%        |
| MiniLM | nfcorpus | binary_rescore, rot=none    | 0.4622  | 101.0%        |

## Recommendations

- **int8_asym** is the safest bet: near-float32 quality at 4x compression, no rotation needed.
- **binary_asym without rotation** gives ~32x compression with surprisingly good NDCG retention.
- Rotation does not provide consistent benefit and can be skipped.
