"""MTEB dataset loading utilities with caching."""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, List
from datasets import load_dataset
from tqdm import tqdm


class MTEBDataLoader:
    """Load MTEB retrieval datasets in BeIR-like format."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(
        self, dataset_name: str, subsample: Optional[int] = None, seed: int = 42
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Load corpus, queries, and qrels for a dataset.

        Args:
            dataset_name: MTEB dataset name (must match mteb/{name} on HuggingFace)
            subsample: If set, subsample the corpus to this many documents.
                All documents referenced by qrels are always retained so that
                metrics remain valid. Additional documents are sampled randomly
                to fill up to the requested size.
            seed: Random seed for reproducible subsampling.

        Returns:
            corpus: Dict[str, Dict] with 'title' and 'text' fields
            queries: Dict[str, str] mapping query_id to text
            qrels: Dict[str, Dict[str, int]] mapping query_id -> doc_id -> relevance
        """
        suffix = f"_sub{subsample}" if subsample else ""
        cache_path = self.cache_dir / f"{dataset_name}{suffix}_data.pkl"

        if cache_path.exists():
            print(f"Loading cached {dataset_name}{suffix} from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print(f"Downloading {dataset_name} from HuggingFace...")

        # Load corpus
        corpus_ds = load_dataset(
            f"mteb/{dataset_name}", "corpus", split="corpus", trust_remote_code=True
        )
        corpus = {}
        for row in tqdm(corpus_ds, desc=f"Loading {dataset_name} corpus"):
            corpus[row["_id"]] = {
                "title": row.get("title", ""),
                "text": row["text"],
            }

        # Load queries
        queries_ds = load_dataset(
            f"mteb/{dataset_name}", "queries", split="queries", trust_remote_code=True
        )
        queries = {row["_id"]: row["text"] for row in queries_ds}

        # Load qrels (test split for evaluation)
        # For MSMARCO, use 'dev' as 'test' is hidden. For others, try 'test' then 'dev' or 'default'.
        if dataset_name == "msmarco":
            split_name = "dev"
        else:
            split_name = "test"
            
        try:
            qrels_ds = load_dataset(
                f"mteb/{dataset_name}", "default", split=split_name, trust_remote_code=True
            )
        except Exception:
             # Fallback if test doesn't exist (e.g. some datasets only have train/dev)
            qrels_ds = load_dataset(
                f"mteb/{dataset_name}", "default", split="dev", trust_remote_code=True
            )

        qrels = {}
        for row in qrels_ds:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = int(row["score"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

        # Subsample corpus if requested
        if subsample and len(corpus) > subsample:
            corpus = self._subsample_corpus(corpus, qrels, subsample, seed)

        # Cache for future use
        with open(cache_path, "wb") as f:
            pickle.dump((corpus, queries, qrels), f)

        print(f"Cached {dataset_name}{suffix} to {cache_path} ({len(corpus)} docs)")
        return corpus, queries, qrels

    @staticmethod
    def _subsample_corpus(
        corpus: Dict, qrels: Dict, n: int, seed: int
    ) -> Dict:
        """Subsample corpus to n documents, always keeping qrel-referenced docs."""
        # Collect all doc IDs referenced by qrels
        required_ids: Set[str] = set()
        for qid_rels in qrels.values():
            required_ids.update(qid_rels.keys())
        # Keep only those that exist in the corpus
        required_ids &= set(corpus.keys())

        if len(required_ids) >= n:
            print(f"  Subsampling: qrels reference {len(required_ids)} docs "
                  f"(>= requested {n}), keeping all qrel docs only")
            return {did: corpus[did] for did in required_ids}

        # Fill remaining slots with random non-qrel documents
        other_ids = list(set(corpus.keys()) - required_ids)
        n_extra = n - len(required_ids)
        rng = np.random.default_rng(seed)
        sampled_ids = rng.choice(other_ids, size=min(n_extra, len(other_ids)), replace=False)

        keep_ids = required_ids | set(sampled_ids.tolist())
        subsampled = {did: corpus[did] for did in corpus if did in keep_ids}
        print(f"  Subsampled corpus: {len(corpus)} -> {len(subsampled)} docs "
              f"({len(required_ids)} from qrels + {len(sampled_ids)} random)")
        return subsampled

    def get_texts_for_embedding(
        self, corpus: Dict, queries: Dict
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Extract ordered lists for embedding generation.

        Returns:
            doc_ids: List of document IDs
            doc_texts: List of document texts (title + text)
            query_ids: List of query IDs
            query_texts: List of query texts
        """
        doc_ids = list(corpus.keys())
        doc_texts = [
            f"{corpus[did]['title']} {corpus[did]['text']}".strip() for did in doc_ids
        ]

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        return doc_ids, doc_texts, query_ids, query_texts
