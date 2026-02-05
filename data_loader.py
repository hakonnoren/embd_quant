"""MTEB dataset loading utilities with caching."""
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
from datasets import load_dataset
from tqdm import tqdm


class MTEBDataLoader:
    """Load MTEB retrieval datasets in BeIR-like format."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, dataset_name: str) -> Tuple[Dict, Dict, Dict]:
        """
        Load corpus, queries, and qrels for a dataset.

        Returns:
            corpus: Dict[str, Dict] with 'title' and 'text' fields
            queries: Dict[str, str] mapping query_id to text
            qrels: Dict[str, Dict[str, int]] mapping query_id -> doc_id -> relevance
        """
        cache_path = self.cache_dir / f"{dataset_name}_data.pkl"

        if cache_path.exists():
            print(f"Loading cached {dataset_name} from {cache_path}")
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
        qrels_ds = load_dataset(
            f"mteb/{dataset_name}", "default", split="test", trust_remote_code=True
        )
        qrels = {}
        for row in qrels_ds:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = int(row["score"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

        # Cache for future use
        with open(cache_path, "wb") as f:
            pickle.dump((corpus, queries, qrels), f)

        print(f"Cached {dataset_name} to {cache_path}")
        return corpus, queries, qrels

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
