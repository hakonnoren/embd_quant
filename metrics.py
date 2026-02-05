"""Evaluation metrics for retrieval."""
import numpy as np
from typing import Dict, List, Set


class RetrievalMetrics:
    """Compute retrieval evaluation metrics."""

    @staticmethod
    def recall_at_k(
        retrieved_indices: np.ndarray, relevant_docs: List[Set[int]], k: int
    ) -> float:
        """
        Compute Recall@k averaged over queries.

        Args:
            retrieved_indices: (n_queries, max_k) retrieved doc indices
            relevant_docs: List of sets, each containing relevant doc indices for a query
            k: Cutoff for recall computation

        Returns:
            Average recall@k across queries
        """
        recalls = []
        for i, rel_set in enumerate(relevant_docs):
            if len(rel_set) == 0:
                continue
            retrieved_set = set(retrieved_indices[i, :k].tolist())
            recall = len(retrieved_set & rel_set) / len(rel_set)
            recalls.append(recall)

        return float(np.mean(recalls)) if recalls else 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_indices: np.ndarray,
        qrels_idx: Dict[str, Dict[int, int]],
        query_ids: List[str],
        k: int,
    ) -> float:
        """
        Compute NDCG@k averaged over queries.

        Args:
            retrieved_indices: (n_queries, max_k) retrieved doc indices
            qrels_idx: Mapping query_id -> doc_idx -> relevance
            query_ids: List mapping row index to query ID
            k: Cutoff for NDCG

        Returns:
            Average NDCG@k across queries
        """
        ndcg_scores = []

        for i, qid in enumerate(query_ids):
            if qid not in qrels_idx or len(qrels_idx[qid]) == 0:
                continue

            # Get relevance scores for retrieved docs
            retrieved = retrieved_indices[i, :k]
            relevances = np.array(
                [qrels_idx[qid].get(int(did), 0) for did in retrieved]
            )

            if relevances.sum() == 0:
                # No relevant docs retrieved
                continue

            # Compute DCG
            dcg = RetrievalMetrics._dcg(relevances)

            # Compute ideal DCG
            ideal_relevances = sorted(qrels_idx[qid].values(), reverse=True)[:k]
            ideal_relevances = np.array(
                ideal_relevances + [0] * (k - len(ideal_relevances))
            )
            idcg = RetrievalMetrics._dcg(ideal_relevances)

            if idcg > 0:
                ndcg_scores.append(dcg / idcg)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    @staticmethod
    def _dcg(relevances: np.ndarray) -> float:
        """Compute Discounted Cumulative Gain."""
        positions = np.arange(1, len(relevances) + 1)
        return float(np.sum(relevances / np.log2(positions + 1)))

    @staticmethod
    def compute_all_metrics(
        retrieved_indices: np.ndarray,
        qrels: Dict[str, Dict[str, int]],
        doc_id_to_idx: Dict[str, int],
        query_ids: List[str],
        k_values: List[int],
    ) -> Dict[str, float]:
        """
        Compute all metrics for multiple k values.

        Args:
            retrieved_indices: (n_queries, max_k) retrieved doc indices
            qrels: Original qrels mapping query_id -> doc_id -> relevance
            doc_id_to_idx: Mapping from doc ID to index
            query_ids: List of query IDs in order
            k_values: List of k values to evaluate

        Returns:
            Dictionary with metric names and values
        """
        results = {}

        # Convert qrels to index-based format
        qrels_idx: Dict[str, Dict[int, int]] = {}
        for qid in query_ids:
            if qid in qrels:
                qrels_idx[qid] = {
                    doc_id_to_idx[did]: rel
                    for did, rel in qrels[qid].items()
                    if did in doc_id_to_idx
                }

        # Prepare relevant doc sets for recall
        relevant_docs: List[Set[int]] = []
        for qid in query_ids:
            if qid in qrels_idx:
                rel_indices = set(qrels_idx[qid].keys())
            else:
                rel_indices = set()
            relevant_docs.append(rel_indices)

        for k in k_values:
            results[f"recall@{k}"] = RetrievalMetrics.recall_at_k(
                retrieved_indices, relevant_docs, k
            )
            results[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(
                retrieved_indices, qrels_idx, query_ids, k
            )

        return results
