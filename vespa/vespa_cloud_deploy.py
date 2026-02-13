"""Deploy cached embeddings to Vespa Cloud with multiple ranking schemes.

Supports binary hamming retrieval with float/int8 rescoring,
and Matryoshka dimension reduction variants.

Usage:
    python vespa_cloud_deploy.py --dataset NFCorpus --tenant vespa-team
    python vespa_cloud_deploy.py --dataset SciFact --tenant vespa-team --deploy-only
    python vespa_cloud_deploy.py --dataset NFCorpus --tenant vespa-team --evaluate-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import MODELS, DATASETS, OVERSAMPLE, K_VALUES, EmbeddingData
from data import load_data
from data_loader import MTEBDataLoader
from quantization import QuantizationHandler

# ---------------------------------------------------------------------------
# Schema & rank profile construction
# ---------------------------------------------------------------------------

from vespa.package import (
    ApplicationPackage,
    Document,
    Field,
    FieldSet,
    FirstPhaseRanking,
    Function,
    RankProfile,
    Schema,
    SecondPhaseRanking,
)



# Matryoshka binary dimensions: full_float_dim -> packed_int8_dim
MRL_BINARY_CONFIGS = {
    1024: 128,  # 1024 bits -> 128 int8
    512: 64,    # 512 bits -> 64 int8
    256: 32,    # 256 bits -> 32 int8
    128: 16,    # 128 bits -> 16 int8
    64: 8,      # 64 bits -> 8 int8
}

# MRL float dimensions for brute-force baselines (1024 is already float_vector)
MRL_FLOAT_DIMS = [512, 256, 128, 64]


def build_schema(rerank_count: int = 100) -> Schema:
    """Build Vespa schema with binary vectors at multiple MRL dims, int8, and full float."""
    fields = [
        Field(
            name="doc_id",
            type="string",
            indexing=["summary", "attribute"],
        ),
        Field(
            name="dataset",
            type="string",
            indexing=["summary", "attribute", "index"],
            attribute=["fast-search"],
        ),
        Field(
            name="text",
            type="string",
            indexing=["summary", "index"],
            index="enable-bm25",
        ),
        # Full float vector — paged attribute for rescoring only
        Field(
            name="float_vector",
            type="tensor<float>(x[1024])",
            indexing=["attribute"],
            attribute=["paged", "distance-metric: prenormalized-angular"],
        ),
        # Int8 vector — paged attribute for int8 rescoring
        Field(
            name="int8_vector",
            type="tensor<int8>(x[1024])",
            indexing=["attribute"],
            attribute=["paged"],
        ),
    ]

    # MRL float vectors — paged attributes for brute-force baselines
    for mrl_dim in MRL_FLOAT_DIMS:
        fields.append(
            Field(
                name=f"float_{mrl_dim}",
                type=f"tensor<float>(x[{mrl_dim}])",
                indexing=["attribute"],
                attribute=["paged"],
            )
        )

    # Binary vectors at each MRL dimension with HNSW hamming index
    for float_dim, packed_dim in MRL_BINARY_CONFIGS.items():
        fields.append(
            Field(
                name=f"binary_{float_dim}",
                type=f"tensor<int8>(x[{packed_dim}])",
                indexing=["attribute", "index"],
                attribute=[f"distance-metric: hamming"],
            )
        )

    schema = Schema(
        name="doc",
        mode="index",
        document=Document(fields=fields),
        fieldsets=[FieldSet(name="default", fields=["text"])],
    )

    # =====================================================================
    # Rank profiles
    # =====================================================================

    # -- Binary-only (no rescore) --

    schema.add_rank_profile(
        RankProfile(
            name="binary-only",
            inputs=[("query(q_binary_1024)", "tensor<int8>(x[128])")],
            first_phase=FirstPhaseRanking(
                expression="closeness(field, binary_1024)"
            ),
            match_features=["distance(field, binary_1024)"],
        )
    )

    # -- Binary retrieval → rescore variants (full 1024-dim retrieval) --

    schema.add_rank_profile(
        RankProfile(
            name="binary-rescore-unpack",
            inputs=[
                ("query(q_binary_1024)", "tensor<int8>(x[128])"),
                ("query(q_full)", "tensor<float>(x[1024])"),
            ],
            functions=[
                Function(
                    name="unpack_binary",
                    expression="2*unpack_bits(attribute(binary_1024), float) - 1",
                ),
            ],
            first_phase=FirstPhaseRanking(
                expression="closeness(field, binary_1024)"
            ),
            second_phase=SecondPhaseRanking(
                expression="sum(query(q_full) * unpack_binary)",
                rerank_count=rerank_count,
            ),
            match_features=["distance(field, binary_1024)"],
        )
    )

    schema.add_rank_profile(
        RankProfile(
            name="binary-rescore-float",
            inputs=[
                ("query(q_binary_1024)", "tensor<int8>(x[128])"),
                ("query(q_full)", "tensor<float>(x[1024])"),
            ],
            first_phase=FirstPhaseRanking(
                expression="closeness(field, binary_1024)"
            ),
            second_phase=SecondPhaseRanking(
                expression="sum(query(q_full) * attribute(float_vector))",
                rerank_count=rerank_count,
            ),
            match_features=["distance(field, binary_1024)"],
        )
    )

    # Binary retrieval (1024) → int8 rescore (asymmetric: float query × int8 doc)
    schema.add_rank_profile(
        RankProfile(
            name="binary-rescore-int8",
            inputs=[
                ("query(q_binary_1024)", "tensor<int8>(x[128])"),
                ("query(q_full)", "tensor<float>(x[1024])"),
            ],
            first_phase=FirstPhaseRanking(
                expression="closeness(field, binary_1024)"
            ),
            second_phase=SecondPhaseRanking(
                expression="sum(query(q_full) * attribute(int8_vector))",
                rerank_count=rerank_count,
            ),
            match_features=["distance(field, binary_1024)"],
        )
    )

    # -- MRL binary retrieval → rescore variants --

    for mrl_dim, packed_dim in [(512, 64), (256, 32), (128, 16), (64, 8)]:
        # MRL binary → float rescore
        schema.add_rank_profile(
            RankProfile(
                name=f"mrl{mrl_dim}-binary-rescore-float",
                inputs=[
                    (f"query(q_binary_{mrl_dim})", f"tensor<int8>(x[{packed_dim}])"),
                    ("query(q_full)", "tensor<float>(x[1024])"),
                ],
                first_phase=FirstPhaseRanking(
                    expression=f"closeness(field, binary_{mrl_dim})"
                ),
                second_phase=SecondPhaseRanking(
                    expression="sum(query(q_full) * attribute(float_vector))",
                    rerank_count=rerank_count,
                ),
                match_features=[f"distance(field, binary_{mrl_dim})"],
            )
        )

        # MRL binary → int8 rescore
        schema.add_rank_profile(
            RankProfile(
                name=f"mrl{mrl_dim}-binary-rescore-int8",
                inputs=[
                    (f"query(q_binary_{mrl_dim})", f"tensor<int8>(x[{packed_dim}])"),
                    ("query(q_full)", "tensor<float>(x[1024])"),
                ],
                first_phase=FirstPhaseRanking(
                    expression=f"closeness(field, binary_{mrl_dim})"
                ),
                second_phase=SecondPhaseRanking(
                    expression="sum(query(q_full) * attribute(int8_vector))",
                    rerank_count=rerank_count,
                ),
                match_features=[f"distance(field, binary_{mrl_dim})"],
            )
        )

        # MRL binary → unpack full binary rescore
        schema.add_rank_profile(
            RankProfile(
                name=f"mrl{mrl_dim}-binary-rescore-unpack",
                inputs=[
                    (f"query(q_binary_{mrl_dim})", f"tensor<int8>(x[{packed_dim}])"),
                    ("query(q_full)", "tensor<float>(x[1024])"),
                ],
                functions=[
                    Function(
                        name="unpack_binary",
                        expression="2*unpack_bits(attribute(binary_1024), float) - 1",
                    ),
                ],
                first_phase=FirstPhaseRanking(
                    expression=f"closeness(field, binary_{mrl_dim})"
                ),
                second_phase=SecondPhaseRanking(
                    expression="sum(query(q_full) * unpack_binary)",
                    rerank_count=rerank_count,
                ),
                match_features=[f"distance(field, binary_{mrl_dim})"],
            )
        )

    # -- Float baselines (brute force, no HNSW) --

    # Full 1024-dim float
    schema.add_rank_profile(
        RankProfile(
            name="float-baseline",
            inputs=[("query(q_full)", "tensor<float>(x[1024])")],
            first_phase=FirstPhaseRanking(
                expression="sum(query(q_full) * attribute(float_vector))"
            ),
        )
    )

    # MRL float at each truncated dimension
    for mrl_dim in MRL_FLOAT_DIMS:
        schema.add_rank_profile(
            RankProfile(
                name=f"float-mrl{mrl_dim}",
                inputs=[(f"query(q_float_{mrl_dim})", f"tensor<float>(x[{mrl_dim}])")],
                first_phase=FirstPhaseRanking(
                    expression=f"sum(query(q_float_{mrl_dim}) * attribute(float_{mrl_dim}))"
                ),
            )
        )

    return schema


def build_application(rerank_count: int = 100) -> ApplicationPackage:
    """Build the full Vespa application package."""
    schema = build_schema(rerank_count=rerank_count)
    return ApplicationPackage(name="binarysearch", schema=[schema])


# ---------------------------------------------------------------------------
# Data loading & quantization
# ---------------------------------------------------------------------------


def load_and_prepare(
    dataset: str,
    model: str = "mxbai-embed-large-v1",
    cache_dir: Path = Path("cache/embeddings"),
    dataset_cache_dir: Path = Path("cache/datasets"),
) -> Tuple[EmbeddingData, List[str], Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray]]:
    """Load cached embeddings, document texts, and quantize for all MRL dims.

    Returns:
        data: EmbeddingData with float embeddings and qrels
        doc_texts: Document texts for feeding
        binary_vectors: {dim: (n_docs, packed_dim) int8 array}
        int8_vectors: (n_docs, 1024) int8 array
        mrl_float_vectors: {dim: (n_docs, dim) float32 array} truncated+renormalized
    """
    data = load_data(model, dataset, cache_dir, dataset_cache_dir)

    # Load document texts from dataset cache
    loader = MTEBDataLoader(dataset_cache_dir)
    ds_cfg = DATASETS.get(dataset, {})
    subsample = ds_cfg.get("subsample", None)
    corpus, queries_dict, _ = loader.load_dataset(dataset, subsample=subsample)
    doc_ids_text, doc_texts, _, _ = loader.get_texts_for_embedding(corpus, queries_dict)

    # Build doc_id -> text mapping
    text_map = dict(zip(doc_ids_text, doc_texts))
    ordered_texts = [text_map.get(did, "") for did in data.doc_ids]

    # Quantize corpus embeddings at each MRL dimension
    qh = QuantizationHandler()
    binary_vectors = {}
    for float_dim in MRL_BINARY_CONFIGS:
        truncated = qh.truncate_matryoshka(data.corpus_emb, float_dim)
        binary = qh.quantize_to_binary(truncated)
        # quantize_to_binary returns uint8 (ubinary), Vespa expects int8
        binary_vectors[float_dim] = binary.astype(np.int8)
        print(f"  Binary {float_dim}: {binary_vectors[float_dim].shape}")

    # Int8 quantization (calibrated on corpus)
    int8_vectors = qh.quantize_to_int8(
        data.corpus_emb, calibration_embeddings=data.corpus_emb
    )
    print(f"  Int8: {int8_vectors.shape} (dtype={int8_vectors.dtype})")

    # MRL float vectors (truncated + renormalized) for brute-force baselines
    mrl_float_vectors = {}
    for mrl_dim in MRL_FLOAT_DIMS:
        mrl_float_vectors[mrl_dim] = qh.truncate_matryoshka(data.corpus_emb, mrl_dim)
        print(f"  Float MRL {mrl_dim}: {mrl_float_vectors[mrl_dim].shape}")

    return data, ordered_texts, binary_vectors, int8_vectors, mrl_float_vectors


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


def deploy(tenant: str, app_package: ApplicationPackage, key: Optional[str] = None):
    """Deploy application to Vespa Cloud and return the Vespa app connection."""
    from vespa.deployment import VespaCloud

    if key is not None:
        key = key.replace(r"\n", "\n")

    vespa_cloud = VespaCloud(
        tenant=tenant,
        application="binarysearch",
        key_content=key,
        application_package=app_package,
    )

    print("Deploying to Vespa Cloud (may take ~2 min for first deploy)...")
    from vespa.application import Vespa

    app: Vespa = vespa_cloud.deploy()
    print(f"Deployed! Endpoint: {app.url}")
    return app, vespa_cloud


# ---------------------------------------------------------------------------
# Feeding
# ---------------------------------------------------------------------------


def feed_documents(
    app,
    data: EmbeddingData,
    doc_texts: List[str],
    binary_vectors: Dict[int, np.ndarray],
    int8_vectors: np.ndarray,
    mrl_float_vectors: Dict[int, np.ndarray],
    dataset: str = "",
    max_workers: int = 64,
):
    """Feed documents to Vespa using async HTTP/2 (feed_async_iterable).

    Uses httpx.AsyncClient with HTTP/2 multiplexing — same fast approach
    as query_many(). Much faster than the threaded feed_iterable().
    """
    from vespa.io import VespaResponse

    n_docs = len(data.doc_ids)
    print(f"Feeding {n_docs} documents (async HTTP/2, max_workers={max_workers})...")

    # Build feed iterable in the format pyvespa expects
    feed_data = []
    for i in range(n_docs):
        fields = {
            "doc_id": data.doc_ids[i],
            "dataset": dataset,
            "text": doc_texts[i],
            "float_vector": data.corpus_emb[i].tolist(),
            "int8_vector": int8_vectors[i].tolist(),
        }
        for float_dim in MRL_BINARY_CONFIGS:
            fields[f"binary_{float_dim}"] = binary_vectors[float_dim][i].tolist()
        for mrl_dim in MRL_FLOAT_DIMS:
            fields[f"float_{mrl_dim}"] = mrl_float_vectors[mrl_dim][i].tolist()

        feed_data.append({
            "id": data.doc_ids[i],
            "fields": fields,
        })

    failed = []
    fed_count = 0

    def callback(response: VespaResponse, id: str):
        nonlocal fed_count
        if not response.is_successful():
            failed.append(id)
            if len(failed) <= 3:
                print(f"  Feed error for doc {id}: {response.get_json()}")
        else:
            fed_count += 1
            if fed_count % 1000 == 0 or fed_count == n_docs:
                print(f"  Fed {fed_count}/{n_docs}...")

    start_time = time.time()
    app.feed_async_iterable(
        feed_data,
        schema="doc",
        callback=callback,
        max_workers=max_workers,
        max_connections=1,  # HTTP/2 multiplexes on one connection
        max_queue_size=1000,
    )
    elapsed = time.time() - start_time

    print(f"Fed {n_docs - len(failed)}/{n_docs} documents in {elapsed:.1f}s "
          f"({elapsed/n_docs*1000:.0f}ms/doc)")
    if failed:
        print(f"  {len(failed)} documents failed to feed")


# ---------------------------------------------------------------------------
# Query evaluation
# ---------------------------------------------------------------------------

# Each rank profile config with metadata for result export.
# Fields match runner.py ExperimentResult / visualize.py expectations:
#   method, truncate_dim, oversample, retrieval, rescore, funnel, funnel_factor
RANK_PROFILES = [
    # -- Binary-only (no rescore) --
    {
        "name": "binary-only",
        "nn_field": "binary_1024",
        "nn_query_var": "q_binary_1024",
        "method": "binary",
        "truncate_dim": 1024,
        "retrieval": "binary",
        "rescore": "none",
    },
    # -- Full binary retrieval (1024) → rescore --
    {
        "name": "binary-rescore-unpack",
        "nn_field": "binary_1024",
        "nn_query_var": "q_binary_1024",
        "method": "binary→binary",
        "truncate_dim": 1024,
        "retrieval": "binary",
        "rescore": "binary",
    },
    {
        "name": "binary-rescore-float",
        "nn_field": "binary_1024",
        "nn_query_var": "q_binary_1024",
        "method": "binary→float32",
        "truncate_dim": 1024,
        "retrieval": "binary",
        "rescore": "float32",
    },
    {
        "name": "binary-rescore-int8",
        "nn_field": "binary_1024",
        "nn_query_var": "q_binary_1024",
        "method": "binary→int8",
        "truncate_dim": 1024,
        "retrieval": "binary",
        "rescore": "int8",
    },
]

# Generate MRL profiles for each dimension
for _mrl_dim in [512, 256, 128, 64]:
    for _rescore_name, _rescore_label, _vespa_profile_suffix in [
        ("float32", "float32", "rescore-float"),
        ("int8", "int8", "rescore-int8"),
        ("binary", "binary", "rescore-unpack"),
    ]:
        RANK_PROFILES.append({
            "name": f"mrl{_mrl_dim}-binary-{_vespa_profile_suffix}",
            "nn_field": f"binary_{_mrl_dim}",
            "nn_query_var": f"q_binary_{_mrl_dim}",
            "method": f"binary→{_rescore_label}",
            "truncate_dim": _mrl_dim,
            "retrieval": "binary",
            "rescore": _rescore_name,
        })

# -- Float baselines (brute force, no HNSW index) --
RANK_PROFILES.append({
    "name": "float-baseline",
    "nn_field": None,
    "nn_query_var": None,
    "query_var": "q_full",
    "method": "float32",
    "truncate_dim": 1024,
    "retrieval": "float32",
    "rescore": "none",
})
for _mrl_dim in MRL_FLOAT_DIMS:
    RANK_PROFILES.append({
        "name": f"float-mrl{_mrl_dim}",
        "nn_field": None,
        "nn_query_var": None,
        "query_var": f"q_float_{_mrl_dim}",
        "method": "float32",
        "truncate_dim": _mrl_dim,
        "retrieval": "float32",
        "rescore": "none",
    })


def prepare_query_vectors(
    query_emb: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pre-compute all query vector representations."""
    qh = QuantizationHandler()
    vectors = {"q_full": query_emb}

    for float_dim in MRL_BINARY_CONFIGS:
        truncated = qh.truncate_matryoshka(query_emb, float_dim)
        binary = qh.quantize_to_binary(truncated).astype(np.int8)
        vectors[f"q_binary_{float_dim}"] = binary

    # Truncated float query vectors for MRL float baselines
    for mrl_dim in MRL_FLOAT_DIMS:
        vectors[f"q_float_{mrl_dim}"] = qh.truncate_matryoshka(query_emb, mrl_dim)

    return vectors


def evaluate_rank_profile(
    app,
    profile: dict,
    query_vectors: Dict[str, np.ndarray],
    data: EmbeddingData,
    dataset: str = "",
    rerank_count: int = 100,
    k_values: List[int] = [10, 100],
    max_concurrent: int = 64,
) -> Dict[str, float]:
    """Run all queries through a rank profile and compute metrics.

    Uses pyvespa's query_many() which uses httpx.AsyncClient with HTTP/2
    multiplexing, adaptive throttling, and automatic retries — much faster
    than sequential queries and avoids the malloc double-free crash that
    ThreadPoolExecutor caused on macOS.
    """
    from metrics import RetrievalMetrics

    n_queries = len(data.query_ids)
    max_k = max(k_values)

    nn_field = profile["nn_field"]
    nn_query_var = profile["nn_query_var"]
    rank_name = profile["name"]
    is_brute_force = nn_field is None

    # Pre-build all query bodies for query_many()
    query_bodies = []
    for qi in range(n_queries):
        if is_brute_force:
            # Float baseline / MRL float: brute force over all docs
            query_var = profile["query_var"]
            body = {
                f"input.query({query_var})": query_vectors[query_var][qi].tolist(),
                "hits": max_k,
                "presentation.timing": True,
                "ranking": rank_name,
            }
            yql = f"select doc_id from doc where dataset contains '{dataset}'"
        else:
            # Binary HNSW retrieval (± rescore)
            binary_key = nn_query_var
            body = {
                f"input.query({binary_key})": query_vectors[binary_key][qi].tolist(),
                "hits": max_k,
                "presentation.timing": True,
                "ranking": rank_name,
            }
            if "rescore" in rank_name:
                body["input.query(q_full)"] = query_vectors["q_full"][qi].tolist()
            yql = f"select doc_id from doc where dataset contains '{dataset}' and ({{targetHits:{rerank_count}, approximate:true}}nearestNeighbor({nn_field},{binary_key}))"

        body["yql"] = yql
        query_bodies.append(body)

    # Debug: print the first query body
    print(f"  [DEBUG] First query YQL: {query_bodies[0].get('yql', 'N/A')}")
    print(f"  [DEBUG] First query ranking: {query_bodies[0].get('ranking', 'N/A')}")

    # Fire all queries concurrently using pyvespa's async HTTP/2 client
    print(f"  Sending {n_queries} queries concurrently (max_concurrent={max_concurrent})...")
    start_time = time.time()
    responses = app.query_many(
        queries=query_bodies,
        num_connections=1,        # HTTP/2 multiplexes on one connection
        max_concurrent=max_concurrent,
        adaptive=True,            # auto-adjusts concurrency based on server load
    )
    client_wall_time = time.time() - start_time
    print(f"  All {n_queries} queries completed in {client_wall_time:.1f}s "
          f"({client_wall_time/n_queries*1000:.0f}ms avg)")

    # Process responses (they come back in order)
    results_by_qi = [None] * n_queries
    server_search_times = []
    server_total_times = []

    for qi, response in enumerate(responses):
        if qi == 0:
            # Debug first response
            resp_json = response.json
            total_count = resp_json.get("root", {}).get("fields", {}).get("totalCount", "N/A")
            n_hits = len(response.hits) if response.hits else 0
            print(f"  [DEBUG] First response: is_successful={response.is_successful()}, "
                  f"totalCount={total_count}, hits={n_hits}")
            if n_hits > 0:
                print(f"  [DEBUG] First hit fields: {response.hits[0].get('fields', {})}")
            elif "errors" in resp_json.get("root", {}):
                print(f"  [DEBUG] Errors: {resp_json['root']['errors']}")
            else:
                print(f"  [DEBUG] Response root keys: {list(resp_json.get('root', {}).keys())}")

        if not response.is_successful():
            results_by_qi[qi] = []
            continue

        timing = response.json.get("timing", {})
        search_time_s = timing.get("searchtime", 0)
        query_time_s = timing.get("querytime", 0)
        server_search_times.append(search_time_s * 1000)
        server_total_times.append(query_time_s * 1000)

        doc_ids_retrieved = [
            hit["fields"].get("doc_id", "") for hit in response.hits
        ]
        results_by_qi[qi] = doc_ids_retrieved[:max_k]

    # Convert to index-based for metrics
    doc_id_to_idx = {did: i for i, did in enumerate(data.doc_ids)}

    retrieved_indices = np.full((n_queries, max_k), -1, dtype=np.int64)
    for qi, doc_list in enumerate(results_by_qi):
        if doc_list is None:
            doc_list = []
        for rank, did in enumerate(doc_list):
            if did in doc_id_to_idx and rank < max_k:
                retrieved_indices[qi, rank] = doc_id_to_idx[did]

    metrics = RetrievalMetrics.compute_all_metrics(
        retrieved_indices, data.qrels, doc_id_to_idx, data.query_ids, k_values
    )

    # Client-side timing
    metrics["client_total_sec"] = client_wall_time
    metrics["client_avg_ms"] = (client_wall_time / n_queries) * 1000

    # Server-side timing (Vespa internal)
    if server_search_times:
        st = np.array(server_search_times)
        tt = np.array(server_total_times)
        metrics["vespa_search_avg_ms"] = float(np.mean(st))
        metrics["vespa_search_p50_ms"] = float(np.median(st))
        metrics["vespa_search_p95_ms"] = float(np.percentile(st, 95))
        metrics["vespa_search_p99_ms"] = float(np.percentile(st, 99))
        metrics["vespa_query_avg_ms"] = float(np.mean(tt))
        metrics["vespa_query_p95_ms"] = float(np.percentile(tt, 95))

    return metrics


def run_evaluation(
    app,
    data: EmbeddingData,
    model: str,
    dataset: str,
    rerank_count: int = 100,
    profiles: Optional[List[str]] = None,
) -> Tuple[Dict, List[dict]]:
    """Run evaluation across all (or selected) rank profiles.

    Returns:
        results: dict of profile_name -> metrics (for summary table)
        experiment_results: list of dicts matching runner.py ExperimentResult format
    """
    query_vectors = prepare_query_vectors(data.query_emb)

    results = {}
    experiment_results = []

    for profile in RANK_PROFILES:
        if profiles and profile["name"] not in profiles:
            continue

        desc = f"{profile['method']} dim={profile['truncate_dim']}"
        print(f"\nEvaluating: {profile['name']} — {desc}")
        metrics = evaluate_rank_profile(
            app, profile, query_vectors, data, dataset=dataset, rerank_count=rerank_count
        )
        results[profile["name"]] = metrics

        for key, val in sorted(metrics.items()):
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")

        # Build result dict matching runner.py ExperimentResult / visualize.py format
        eval_metrics = {
            k: v for k, v in metrics.items()
            if k.startswith("recall@") or k.startswith("ndcg@")
        }
        # Include Vespa timing in metrics
        for k in ["vespa_search_avg_ms", "vespa_search_p50_ms",
                   "vespa_search_p95_ms", "vespa_search_p99_ms"]:
            if k in metrics:
                eval_metrics[k] = metrics[k]

        experiment_results.append({
            "model": model,
            "dataset": dataset.lower(),
            "method": profile["method"],
            "truncate_dim": profile["truncate_dim"],
            "oversample": OVERSAMPLE,
            "funnel_factor": 0,
            "retrieval": profile["retrieval"],
            "rescore": profile["rescore"],
            "funnel": False,
            "metrics": eval_metrics,
        })

    return results, experiment_results


def print_summary_table(results: Dict[str, Dict[str, float]]):
    """Print a comparison table of all evaluated rank profiles."""
    if not results:
        return

    print("\n" + "=" * 120)
    print(f"{'Rank Profile':<35} {'NDCG@10':>8} {'R@10':>8} "
          f"{'NDCG@100':>8} {'R@100':>8} "
          f"{'Vespa avg':>10} {'Vespa p50':>10} {'Vespa p95':>10} {'Client avg':>10}")
    print(f"{'':35} {'':>8} {'':>8} {'':>8} {'':>8} "
          f"{'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10}")
    print("-" * 120)

    for name, metrics in results.items():
        ndcg10 = metrics.get("ndcg@10", 0)
        recall10 = metrics.get("recall@10", 0)
        ndcg100 = metrics.get("ndcg@100", 0)
        recall100 = metrics.get("recall@100", 0)
        vespa_avg = metrics.get("vespa_search_avg_ms", 0)
        vespa_p50 = metrics.get("vespa_search_p50_ms", 0)
        vespa_p95 = metrics.get("vespa_search_p95_ms", 0)
        client_avg = metrics.get("client_avg_ms", 0)
        print(f"{name:<35} {ndcg10:>8.4f} {recall10:>8.4f} "
              f"{ndcg100:>8.4f} {recall100:>8.4f} "
              f"{vespa_avg:>10.2f} {vespa_p50:>10.2f} {vespa_p95:>10.2f} {client_avg:>10.1f}")

    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Deploy cached embeddings to Vespa Cloud and evaluate ranking schemes"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (must match cache, e.g. NFCorpus, SciFact)"
    )
    parser.add_argument(
        "--model", default="mxbai-embed-large-v1",
        help="Model key from config.py (default: mxbai-embed-large-v1)"
    )
    parser.add_argument(
        "--tenant", default="ntnuimf",
        help="Vespa Cloud tenant name"
    )
    parser.add_argument(
        "--deploy-only", action="store_true",
        help="Only deploy schema, don't feed or evaluate"
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip deploy and feed, only run evaluation (assumes app is running)"
    )
    parser.add_argument(
        "--feed-only", action="store_true",
        help="Skip deploy, only feed documents (assumes app is deployed)"
    )
    parser.add_argument(
        "--profiles", nargs="*",
        help="Specific rank profiles to evaluate (default: all)"
    )
    parser.add_argument(
        "--rerank-count", type=int, default=None,
        help="Second-phase rerank count (default: max(K_VALUES) * OVERSAMPLE = %(default)s)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results JSON to this path"
    )

    args = parser.parse_args()

    rerank_count = args.rerank_count or max(K_VALUES) * OVERSAMPLE
    print(f"rerank_count = {rerank_count} (max_k={max(K_VALUES)} * oversample={OVERSAMPLE})")

    key = os.getenv("VESPA_TEAM_API_KEY", None)
    key_path = Path(__file__).parent / "hakon.noren.ntnuimf.pem"
    if key is None and key_path.exists():
        key = key_path.read_text()
    if key is None:
        print("Warning: No API key found. Set VESPA_TEAM_API_KEY env var "
              f"or place a .pem file at {key_path}")
    app_package = build_application(rerank_count=rerank_count)

    if not args.evaluate_only:
        # Load and prepare data
        print(f"\nLoading data for {args.model}/{args.dataset}...")
        data, doc_texts, binary_vectors, int8_vectors, mrl_float_vectors = load_and_prepare(
            args.dataset, args.model
        )

    if not args.evaluate_only and not args.feed_only:
        # Deploy
        app, vespa_cloud = deploy(args.tenant, app_package, key)
    else:
        # Connect to existing deployment
        from vespa.deployment import VespaCloud

        if key is not None:
            key = key.replace(r"\n", "\n")
        vespa_cloud = VespaCloud(
            tenant=args.tenant,
            application="binarysearch",
            key_content=key,
            application_package=app_package,
        )
        from vespa.application import Vespa
        app: Vespa = vespa_cloud.deploy()

    if args.deploy_only:
        print("Deploy complete. Exiting.")
        return

    if not args.evaluate_only:
        # Feed documents
        feed_documents(app, data, doc_texts, binary_vectors, int8_vectors, mrl_float_vectors, dataset=args.dataset)

    if args.feed_only:
        print("Feed complete. Exiting.")
        return

    # Evaluate
    if args.evaluate_only:
        data, _, _, _, _ = load_and_prepare(args.dataset, args.model)

    results, experiment_results = run_evaluation(
        app, data, args.model, args.dataset,
        rerank_count=rerank_count, profiles=args.profiles,
    )
    print_summary_table(results)

    # Save results in standard ExperimentResult format
    profiles_tag = "+".join(args.profiles) if args.profiles else "all"
    experiment_id = f"vespa_{args.model}_{args.dataset.lower()}_{profiles_tag}"
    output_dir = Path(args.output) if args.output else Path("results") / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    json_data = {
        "experiment_id": experiment_id,
        "results": experiment_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
