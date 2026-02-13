#!/usr/bin/env python3
"""Analyze experiment results: cross-model, cross-dataset conclusions.

Usage:
  python analyze_results.py results/.../results.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    raw = data.get("results", data)

    records = []
    for r in raw:
        base = {
            "model": r["model"],
            "dataset": r["dataset"],
            "method": r["method"],
            "truncate_dim": r["truncate_dim"],
            "oversample": r.get("oversample", 0),
            "funnel_factor": r.get("funnel_factor", 0),
            "retrieval": r.get("retrieval", ""),
            "rescore": r.get("rescore", "none"),
            "funnel": r.get("funnel", False),
        }
        base.update(r["metrics"])
        records.append(base)
    return pd.DataFrame(records)


def bytes_per_vector(method: str, dim: int, rescore: str) -> float:
    """Estimate total bytes per vector (index + rescore storage)."""
    retrieval_part = method.split("→")[0] if "→" in method else method

    if retrieval_part == "float32":
        bpv = dim * 4
    elif retrieval_part == "binary":
        bpv = dim // 8
    elif retrieval_part == "int4":
        bpv = max(1, dim // 4)
    else:
        bpv = dim * 4

    if rescore == "float32":
        bpv += dim * 4
    elif rescore == "int8":
        bpv += dim
    elif rescore == "int4":
        bpv += max(1, dim // 4)
    elif rescore in ("binary_asym", "binary_median"):
        bpv += dim // 8
    elif rescore == "lloyd_max":
        bpv += max(1, dim // 4)

    return bpv


def analyze(df: pd.DataFrame):
    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())
    methods = sorted(df["method"].unique())
    dims = sorted(df["truncate_dim"].unique())

    print("=" * 80)
    print("EXPERIMENT OVERVIEW")
    print("=" * 80)
    print(f"Models:   {models}")
    print(f"Datasets: {datasets}")
    print(f"Methods:  {methods}")
    print(f"Dims:     {dims}")
    print(f"Total entries: {len(df)}")
    print()

    # Add bytes_per_vector column
    df["bpv"] = df.apply(
        lambda r: bytes_per_vector(r["method"], int(r["truncate_dim"]), r.get("rescore", "none")),
        axis=1
    )

    # ── 1. RETENTION TABLE: % of float32 baseline at full dim ──────────────
    print("=" * 80)
    print("1. PERFORMANCE RETENTION (% of float32 at model's max dim)")
    print("   Metric: ndcg@10")
    print("=" * 80)

    for model in models:
        for dataset in datasets:
            sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
            max_dim = sub["truncate_dim"].max()
            baseline_rows = sub[(sub["method"] == "float32") & (sub["truncate_dim"] == max_dim)]
            if baseline_rows.empty:
                continue
            baseline = baseline_rows["ndcg@10"].values[0]

            print(f"\n{model} | {dataset} (baseline ndcg@10 = {baseline:.4f})")
            print(f"{'Method':<30} {'Dim':>5} {'NDCG@10':>9} {'Ret%':>7} {'R@100':>9} {'BPV':>6}")
            print("-" * 72)

            for _, row in sub.sort_values(["method", "truncate_dim"], ascending=[True, False]).iterrows():
                ret = (row["ndcg@10"] / baseline * 100) if baseline > 0 else 0
                print(f"{row['method']:<30} {int(row['truncate_dim']):>5} "
                      f"{row['ndcg@10']:>9.4f} {ret:>6.1f}% "
                      f"{row.get('recall@100', 0):>9.4f} {row['bpv']:>6.0f}")

    # ── 2. CROSS-MODEL CROSS-DATASET AVERAGE RETENTION ─────────────────────
    print("\n" + "=" * 80)
    print("2. AVERAGE RETENTION ACROSS ALL (model, dataset) COMBOS")
    print("   How much % of float32@full_dim does each method retain on average?")
    print("=" * 80)

    retention_records = []
    for model in models:
        for dataset in datasets:
            sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
            max_dim = sub["truncate_dim"].max()
            bl_rows = sub[(sub["method"] == "float32") & (sub["truncate_dim"] == max_dim)]
            if bl_rows.empty:
                continue
            bl_ndcg = bl_rows["ndcg@10"].values[0]
            bl_recall = bl_rows["recall@100"].values[0]

            for _, row in sub.iterrows():
                ret_ndcg = (row["ndcg@10"] / bl_ndcg * 100) if bl_ndcg > 0 else 0
                ret_recall = (row["recall@100"] / bl_recall * 100) if bl_recall > 0 else 0
                retention_records.append({
                    "model": model, "dataset": dataset,
                    "method": row["method"], "dim": int(row["truncate_dim"]),
                    "ret_ndcg10": ret_ndcg, "ret_recall100": ret_recall,
                    "bpv": row["bpv"],
                })

    ret_df = pd.DataFrame(retention_records)

    # Group by method + dim, average across (model, dataset)
    avg = ret_df.groupby(["method", "dim"]).agg(
        mean_ret_ndcg=("ret_ndcg10", "mean"),
        std_ret_ndcg=("ret_ndcg10", "std"),
        mean_ret_recall=("ret_recall100", "mean"),
        std_ret_recall=("ret_recall100", "std"),
        mean_bpv=("bpv", "mean"),
        n=("ret_ndcg10", "count"),
    ).reset_index()

    # Show only common dims
    for dim in sorted(avg["dim"].unique(), reverse=True):
        dim_data = avg[avg["dim"] == dim].sort_values("mean_ret_ndcg", ascending=False)
        if dim_data.empty:
            continue
        print(f"\n  Dim = {dim}")
        print(f"  {'Method':<30} {'Avg NDCG Ret%':>14} {'±σ':>6} {'Avg R@100 Ret%':>15} {'±σ':>6} {'BPV':>6} {'n':>3}")
        print(f"  {'-'*83}")
        for _, row in dim_data.iterrows():
            print(f"  {row['method']:<30} {row['mean_ret_ndcg']:>13.1f}% {row['std_ret_ndcg']:>5.1f} "
                  f"{row['mean_ret_recall']:>14.1f}% {row['std_ret_recall']:>5.1f} "
                  f"{row['mean_bpv']:>6.0f} {int(row['n']):>3}")

    # ── 3. MEMORY/ACCURACY SWEET SPOTS ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("3. MEMORY vs ACCURACY SWEET SPOTS")
    print("   Best method at each bytes-per-vector budget")
    print("=" * 80)

    for model in models:
        for dataset in datasets:
            sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
            max_dim = sub["truncate_dim"].max()
            bl_rows = sub[(sub["method"] == "float32") & (sub["truncate_dim"] == max_dim)]
            if bl_rows.empty:
                continue
            bl = bl_rows["ndcg@10"].values[0]

            print(f"\n{model} | {dataset} (float32 baseline: {bl:.4f})")

            # Group by BPV bucket and pick best ndcg@10
            sub_sorted = sub.sort_values("bpv")
            budgets = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
            print(f"  {'Budget(B)':>10} {'Best Method':<30} {'Dim':>5} {'NDCG@10':>9} {'Ret%':>7} {'Actual BPV':>11}")
            print(f"  {'-'*78}")
            for budget in budgets:
                candidates = sub_sorted[sub_sorted["bpv"] <= budget]
                if candidates.empty:
                    continue
                best = candidates.loc[candidates["ndcg@10"].idxmax()]
                ret = (best["ndcg@10"] / bl * 100) if bl > 0 else 0
                print(f"  {budget:>10} {best['method']:<30} {int(best['truncate_dim']):>5} "
                      f"{best['ndcg@10']:>9.4f} {ret:>6.1f}% {best['bpv']:>11.0f}")

    # ── 4. RESCORE METHOD COMPARISON (same retrieval dim) ──────────────────
    print("\n" + "=" * 80)
    print("4. RESCORE METHOD VALUE-ADD (over bare binary)")
    print("   Delta ndcg@10 from adding rescore at each dim")
    print("=" * 80)

    for dim in sorted(dims, reverse=True):
        records_at_dim = []
        for model in models:
            for dataset in datasets:
                sub = df[(df["model"] == model) & (df["dataset"] == dataset)
                         & (df["truncate_dim"] == dim)]
                binary_rows = sub[sub["method"] == "binary"]
                if binary_rows.empty:
                    continue
                binary_ndcg = binary_rows["ndcg@10"].values[0]

                for _, row in sub.iterrows():
                    if row["method"] == "binary":
                        continue
                    delta = row["ndcg@10"] - binary_ndcg
                    records_at_dim.append({
                        "model": model, "dataset": dataset,
                        "method": row["method"],
                        "delta_ndcg": delta,
                        "extra_bpv": row["bpv"] - (dim // 8),
                    })

        if not records_at_dim:
            continue

        delta_df = pd.DataFrame(records_at_dim)
        avg_delta = delta_df.groupby("method").agg(
            mean_delta=("delta_ndcg", "mean"),
            std_delta=("delta_ndcg", "std"),
            mean_extra_bpv=("extra_bpv", "mean"),
            n=("delta_ndcg", "count"),
        ).reset_index().sort_values("mean_delta", ascending=False)

        print(f"\n  Dim = {dim} (binary baseline = {dim//8} BPV)")
        print(f"  {'Method':<30} {'Avg Δ NDCG@10':>14} {'±σ':>7} {'Extra BPV':>10} {'n':>3}")
        print(f"  {'-'*68}")
        for _, row in avg_delta.iterrows():
            sign = "+" if row["mean_delta"] >= 0 else ""
            print(f"  {row['method']:<30} {sign}{row['mean_delta']:>13.4f} {row['std_delta']:>6.4f} "
                  f"{row['mean_extra_bpv']:>10.0f} {int(row['n']):>3}")

    # ── 5. FUNNEL ANALYSIS ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("5. FUNNEL ANALYSIS")
    print("   Does the funnel flag provide value over non-funnel rescore?")
    print("=" * 80)

    funnel_rows = df[df["funnel"] == True]
    non_funnel_rows = df[df["funnel"] == False]

    if funnel_rows.empty:
        print("  No funnel entries found in this results file.")
        print("  (Funnel experiments may be in binary_rerank results instead.)")
    else:
        for dim in sorted(dims, reverse=True):
            for model in models:
                for dataset in datasets:
                    f_sub = funnel_rows[(funnel_rows["model"] == model) &
                                        (funnel_rows["dataset"] == dataset) &
                                        (funnel_rows["truncate_dim"] == dim)]
                    nf_sub = non_funnel_rows[(non_funnel_rows["model"] == model) &
                                              (non_funnel_rows["dataset"] == dataset) &
                                              (non_funnel_rows["truncate_dim"] == dim)]
                    if f_sub.empty:
                        continue
                    print(f"\n  {model} | {dataset} | dim={dim}")
                    for _, fr in f_sub.iterrows():
                        # Find matching non-funnel with same rescore
                        match = nf_sub[nf_sub["rescore"] == fr["rescore"]]
                        if not match.empty:
                            nf_ndcg = match["ndcg@10"].values[0]
                            delta = fr["ndcg@10"] - nf_ndcg
                            print(f"    {fr['method']}: {fr['ndcg@10']:.4f} vs non-funnel {nf_ndcg:.4f} "
                                  f"(Δ={delta:+.4f})")
                        else:
                            print(f"    {fr['method']}: {fr['ndcg@10']:.4f} (no non-funnel match)")

    # ── 6. BINARY_MEDIAN vs BINARY_ASYM COMPARISON ─────────────────────────
    print("\n" + "=" * 80)
    print("6. BINARY_MEDIAN vs BINARY_ASYM HEAD-TO-HEAD")
    print("   Same retrieval dim, same binary retrieval stage")
    print("=" * 80)

    for dim in sorted(dims, reverse=True):
        pairs = []
        for model in models:
            for dataset in datasets:
                sub = df[(df["model"] == model) & (df["dataset"] == dataset)
                         & (df["truncate_dim"] == dim)]
                median_rows = sub[sub["rescore"] == "binary_median"]
                asym_rows = sub[sub["rescore"] == "binary_asym"]
                if median_rows.empty or asym_rows.empty:
                    continue
                m_ndcg = median_rows["ndcg@10"].values[0]
                a_ndcg = asym_rows["ndcg@10"].values[0]
                pairs.append({
                    "model": model, "dataset": dataset,
                    "median_ndcg": m_ndcg, "asym_ndcg": a_ndcg,
                    "delta": m_ndcg - a_ndcg,
                })

        if not pairs:
            continue
        pair_df = pd.DataFrame(pairs)
        avg_delta = pair_df["delta"].mean()
        median_wins = (pair_df["delta"] > 0).sum()
        total = len(pair_df)

        print(f"\n  Dim = {dim}: median wins {median_wins}/{total} times, avg Δ(median−asym) = {avg_delta:+.4f}")
        for _, p in pair_df.iterrows():
            winner = "MEDIAN" if p["delta"] > 0 else "ASYM"
            print(f"    {p['model']:<30} {p['dataset']:<10} "
                  f"median={p['median_ndcg']:.4f} asym={p['asym_ndcg']:.4f} "
                  f"Δ={p['delta']:+.4f} [{winner}]")

    # ── 7. COMPRESSION EFFICIENCY RANKING ──────────────────────────────────
    print("\n" + "=" * 80)
    print("7. COMPRESSION EFFICIENCY: ndcg@10 per byte")
    print("   Which method gives the most quality per byte of storage?")
    print("=" * 80)

    # Average across all (model, dataset) at each (method, dim)
    eff = ret_df.copy()
    eff["ndcg_per_byte"] = eff["ret_ndcg10"] / eff["bpv"]

    eff_avg = eff.groupby(["method", "dim"]).agg(
        mean_eff=("ndcg_per_byte", "mean"),
        mean_ret=("ret_ndcg10", "mean"),
        bpv=("bpv", "first"),
    ).reset_index().sort_values("mean_eff", ascending=False)

    print(f"\n  Top 20 (method, dim) by avg efficiency (retention% / BPV):")
    print(f"  {'Method':<30} {'Dim':>5} {'BPV':>6} {'Avg Ret%':>9} {'Eff':>8}")
    print(f"  {'-'*62}")
    for _, row in eff_avg.head(20).iterrows():
        print(f"  {row['method']:<30} {int(row['dim']):>5} {row['bpv']:>6.0f} "
              f"{row['mean_ret']:>8.1f}% {row['mean_eff']:>8.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results.json>")
        sys.exit(1)
    path = sys.argv[1]
    df = load_results(path)
    analyze(df)


if __name__ == "__main__":
    main()
