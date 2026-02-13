#!/usr/bin/env python3
"""Compute memory savings table for top picks vs float32 baseline."""
import json
from collections import defaultdict

RESULTS_FILE = "results/mxbai+nomic_fiqa+nfco+scif_r_binary+float32+int4_s_binary_asym+binary_median+float32+int4+int8+lloyd_max+none_ff1/results.json"

with open(RESULTS_FILE) as f:
    data = json.load(f)
    results = data["results"] if isinstance(data, dict) and "results" in data else data

mem = defaultdict(lambda: defaultdict(list))

METHOD_MAP = {
    "float32": "float32->float32",
    "float32->float32": "float32->float32",
}

for r in results:
    d = r["truncate_dim"]
    method = r["method"]
    m = r["metrics"]
    idx = m.get("index_mem_mb", 0)
    res = m.get("rescore_vec_mem_mb", 0)
    total = idx + res
    funnel = r.get("funnel", False)

    # Map to canonical names (use ASCII arrows for consistency)
    canon = method.replace("\u2192", "->")
    if canon == "float32":
        key = "float32->float32"
    elif funnel:
        rescore = r.get("rescore", "none")
        key = f"binary->funnel->{rescore}"
    else:
        key = canon

    mem[key][d].append((total, idx, res))

# Our 5 picks + baseline
picks = [
    "float32->float32",
    "binary->funnel->float32",
    "binary->funnel->lloyd_max",
    "binary->funnel->int4",
    "binary->binary_asym",
    "binary->funnel->binary_median",
]

retention = {
    "float32->float32": "100.0%",
    "binary->funnel->float32": "100.2%",
    "binary->funnel->lloyd_max": "97.9%",
    "binary->funnel->int4": "98.1%",
    "binary->binary_asym": "96.2%",
    "binary->funnel->binary_median": "95.9%",
}

dims = [64, 128, 256, 512, 1024]

print("\n=== Memory savings vs float32->float32 (total = index + rescore) ===\n")
print(f"{'Method':<35} {'Ret':>6}", end="")
for d in dims:
    print(f"  {d:>5}d", end="")
print()
print("-" * 85)

for p in picks:
    print(f"{p:<35} {retention[p]:>6}", end="")
    for d in dims:
        f32_vals = mem["float32->float32"].get(d, [])
        p_vals = mem[p].get(d, [])
        if f32_vals and p_vals:
            f32_avg = sum(v[0] for v in f32_vals) / len(f32_vals)
            p_avg = sum(v[0] for v in p_vals) / len(p_vals)
            ratio = f32_avg / p_avg
            print(f"  {ratio:>5.1f}x", end="")
        else:
            print(f"  {'--':>5}x", end="")
    print()

print()
print("=== Breakdown at max dim (1024d, avg across models+datasets) ===\n")
print(f"{'Method':<35} {'Index MB':>10} {'Rescore MB':>10} {'Total MB':>10} {'Savings':>8}")
print("-" * 80)
for p in picks:
    vals = mem[p].get(1024, [])
    if not vals:
        vals = mem[p].get(768, [])
    if vals:
        avg_idx = sum(v[1] for v in vals) / len(vals)
        avg_res = sum(v[2] for v in vals) / len(vals)
        avg_tot = sum(v[0] for v in vals) / len(vals)
        f32_vals = mem["float32->float32"].get(1024, [])
        f32_avg = sum(v[0] for v in f32_vals) / len(f32_vals) if f32_vals else 1
        ratio = f32_avg / avg_tot
        print(f"{p:<35} {avg_idx:>9.3f}M {avg_res:>9.3f}M {avg_tot:>9.3f}M {ratio:>6.1f}x")
