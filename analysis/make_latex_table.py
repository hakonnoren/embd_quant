"""Generate a LaTeX table with compression ratio and quality retention from Vespa results."""

import json
import sys


def bytes_per_vector(method: str, dim: int) -> int:
    retrieval = method.split("\u2192")[0] if "\u2192" in method else method
    if retrieval == "float32":
        return dim * 4
    elif retrieval == "binary":
        return dim // 8
    elif retrieval in ("int8",):
        return dim
    else:
        return dim * 4


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/vespa_mxbai-embed-large-v1_fiqa_all/results.json"
    metric = sys.argv[2] if len(sys.argv) > 2 else "ndcg@100"

    with open(path) as f:
        data = json.load(f)

    results = data["results"]

    # Find float32 full-dim baseline
    baseline = None
    for r in results:
        if r["method"] == "float32" and r["truncate_dim"] == 1024:
            baseline = r
            break
    if not baseline:
        print("No float32 1024-dim baseline found")
        return

    baseline_bpv = bytes_per_vector("float32", 1024)
    baseline_score = baseline["metrics"][metric]

    # Build rows: (method, dim, bpv, compression, score, retention, latency)
    rows = []
    for r in results:
        method = r["method"]
        dim = r["truncate_dim"]
        bpv = bytes_per_vector(method, dim)
        compression = baseline_bpv / bpv
        score = r["metrics"][metric]
        retention = score / baseline_score * 100
        latency = r["metrics"].get("vespa_search_avg_ms", 0)
        rows.append((method, dim, bpv, compression, score, retention, latency))

    # Sort by compression (ascending = least compressed first)
    rows.sort(key=lambda x: (x[3], x[0]))

    # Compute Pareto fronts using dominance check
    # A point is Pareto-optimal if no other point has (cost <= this AND score >= this)
    # with at least one strict inequality.
    def pareto_front(points, cost_idx, score_idx):
        """Return set of (method, dim) keys on the Pareto front."""
        front = set()
        for i, a in enumerate(points):
            dominated = False
            for j, b in enumerate(points):
                if i == j:
                    continue
                # b dominates a if b is at least as good in both and strictly better in one
                if b[cost_idx] <= a[cost_idx] and b[score_idx] >= a[score_idx]:
                    if b[cost_idx] < a[cost_idx] or b[score_idx] > a[score_idx]:
                        dominated = True
                        break
            if not dominated:
                front.add((a[0], a[1]))  # (method, dim)
        return front

    # idx: 0=method, 1=dim, 2=bpv, 3=compression, 4=score, 5=retention, 6=latency
    mem_pareto = pareto_front(rows, cost_idx=2, score_idx=4)
    lat_pareto = pareto_front(rows, cost_idx=6, score_idx=4)

    # Print LaTeX
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\small")
    print(r"\caption{Compression vs.\ quality retention (FiQA, " + metric + r")}")
    print(r"\label{tab:compression_retention}")
    print(r"\begin{tabular}{llrrrr}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Dim} & \textbf{Compr.} & "
          r"\textbf{Retain.} & \textbf{Lat.\,(ms)} \\")
    print(r"\midrule")

    for method, dim, bpv, compression, score, retention, latency in rows:
        method_tex = method.replace("\u2192", r"$\to$")
        key = (method, dim)
        is_mem = key in mem_pareto
        is_lat = key in lat_pareto
        marks = []
        if is_mem:
            marks.append(r"\dagger")
        if is_lat:
            marks.append(r"\star")
        markers = ("$^{" + "".join(marks) + "}$") if marks else ""
        print(f"{method_tex}{markers} & {dim} & {compression:.0f}$\\times$ & "
              f"{retention:.1f}\\% & {latency:.1f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\vspace{0.1cm}")
    print(r"\\{\scriptsize $^{\dagger}$\,Pareto-optimal (memory) \quad $^{\star}$\,Pareto-optimal (latency)}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
