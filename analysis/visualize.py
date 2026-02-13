"""Unified visualization for quantization + reranking experiments.

Usage:
  python visualize.py results/experiment_id/results.json
  python visualize.py results/experiment_id/results.json --show
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from config import (
    METHOD_COLORS, METHOD_MARKERS,
    FALLBACK_COLORS, FALLBACK_MARKERS,
)


class Visualizer:
    """Create visualizations from unified experiment results."""

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.output_dir = self.results_path.parent / "plots"
        self.output_dir.mkdir(exist_ok=True)

        with open(results_path) as f:
            data = json.load(f)
            self.raw_results = data.get("results", data)

        self.df = self._to_dataframe()
        self.methods = sorted(self.df["method"].unique())
        self.models = sorted(self.df["model"].unique())
        self.datasets = sorted(self.df["dataset"].unique())

        # Build color/marker maps for methods present, using config defaults + fallbacks
        self._colors = {}
        self._markers = {}
        fb_ci, fb_mi = 0, 0
        for m in self.methods:
            if m in METHOD_COLORS:
                self._colors[m] = METHOD_COLORS[m]
            else:
                self._colors[m] = FALLBACK_COLORS[fb_ci % len(FALLBACK_COLORS)]
                fb_ci += 1
            if m in METHOD_MARKERS:
                self._markers[m] = METHOD_MARKERS[m]
            else:
                self._markers[m] = FALLBACK_MARKERS[fb_mi % len(FALLBACK_MARKERS)]
                fb_mi += 1

    def _to_dataframe(self) -> pd.DataFrame:
        records = []
        # Collect non-zero oversamples for fp32 baseline replication
        non_zero_os = set()
        for r in self.raw_results:
            if r.get("oversample", 0) > 0:
                non_zero_os.add(r["oversample"])

        for r in self.raw_results:
            # Support both old schema (quantization/matryoshka_dim/rotation)
            # and new schema (method/truncate_dim/retrieval/rescore)
            method = r.get("method") or r.get("quantization", "unknown")
            truncate_dim = r.get("truncate_dim") or r.get("matryoshka_dim", 0)
            base = {
                "model": r["model"],
                "dataset": r["dataset"],
                "method": method,
                "truncate_dim": truncate_dim,
                "oversample": r.get("oversample", 0),
                "funnel_factor": r.get("funnel_factor", 0),
                "retrieval": r.get("retrieval", ""),
                "rescore": r.get("rescore", "none"),
                "funnel": r.get("funnel", False),
            }
            base.update(r["metrics"])

            # Replicate float32 baseline for each oversample so it appears
            # as a reference in every panel
            if method == "float32" and r.get("oversample", 0) == 0:
                for os_val in sorted(non_zero_os) or [0]:
                    row = dict(base)
                    row["oversample"] = os_val
                    records.append(row)
            else:
                records.append(base)

        return pd.DataFrame(records)

    def _bytes_per_vector(self, method: str, dim: int) -> float:
        """Estimate index RAM bytes per vector (retrieval stage only).

        Rescore vectors are paged (on disk) and don't affect corpus scaling,
        so they are excluded. Rescore cost is captured by the latency metric.
        """
        retrieval_part = method.split("→")[0] if "→" in method else method

        if retrieval_part == "float32":
            bpv = dim * 4
        elif retrieval_part == "binary":
            bpv = dim // 8
        elif retrieval_part == "int4":
            bpv = max(1, dim // 4)
        else:
            bpv = dim * 4  # fallback

        return bpv

    def _save(self, fig: plt.Figure, name: str):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    def _make_grid(self, title: str, width: float = 6, height: float = 5
                   ) -> Tuple[plt.Figure, np.ndarray]:
        n_rows = max(1, len(self.models))
        n_cols = max(1, len(self.datasets))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(width * n_cols, height * n_rows),
            squeeze=False,
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig, axes

    def _method_legend(self, ax, methods_shown):
        handles = [
            Line2D([0], [0],
                   marker=self._markers.get(m, "o"),
                   color=self._colors.get(m, "#95a5a6"),
                   linestyle="None", markersize=6, label=m)
            for m in self.methods if m in methods_shown
        ]
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="best")

    # ── Dim sweep ────────────────────────────────────────────────────────────

    def plot_dim_sweep(self, metric: str = "ndcg@10", save: bool = True) -> Optional[plt.Figure]:
        dims = sorted(self.df["truncate_dim"].unique())
        if len(dims) < 2:
            return None

        fig, axes = self._make_grid(f"{metric.upper()} vs Truncation Dimension")

        for i, model in enumerate(self.models):
            for j, dataset in enumerate(self.datasets):
                ax = axes[i, j]
                subset = self.df[(self.df["model"] == model) & (self.df["dataset"] == dataset)]

                # Count plottable methods to center jitter offsets
                plottable = [m for m in self.methods
                             if len(subset[subset["method"] == m]) >= 2]
                n_methods = len(plottable)

                for mi, method in enumerate(plottable):
                    data = subset[subset["method"] == method].sort_values("truncate_dim")
                    # Multiplicative jitter on log2 x-axis to separate overlapping markers
                    jitter = 1.0 + 0.03 * (mi - (n_methods - 1) / 2)
                    x_vals = data["truncate_dim"].values * jitter
                    ax.plot(x_vals, data[metric],
                            marker=self._markers.get(method, "o"),
                            color=self._colors.get(method, "#95a5a6"),
                            label=method, linewidth=2, markersize=6)

                ax.set_xlabel("Truncation dimension")
                ax.set_ylabel(metric.upper())
                ax.set_title(f"{model} | {dataset}")
                ax.set_xscale("log", base=2)
                ax.set_xticks(dims)
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)

        plt.tight_layout()
        if save:
            self._save(fig, f"dim_sweep_{metric.replace('@', '_')}")
        return fig

    # ── Funnel factor sweep ──────────────────────────────────────────────────

    def plot_funnel_factor_sweep(self, metric: str = "ndcg@10", save: bool = True) -> Optional[plt.Figure]:
        funnel_df = self.df[self.df["funnel_factor"] > 0]
        if funnel_df.empty:
            return None
        ffs = sorted(funnel_df["funnel_factor"].unique())
        if len(ffs) < 2:
            return None

        fig, axes = self._make_grid(f"{metric.upper()} vs Funnel Factor")
        dim_styles = {1024: "-", 768: "-", 512: "--", 256: "-.", 128: ":", 64: (0, (1, 1))}

        for i, model in enumerate(self.models):
            for j, dataset in enumerate(self.datasets):
                ax = axes[i, j]
                subset = funnel_df[(funnel_df["model"] == model) & (funnel_df["dataset"] == dataset)]
                dims_present = sorted(subset["truncate_dim"].unique())

                for method in self.methods:
                    for dim in dims_present:
                        data = subset[(subset["method"] == method) & (subset["truncate_dim"] == dim)]
                        data = data.sort_values("funnel_factor")
                        if len(data) < 2:
                            continue
                        ls = dim_styles.get(dim, "-")
                        ax.plot(data["funnel_factor"], data[metric],
                                marker=self._markers.get(method, "o"),
                                color=self._colors.get(method, "#95a5a6"),
                                linestyle=ls, linewidth=2, markersize=5,
                                label=f"{method} {dim}d")

                ax.set_xlabel("Funnel factor")
                ax.set_ylabel(metric.upper())
                ax.set_title(f"{model} | {dataset}")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=6, ncol=2)

        plt.tight_layout()
        if save:
            self._save(fig, f"funnel_factor_sweep_{metric.replace('@', '_')}")
        return fig

    # ── Pareto front (time) ──────────────────────────────────────────────────

    def plot_pareto_front(self, cost: str = "total_time", metric: str = "ndcg@10",
                          save: bool = True) -> Optional[plt.Figure]:
        if cost == "total_time":
            if "search_sec" not in self.df.columns or "rescore_sec" not in self.df.columns:
                return None
            self.df["_cost"] = self.df["search_sec"] + self.df["rescore_sec"]
            cost_label = "Total time (s)"
        elif cost == "total_memory":
            if "index_mem_mb" not in self.df.columns or "rescore_vec_mem_mb" not in self.df.columns:
                return None
            self.df["_cost"] = self.df["index_mem_mb"] + self.df["rescore_vec_mem_mb"]
            cost_label = "Total memory (MB)"
        elif cost == "bytes_per_vector":
            self.df["_cost"] = [
                self._bytes_per_vector(row["method"], int(row["truncate_dim"]))
                for _, row in self.df.iterrows()
            ]
            cost_label = "Bytes per vector (RAM)"
        elif cost == "vespa_latency":
            if "vespa_search_avg_ms" not in self.df.columns:
                return None
            self.df["_cost"] = self.df["vespa_search_avg_ms"]
            cost_label = "Vespa search latency (ms)"
        else:
            raise ValueError(f"Unknown cost: {cost}")

        fig, axes = self._make_grid(
            f"Pareto Front: {cost_label} vs {metric.upper()}", width=7, height=5.5)

        for i, model in enumerate(self.models):
            for j, dataset in enumerate(self.datasets):
                ax = axes[i, j]
                subset = self.df[(self.df["model"] == model) & (self.df["dataset"] == dataset)]

                all_points = []
                for _, row in subset.iterrows():
                    c = row["_cost"]
                    q = row[metric]
                    m = row["method"]
                    d = int(row["truncate_dim"])
                    all_points.append((c, q, m, d))

                if not all_points:
                    ax.set_title(f"{model} | {dataset}\n(no data)")
                    continue

                # Plot all points (faded)
                methods_shown = set()
                for c, q, m, d in all_points:
                    ax.scatter(c, q,
                               marker=self._markers.get(m, "o"),
                               color=self._colors.get(m, "#95a5a6"),
                               s=40, alpha=0.35, zorder=2)
                    methods_shown.add(m)

                # Pareto front
                sorted_pts = sorted(all_points, key=lambda p: (p[0], -p[1]))
                pareto = []
                best_quality = -1
                for c, q, m, d in sorted_pts:
                    if q > best_quality:
                        pareto.append((c, q, m, d))
                        best_quality = q

                # Step-function frontier
                pareto_x = [p[0] for p in pareto]
                pareto_y = [p[1] for p in pareto]
                step_x, step_y = [], []
                for pi in range(len(pareto)):
                    step_x.append(pareto_x[pi])
                    step_y.append(pareto_y[pi])
                    if pi < len(pareto) - 1:
                        step_x.append(pareto_x[pi + 1])
                        step_y.append(pareto_y[pi])
                ax.plot(step_x, step_y, "k-", linewidth=1.5, alpha=0.3, zorder=1)

                # Highlight Pareto-optimal points
                for pi, (c, q, m, d) in enumerate(pareto):
                    ax.scatter(c, q,
                               marker=self._markers.get(m, "o"),
                               color=self._colors.get(m, "#95a5a6"),
                               s=120, alpha=1.0, zorder=3,
                               edgecolors="black", linewidths=1.5)
                    label = f"{d}d"
                    y_off = 8 if pi % 2 == 0 else -14
                    ax.annotate(label, (c, q),
                                textcoords="offset points", xytext=(8, y_off),
                                fontsize=6.5, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                          alpha=0.85, edgecolor="gray"),
                                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4))

                ax.set_xlabel(cost_label)
                ax.set_ylabel(metric.upper())
                ax.set_title(f"{model} | {dataset}")
                if "memory" in cost or "bytes" in cost or "latency" in cost:
                    ax.set_xscale("log")
                ax.grid(True, alpha=0.3)
                self._method_legend(ax, methods_shown)

        self.df.drop(columns=["_cost"], inplace=True)
        plt.tight_layout()
        if save:
            self._save(fig, f"pareto_{cost}_{metric.replace('@', '_')}")
        return fig

    # ── Retention heatmap ────────────────────────────────────────────────────

    def plot_retention_heatmap(self, metric: str = "ndcg@10", save: bool = True) -> Optional[plt.Figure]:
        # Need float32 baseline
        float32_data = self.df[self.df["method"] == "float32"]
        if float32_data.empty:
            return None

        fig, axes = self._make_grid(
            f"Performance Retention (% of Float32 at Full Dim)", width=7, height=5)

        im = None
        for i, model in enumerate(self.models):
            for j, dataset in enumerate(self.datasets):
                ax = axes[i, j]
                subset = self.df[(self.df["model"] == model) & (self.df["dataset"] == dataset)]

                max_dim = subset["truncate_dim"].max()
                baseline = subset[(subset["method"] == "float32") & (subset["truncate_dim"] == max_dim)][metric].values
                if len(baseline) == 0:
                    ax.set_title(f"{model} | {dataset}\n(no baseline)")
                    ax.axis("off")
                    continue
                baseline = baseline[0]

                dims = sorted(subset["truncate_dim"].unique(), reverse=True)
                methods = [m for m in self.methods if m != "float32"]
                if not methods:
                    continue

                retention = np.zeros((len(dims), len(methods)))
                for di, dim in enumerate(dims):
                    for mi, method in enumerate(methods):
                        val = subset[(subset["truncate_dim"] == dim) & (subset["method"] == method)][metric].values
                        if len(val) > 0 and baseline > 0:
                            retention[di, mi] = (val[0] / baseline) * 100

                im = ax.imshow(retention, cmap="RdYlGn", vmin=50, vmax=105, aspect="auto")
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(dims)))
                ax.set_yticklabels(dims)
                ax.set_ylabel("Dimension")
                ax.set_title(f"{model} | {dataset}")

                for di in range(len(dims)):
                    for mi in range(len(methods)):
                        val = retention[di, mi]
                        if val > 0:
                            color = "white" if val < 70 or val > 95 else "black"
                            ax.text(mi, di, f"{val:.0f}%", ha="center", va="center",
                                    color=color, fontsize=7)

        if im is not None:
            fig.colorbar(im, ax=axes, label="Retention %", shrink=0.6)
        plt.tight_layout()
        if save:
            self._save(fig, f"retention_heatmap_{metric.replace('@', '_')}")
        return fig

    # ── Plot all ─────────────────────────────────────────────────────────────

    def plot_all(self, save: bool = True):
        print(f"\nGenerating plots in {self.output_dir}/\n")

        metrics = [col for col in self.df.columns if col.startswith(("ndcg@", "recall@"))]
        for metric in metrics:
            self.plot_dim_sweep(metric, save)
            self.plot_funnel_factor_sweep(metric, save)
            self.plot_pareto_front("total_time", metric, save)
            self.plot_pareto_front("total_memory", metric, save)
            self.plot_pareto_front("bytes_per_vector", metric, save)
            self.plot_pareto_front("vespa_latency", metric, save)
            self.plot_retention_heatmap(metric, save)



        print(f"\nAll plots saved to {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("results_path", type=str, help="Path to results.json")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    viz = Visualizer(args.results_path)
    viz.plot_all(save=True)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
