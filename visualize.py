"""Visualization module for embedding quantization experiments.

Usage:
    python visualize.py results/minilm+mxbai+nomic_fiqa+nfco+scif/results.json

Or in code:
    from visualize import ExperimentVisualizer
    viz = ExperimentVisualizer("results/experiment/results.json")
    viz.plot_all()
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


# Style configuration
COLORS = {
    "float32": "#2ecc71",        # Green
    "int8": "#3498db",           # Blue
    "int8_asym": "#1abc9c",      # Teal
    "binary": "#e74c3c",         # Red
    "binary_asym": "#e67e22",    # Orange
    "binary_rescore": "#9b59b6", # Purple
    "binary_median": "#d35400",  # Dark orange
    "binary_median_asym": "#c0392b", # Dark red
    "quaternary_asym": "#8e44ad",    # Purple
    "lloyd_max_gauss": "#2c3e50",    # Dark blue-gray
}

MARKERS = {
    "float32": "o",
    "int8": "s",
    "int8_asym": "p",
    "binary": "^",
    "binary_asym": "v",
    "binary_rescore": "D",
    "binary_median": "h",
    "binary_median_asym": "H",
    "quaternary_asym": "P",
    "lloyd_max_gauss": "*",
}

MODEL_NAMES = {
    "mxbai-embed-large-v1": "MxBai Large",
    "nomic-embed-text-v1.5": "Nomic v1.5",
    "all-MiniLM-L6-v2": "MiniLM-L6",
}

QUANT_LABELS = {
    "float32": "Float32",
    "int8": "Int8 (4×)",
    "int8_asym": "Int8 Asym (4×)",
    "binary": "Binary (32×)",
    "binary_asym": "Binary Asym (32×)",
    "binary_rescore": "Binary+Rescore",
    "binary_median": "Binary Median (32×)",
    "binary_median_asym": "Binary Median Asym (32×)",
    "quaternary_asym": "Quaternary Asym (16×)",
    "lloyd_max_gauss": "Lloyd-Max Gauss (16×)",
}


# Canonical ordering for quantization types
QUANT_ORDER = ["float32", "int8", "int8_asym", "quaternary_asym", "lloyd_max_gauss", "binary", "binary_asym", "binary_rescore", "binary_median", "binary_median_asym"]


class ExperimentVisualizer:
    """Create visualizations from experiment results."""

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.output_dir = self.results_path.parent / "plots"
        self.output_dir.mkdir(exist_ok=True)

        with open(results_path) as f:
            data = json.load(f)
            self.results = data.get("results", data)

        self.df = self._to_dataframe()
        # Quantization types present in data, in canonical order
        present = set(self.df["quantization"].unique())
        self.quants = [q for q in QUANT_ORDER if q in present]

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        records = []
        for r in self.results:
            record = {
                "model": r["model"],
                "dataset": r["dataset"],
                "quantization": r["quantization"],
                "rotation": r.get("rotation", "none"),
                "dim": r["matryoshka_dim"],
                "memory_mb": r["memory_bytes"] / 1024 / 1024,
                "latency": r["latency_seconds"],
            }
            record.update(r["metrics"])
            records.append(record)
        return pd.DataFrame(records)

    def _compute_compression(self, model: str, dim: int, quant: str) -> float:
        """Compute compression ratio vs float32 at full dimension."""
        full_dims = {"mxbai-embed-large-v1": 1024, "nomic-embed-text-v1.5": 768, "all-MiniLM-L6-v2": 384}
        full_dim = full_dims.get(model, 1024)
        baseline = full_dim * 4

        if quant == "float32":
            actual = dim * 4
        elif quant.startswith("int8"):
            actual = dim * 1
        elif quant in ("quaternary_asym", "lloyd_max_gauss"):
            actual = max(1, dim // 4)
        else:  # binary variants
            actual = dim // 8

        return baseline / actual if actual > 0 else 1.0

    def plot_dim_vs_quality(self, metric: str = "ndcg@10", save: bool = True) -> plt.Figure:
        """
        Line plot: Matryoshka dimension vs quality for each quantization method.
        One subplot per model, one line per quantization.
        """
        models = self.df["model"].unique()
        datasets = self.df["dataset"].unique()

        fig, axes = plt.subplots(len(models), len(datasets), figsize=(5*len(datasets), 4*len(models)), squeeze=False)
        fig.suptitle(f"{metric.upper()} vs Embedding Dimension", fontsize=14, fontweight="bold")

        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                subset = self.df[(self.df["model"] == model) & (self.df["dataset"] == dataset)]

                for quant in self.quants:
                    data = subset[subset["quantization"] == quant].sort_values("dim")
                    if len(data) > 0:
                        ax.plot(data["dim"], data[metric],
                               marker=MARKERS[quant], color=COLORS[quant],
                               label=QUANT_LABELS[quant], linewidth=2, markersize=6)

                ax.set_xlabel("Dimension")
                ax.set_ylabel(metric.upper())
                ax.set_title(f"{MODEL_NAMES.get(model, model)} - {dataset}")
                ax.set_xscale("log", base=2)
                ax.set_xticks([64, 128, 256, 512, 768, 1024])
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

        plt.tight_layout()

        if save:
            path = self.output_dir / f"dim_vs_{metric.replace('@', '_')}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_compression_vs_quality(self, metric: str = "ndcg@10", save: bool = True) -> plt.Figure:
        """
        Scatter plot: Compression ratio vs quality (Pareto frontier style).
        """
        models = self.df["model"].unique()
        datasets = self.df["dataset"].unique()

        fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 5), squeeze=False)
        fig.suptitle(f"Compression vs {metric.upper()} (Pareto Frontier)", fontsize=14, fontweight="bold")

        for j, dataset in enumerate(datasets):
            ax = axes[0, j]

            for model in models:
                subset = self.df[(self.df["model"] == model) & (self.df["dataset"] == dataset)]

                for quant in self.quants:
                    data = subset[subset["quantization"] == quant]
                    if len(data) > 0:
                        compressions = [self._compute_compression(model, d, quant) for d in data["dim"]]
                        ax.scatter(compressions, data[metric],
                                  marker=MARKERS[quant], color=COLORS[quant],
                                  s=60, alpha=0.7, label=f"{MODEL_NAMES.get(model, model)[:6]} {QUANT_LABELS[quant]}")

            ax.set_xlabel("Compression Ratio (×)")
            ax.set_ylabel(metric.upper())
            ax.set_title(dataset)
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)
            # Only show legend for first plot
            if j == 0:
                ax.legend(fontsize=6, loc="lower left", ncol=2)

        plt.tight_layout()

        if save:
            path = self.output_dir / f"compression_vs_{metric.replace('@', '_')}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_retention_heatmap(self, metric: str = "ndcg@10", save: bool = True) -> Optional[plt.Figure]:
        """
        Heatmap: Performance retention (% of float32 baseline) for each dim × quantization.
        """
        models = self.df["model"].unique()
        datasets = self.df["dataset"].unique()

        # Check if we have float32 baseline data
        float32_data = self.df[self.df["quantization"] == "float32"]
        if len(float32_data) == 0:
            print("  Skipping retention heatmap (no float32 baseline)")
            return None

        fig, axes = plt.subplots(len(models), len(datasets), figsize=(6*len(datasets), 4*len(models)), squeeze=False)
        fig.suptitle(f"Performance Retention (% of Float32 at Full Dim)", fontsize=14, fontweight="bold")

        im = None
        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                subset = self.df[(self.df["model"] == model) & (self.df["dataset"] == dataset)]

                # Get baseline (float32 at max dim)
                max_dim = subset["dim"].max()
                baseline = subset[(subset["quantization"] == "float32") & (subset["dim"] == max_dim)][metric].values
                if len(baseline) == 0:
                    ax.set_title(f"{MODEL_NAMES.get(model, model)} - {dataset}\n(no float32 baseline)")
                    ax.axis("off")
                    continue
                baseline = baseline[0]

                dims = sorted(subset["dim"].unique(), reverse=True)
                quants = self.quants

                # Build retention matrix
                retention = np.zeros((len(dims), len(quants)))
                for di, dim in enumerate(dims):
                    for qi, quant in enumerate(quants):
                        val = subset[(subset["dim"] == dim) & (subset["quantization"] == quant)][metric].values
                        if len(val) > 0:
                            retention[di, qi] = (val[0] / baseline) * 100

                # Plot heatmap
                im = ax.imshow(retention, cmap="RdYlGn", vmin=50, vmax=105, aspect="auto")
                ax.set_xticks(range(len(quants)))
                ax.set_xticklabels([QUANT_LABELS[q] for q in quants], rotation=45, ha="right")
                ax.set_yticks(range(len(dims)))
                ax.set_yticklabels(dims)
                ax.set_ylabel("Dimension")
                ax.set_title(f"{MODEL_NAMES.get(model, model)} - {dataset}")

                # Add text annotations
                for di in range(len(dims)):
                    for qi in range(len(quants)):
                        val = retention[di, qi]
                        color = "white" if val < 70 or val > 95 else "black"
                        ax.text(qi, di, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=8)

        # Add colorbar only if we have at least one heatmap
        if im is not None:
            fig.colorbar(im, ax=axes, label="Retention %", shrink=0.6)
        plt.tight_layout()

        if save:
            path = self.output_dir / f"retention_heatmap_{metric.replace('@', '_')}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_quantization_comparison(self, dim: Optional[int] = None, metric: str = "ndcg@10", save: bool = True) -> plt.Figure:
        """
        Grouped bar chart: Compare quantization methods at a specific dimension.
        If dim is None, uses full dimension for each model.
        """
        datasets = self.df["dataset"].unique()
        models = self.df["model"].unique()

        fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 5), squeeze=False)
        title = f"Quantization Comparison" + (f" (dim={dim})" if dim else " (full dimension)")
        fig.suptitle(title, fontsize=14, fontweight="bold")

        quants = self.quants
        x = np.arange(len(models))
        width = 0.2

        for j, dataset in enumerate(datasets):
            ax = axes[0, j]

            for qi, quant in enumerate(quants):
                values = []
                for model in models:
                    subset = self.df[(self.df["model"] == model) &
                                    (self.df["dataset"] == dataset) &
                                    (self.df["quantization"] == quant)]
                    if dim:
                        subset = subset[subset["dim"] == dim]
                    else:
                        # Use max dim
                        subset = subset[subset["dim"] == subset["dim"].max()]

                    if len(subset) > 0:
                        values.append(subset[metric].values[0])
                    else:
                        values.append(0)

                ax.bar(x + qi * width, values, width, label=QUANT_LABELS[quant], color=COLORS[quant])

            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(dataset)
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels([MODEL_NAMES.get(m, m)[:8] for m in models])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save:
            dim_str = f"_dim{dim}" if dim else "_fullDim"
            path = self.output_dir / f"quant_comparison{dim_str}_{metric.replace('@', '_')}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_sweet_spot(self, target_retention: float = 0.95, save: bool = True) -> Optional[plt.Figure]:
        """
        Find and plot the 'sweet spot' - highest compression at target retention level.
        Similar to mixedbread's analysis.
        """
        # Check if we have float32 baseline data
        float32_data = self.df[self.df["quantization"] == "float32"]
        if len(float32_data) == 0:
            print(f"  Skipping sweet spot plot (no float32 baseline)")
            return None

        models = self.df["model"].unique()
        datasets = self.df["dataset"].unique()

        results = []
        for model in models:
            for dataset in datasets:
                subset = self.df[(self.df["model"] == model) & (self.df["dataset"] == dataset)]

                # Get baseline
                max_dim = subset["dim"].max()
                baseline = subset[(subset["quantization"] == "float32") & (subset["dim"] == max_dim)]["ndcg@10"].values
                if len(baseline) == 0:
                    continue
                baseline = baseline[0]

                for _, row in subset.iterrows():
                    retention = row["ndcg@10"] / baseline
                    compression = self._compute_compression(model, row["dim"], row["quantization"])

                    if retention >= target_retention:
                        results.append({
                            "model": model,
                            "dataset": dataset,
                            "quant": row["quantization"],
                            "dim": row["dim"],
                            "retention": retention,
                            "compression": compression,
                            "label": f"{row['quantization']}\nd={row['dim']}"
                        })

        if len(results) == 0:
            print(f"  Skipping sweet spot plot (no results meet {target_retention*100:.0f}% retention)")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Sweet Spot: Max Compression at ≥{target_retention*100:.0f}% Retention", fontsize=14, fontweight="bold")

        # Plot results
        df_results = pd.DataFrame(results)
        for model in models:
            model_data = df_results[df_results["model"] == model]
            if len(model_data) > 0:
                # Find best per dataset
                best = model_data.loc[model_data.groupby("dataset")["compression"].idxmax()]
                for _, row in best.iterrows():
                    ax.scatter(row["retention"] * 100, row["compression"],
                              s=100, label=f"{MODEL_NAMES.get(model, model)[:6]} - {row['dataset']}")
                    ax.annotate(row["label"], (row["retention"] * 100, row["compression"]),
                               textcoords="offset points", xytext=(5, 5), fontsize=7)

        ax.axvline(x=target_retention * 100, color="red", linestyle="--", alpha=0.5, label=f"{target_retention*100:.0f}% threshold")
        ax.set_xlabel("Performance Retention (%)")
        ax.set_ylabel("Compression Ratio (×)")
        ax.set_yscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout()

        if save:
            path = self.output_dir / f"sweet_spot_{int(target_retention*100)}pct.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_pareto_front(self, metric: str = "ndcg@10", save: bool = True,
                          exclude_rescore: bool = False) -> plt.Figure:
        """
        Pareto front: bytes per vector vs quality.
        One subplot per model×dataset. All quant×dim combos plotted,
        Pareto-optimal points highlighted and connected.
        Uses rotation=none only to keep the plot clean.
        If exclude_rescore=True, omits binary_rescore (which needs float32 corpus in memory).
        """
        models = sorted(self.df["model"].unique())
        datasets = sorted(self.df["dataset"].unique())
        n_models = len(models)
        n_datasets = len(datasets)

        fig, axes = plt.subplots(n_models, n_datasets,
                                 figsize=(6 * n_datasets, 5 * n_models), squeeze=False)
        suffix = " (no rescore)" if exclude_rescore else ""
        fig.suptitle(f"Pareto Front: Storage vs {metric.upper()}{suffix}", fontsize=14, fontweight="bold")

        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                subset = self.df[(self.df["model"] == model) &
                                 (self.df["dataset"] == dataset) &
                                 (self.df["rotation"] == "none")]
                if exclude_rescore:
                    subset = subset[subset["quantization"] != "binary_rescore"]

                all_points = []  # (bytes_per_vec, quality, quant, dim)

                for _, row in subset.iterrows():
                    quant = row["quantization"]
                    dim = int(row["dim"])
                    quality = row[metric]

                    # Compute bytes per vector
                    if quant == "float32":
                        bpv = dim * 4
                    elif quant.startswith("int8"):
                        bpv = dim * 1
                    elif quant in ("quaternary_asym", "lloyd_max_gauss"):
                        bpv = max(1, dim // 4)  # 2-bit: 4 codes per byte
                    else:  # binary variants
                        bpv = dim // 8

                    all_points.append((bpv, quality, quant, dim))

                if not all_points:
                    ax.set_title(f"{MODEL_NAMES.get(model, model)} - {dataset}\n(no data)")
                    continue

                # Plot all points (no per-point annotations to avoid clutter)
                for bpv, quality, quant, dim in all_points:
                    ax.scatter(bpv, quality,
                               marker=MARKERS.get(quant, "o"),
                               color=COLORS.get(quant, "#95a5a6"),
                               s=40, alpha=0.35, zorder=2)

                # Compute Pareto front: maximize quality, minimize bytes
                # Sort by bytes ascending, then sweep for quality
                sorted_pts = sorted(all_points, key=lambda p: (p[0], -p[1]))
                pareto = []
                best_quality = -1
                for bpv, quality, quant, dim in sorted_pts:
                    if quality > best_quality:
                        pareto.append((bpv, quality, quant, dim))
                        best_quality = quality

                # Plot Pareto front
                pareto_x = [p[0] for p in pareto]
                pareto_y = [p[1] for p in pareto]
                # Step-function style: for each Pareto point, extend horizontally
                # to the next point's x, showing the frontier clearly
                step_x, step_y = [], []
                for pi in range(len(pareto)):
                    step_x.append(pareto_x[pi])
                    step_y.append(pareto_y[pi])
                    if pi < len(pareto) - 1:
                        step_x.append(pareto_x[pi + 1])
                        step_y.append(pareto_y[pi])
                ax.plot(step_x, step_y, "k-", linewidth=1.5, alpha=0.3, zorder=1)

                # Highlight Pareto-optimal points
                pareto_annotations = []
                for bpv, quality, quant, dim in pareto:
                    ax.scatter(bpv, quality,
                               marker=MARKERS.get(quant, "o"),
                               color=COLORS.get(quant, "#95a5a6"),
                               s=120, alpha=1.0, zorder=3,
                               edgecolors="black", linewidths=1.5)
                    short_label = quant.replace("binary_", "b_").replace("float32", "f32")
                    pareto_annotations.append((bpv, quality, f"{short_label}\n{dim}d"))

                # Place annotations with alternating offsets to reduce overlap
                for idx, (bpv, quality, label) in enumerate(pareto_annotations):
                    y_off = 8 if idx % 2 == 0 else -14
                    ax.annotate(label, (bpv, quality),
                                textcoords="offset points", xytext=(8, y_off),
                                fontsize=6.5, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="gray"),
                                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4))

                ax.set_xlabel("Bytes per vector")
                ax.set_ylabel(metric.upper())
                ax.set_xscale("log", base=2)
                ax.set_title(f"{MODEL_NAMES.get(model, model)} - {dataset}")
                ax.grid(True, alpha=0.3)

                # Add compression ratio as secondary x-axis labels
                full_dims = {"mxbai-embed-large-v1": 1024, "nomic-embed-text-v1.5": 768, "all-MiniLM-L6-v2": 384}
                full_bpv = full_dims.get(model, 1024) * 4

                # Legend: one entry per quant type present, in canonical order
                from matplotlib.lines import Line2D
                seen_quants = set(p[2] for p in all_points)
                handles = []
                for q in QUANT_ORDER:
                    if q in seen_quants:
                        handles.append(Line2D([0], [0],
                                              marker=MARKERS.get(q, "o"),
                                              color=COLORS.get(q, "#95a5a6"),
                                              linestyle="None", markersize=6,
                                              label=QUANT_LABELS.get(q, q)))
                ax.legend(handles=handles, fontsize=7, loc="lower right")

        plt.tight_layout()

        if save:
            rescore_tag = "_no_rescore" if exclude_rescore else ""
            path = self.output_dir / f"pareto_front_{metric.replace('@', '_')}{rescore_tag}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_rotation_comparison(self, metric: str = "ndcg@10", save: bool = True) -> Optional[plt.Figure]:
        """
        Compare rotation methods for quantized representations.
        Only generates plot if multiple rotation methods are present.
        """
        # Check if we have rotation data
        if "rotation" not in self.df.columns:
            return None

        rotations = [r for r in self.df["rotation"].unique() if r != "none"]
        if len(rotations) == 0:
            return None

        models = self.df["model"].unique()
        datasets = self.df["dataset"].unique()

        fig, axes = plt.subplots(len(models), len(datasets), figsize=(5*len(datasets), 4*len(models)), squeeze=False)
        fig.suptitle(f"Rotation Impact on Quantization ({metric.upper()})", fontsize=14, fontweight="bold")

        rot_colors = {"none": "#e74c3c", "qr": "#3498db", "hadamard": "#2ecc71"}
        rot_markers = {"none": "^", "qr": "s", "hadamard": "D"}

        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                subset = self.df[(self.df["model"] == model) & (self.df["dataset"] == dataset)]

                # Plot for each quantization that has rotation data
                quant_linestyles = {
                    "int8": "--", "int8_asym": "--",
                    "binary": "-", "binary_asym": "-",
                }
                quant_suffixes = {
                    "int8": "int8", "int8_asym": "int8-asym",
                    "binary": "bin", "binary_asym": "bin-asym",
                }
                for quant in ["int8", "int8_asym", "binary", "binary_asym"]:
                    quant_data = subset[subset["quantization"] == quant]
                    if len(quant_data) == 0:
                        continue

                    is_asym = quant.endswith("_asym")
                    for rot in ["none"] + list(rotations):
                        data = quant_data[quant_data["rotation"] == rot].sort_values("dim")
                        if len(data) > 0:
                            linestyle = quant_linestyles.get(quant, "-")
                            marker = rot_markers.get(rot, "o")
                            if is_asym:
                                marker = "x" if marker == "^" else marker
                            ax.plot(data["dim"], data[metric],
                                   marker=marker,
                                   color=rot_colors.get(rot, "#95a5a6"),
                                   linestyle=linestyle,
                                   label=f"{quant_suffixes[quant]} ({rot})",
                                   linewidth=2 if not is_asym else 1.5,
                                   markersize=6,
                                   alpha=1.0 if not is_asym else 0.7)

                ax.set_xlabel("Dimension")
                ax.set_ylabel(metric.upper())
                ax.set_title(f"{MODEL_NAMES.get(model, model)} - {dataset}")
                ax.set_xscale("log", base=2)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc="lower right")

        plt.tight_layout()

        if save:
            path = self.output_dir / f"rotation_comparison_{metric.replace('@', '_')}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_rotation_effect(self, metric: str = "ndcg@10", save: bool = True) -> Optional[plt.Figure]:
        """
        Bar chart showing the effect of rotation (% change from no rotation).
        """
        if "rotation" not in self.df.columns:
            return None

        rotations = [r for r in self.df["rotation"].unique() if r != "none"]
        if len(rotations) == 0:
            return None

        models = self.df["model"].unique()
        datasets = self.df["dataset"].unique()
        quants = [q for q in ["int8", "int8_asym", "binary", "binary_asym"] if q in self.df["quantization"].values]

        # Compute effect for each combination
        effects = []
        for model in models:
            for dataset in datasets:
                for quant in quants:
                    subset = self.df[(self.df["model"] == model) &
                                    (self.df["dataset"] == dataset) &
                                    (self.df["quantization"] == quant)]

                    # Get baseline (no rotation)
                    baseline = subset[subset["rotation"] == "none"][metric].values
                    if len(baseline) == 0:
                        continue
                    baseline = baseline[0]

                    for rot in rotations:
                        rotated = subset[subset["rotation"] == rot][metric].values
                        if len(rotated) > 0:
                            pct_change = ((rotated[0] - baseline) / baseline) * 100
                            effects.append({
                                "model": MODEL_NAMES.get(model, model)[:8],
                                "dataset": dataset,
                                "quant": quant,
                                "rotation": rot,
                                "pct_change": pct_change,
                            })

        if len(effects) == 0:
            return None

        effects_df = pd.DataFrame(effects)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"Rotation Effect on {metric.upper()} (% Change from No Rotation)", fontsize=14, fontweight="bold")

        # Create labels for x-axis
        effects_df["label"] = effects_df["model"] + "\n" + effects_df["dataset"] + "\n" + effects_df["quant"]

        x = np.arange(len(effects_df))
        rot_colors_map = {"qr": "#3498db", "hadamard": "#2ecc71"}

        bars = ax.bar(x, effects_df["pct_change"], color=[rot_colors_map.get(r, "#95a5a6") for r in effects_df["rotation"]])

        # Color bars based on positive/negative
        for bar, val in zip(bars, effects_df["pct_change"]):
            if val < 0:
                bar.set_alpha(0.6)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(effects_df["label"], fontsize=8)
        ax.set_ylabel(f"% Change in {metric.upper()}")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, effects_df["pct_change"]):
            height = bar.get_height()
            ax.annotate(f"{val:+.1f}%",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -10),
                       textcoords="offset points",
                       ha="center", va="bottom" if height >= 0 else "top",
                       fontsize=8)

        # Add legend for rotation types
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=rot_colors_map[r], label=r) for r in rotations if r in rot_colors_map]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        if save:
            path = self.output_dir / f"rotation_effect_{metric.replace('@', '_')}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

        return fig

    def plot_all(self, save: bool = True):
        """Generate all plots."""
        print(f"\nGenerating plots in {self.output_dir}/\n")

        self.plot_dim_vs_quality("ndcg@10", save)
        self.plot_dim_vs_quality("recall@10", save)
        self.plot_compression_vs_quality("ndcg@10", save)
        self.plot_pareto_front("ndcg@10", save)
        self.plot_pareto_front("recall@10", save)
        self.plot_pareto_front("ndcg@10", save, exclude_rescore=True)
        self.plot_pareto_front("recall@10", save, exclude_rescore=True)
        self.plot_retention_heatmap("ndcg@10", save)
        self.plot_quantization_comparison(None, "ndcg@10", save)
        self.plot_quantization_comparison(256, "ndcg@10", save)
        self.plot_sweet_spot(0.95, save)
        self.plot_sweet_spot(0.90, save)
        # Generate rotation plots for all accuracy metrics present in the data
        accuracy_metrics = [c for c in self.df.columns if c.startswith(("ndcg@", "recall@", "mrr@", "map@"))]
        for m in accuracy_metrics:
            self.plot_rotation_comparison(m, save)
            self.plot_rotation_effect(m, save)

        print(f"\nAll plots saved to {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("results_path", type=str, help="Path to results.json")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    viz = ExperimentVisualizer(args.results_path)
    viz.plot_all(save=True)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
