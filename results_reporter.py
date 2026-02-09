"""Results aggregation and reporting."""
import pandas as pd
from typing import List
from pathlib import Path
import json

from config import MODELS
from experiment_runner import ExperimentResult


class ResultsReporter:
    """Aggregate and report experiment results."""

    def __init__(self, results: List[ExperimentResult], experiment_id: str = None):
        self.results = results
        self.experiment_id = experiment_id
        self.df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        records = []
        for r in self.results:
            record = {
                "model": r.model,
                "dataset": r.dataset,
                "quantization": r.quantization,
                "rotation": getattr(r, "rotation", "none"),
                "matryoshka_dim": r.matryoshka_dim,
                "latency_seconds": r.latency_seconds,
                "memory_mb": r.memory_bytes / 1024 / 1024,
                "compression_ratio": self._compute_compression(r),
            }
            record.update(r.metrics)
            records.append(record)
        return pd.DataFrame(records)

    def _compute_compression(self, result: ExperimentResult) -> float:
        """Compute compression ratio vs float32 at full dimension."""
        full_dim = MODELS[result.model].dim
        baseline_bytes = full_dim * 4  # float32 per vector

        if result.quantization == "float32":
            actual_bytes = result.matryoshka_dim * 4
        elif result.quantization == "int8":
            actual_bytes = result.matryoshka_dim * 1
        elif result.quantization in ["binary", "binary_rescore"]:
            actual_bytes = result.matryoshka_dim // 8
        else:
            actual_bytes = baseline_bytes

        return baseline_bytes / actual_bytes if actual_bytes > 0 else 1.0

    def summary_table(self, sort_by: str = "recall@10") -> pd.DataFrame:
        """Generate summary table sorted by a metric."""
        return self.df.sort_values(sort_by, ascending=False)

    def pivot_by_quantization(self, metric: str = "recall@10") -> pd.DataFrame:
        """Pivot table showing metric across quantization methods."""
        return self.df.pivot_table(
            values=metric,
            index=["model", "dataset", "matryoshka_dim"],
            columns="quantization",
            aggfunc="first",
        )

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print("\n" + "=" * 100)
        print("EXPERIMENT RESULTS SUMMARY")
        if self.experiment_id:
            print(f"Experiment: {self.experiment_id}")
        print("=" * 100)

        for model in self.df["model"].unique():
            print(f"\n--- {model} ---")
            model_df = self.df[self.df["model"] == model]

            for dataset in model_df["dataset"].unique():
                print(f"\n  Dataset: {dataset}")
                ds_df = model_df[model_df["dataset"] == dataset]

                # Select columns to display
                cols = [
                    "matryoshka_dim",
                    "quantization",
                    "rotation",
                    "recall@10",
                    "recall@100",
                    "ndcg@10",
                    "latency_seconds",
                    "memory_mb",
                    "compression_ratio",
                ]
                available_cols = [c for c in cols if c in ds_df.columns]

                # Format and print
                display_df = ds_df[available_cols].copy()
                for col in ["recall@10", "recall@100", "ndcg@10"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                if "latency_seconds" in display_df.columns:
                    display_df["latency_seconds"] = display_df["latency_seconds"].apply(
                        lambda x: f"{x:.3f}"
                    )
                if "memory_mb" in display_df.columns:
                    display_df["memory_mb"] = display_df["memory_mb"].apply(
                        lambda x: f"{x:.2f}"
                    )
                if "compression_ratio" in display_df.columns:
                    display_df["compression_ratio"] = display_df["compression_ratio"].apply(
                        lambda x: f"{x:.1f}x"
                    )

                print(display_df.to_string(index=False))

    def save_results(self, output_dir: Path) -> None:
        """Save results to files.

        Note: JSON is saved by ExperimentRunner incrementally for crash recovery.
        This method only saves CSV and Markdown summaries.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full CSV
        csv_path = output_dir / "results.csv"
        self.df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")

        # JSON is saved by ExperimentRunner._save_results() - don't duplicate
        json_path = output_dir / "results.json"
        if json_path.exists():
            print(f"JSON already saved to {json_path}")
        else:
            # Fallback: save JSON if ExperimentRunner didn't (shouldn't happen)
            json_data = {
                "experiment_id": self.experiment_id,
                "results": [r.to_dict() for r in self.results],
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)
            print(f"Saved JSON to {json_path}")

        # Save summary markdown
        md_path = output_dir / "summary.md"
        with open(md_path, "w") as f:
            f.write("# Embedding Quantization Experiment Results\n\n")
            if self.experiment_id:
                f.write(f"**Experiment ID:** `{self.experiment_id}`\n\n")
            f.write("## Full Results\n\n")
            f.write(self.df.to_markdown(index=False))
            f.write("\n\n## Best Results by Model\n\n")

            # Add best results summary
            for model in self.df["model"].unique():
                f.write(f"\n### {model}\n\n")
                model_df = self.df[self.df["model"] == model]
                best_idx = model_df["recall@10"].idxmax()
                best = model_df.loc[best_idx]
                f.write(f"- Best Recall@10: {best['recall@10']:.4f}\n")
                f.write(f"- Config: dim={best['matryoshka_dim']}, quant={best['quantization']}\n")
                f.write(f"- Compression: {best['compression_ratio']:.1f}x\n")

        print(f"Saved Markdown to {md_path}")
