from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _boolean_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame.get(column, pd.Series(False, index=frame.index))
    if values.dtype == bool:
        return values
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float) != 0
    return values.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def plot_results(frame: pd.DataFrame, output_dir: str | Path | None = None) -> list[Path]:
    if frame is None or frame.empty:
        print("No experiment data found; no plots generated.")
        return []

    output_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "results" / "plots"
    output_path.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    if "method" not in frame.columns:
        return []
    methods = [method for method in ("opencv", "yolo", "hybrid") if method in set(frame["method"].dropna())]
    methods.extend(method for method in sorted(frame["method"].dropna().unique()) if method not in methods)

    if methods:
        summary = frame.groupby("method", dropna=False).agg(
            execution_time_s=("total_execution_time_s", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).mean()) if len(s) else 0.0),
        )
        if "grasp_success_rate" in frame.columns:
            summary["grasp_success_rate"] = frame.groupby("method")["grasp_success_rate"].mean()
        else:
            summary["grasp_success_rate"] = frame.assign(_grasp_success=_boolean_series(frame, "grasp_success")).groupby("method")["_grasp_success"].mean()
        summary = summary.reindex(methods).fillna(0.0)
        labels = {"opencv": "OpenCV", "yolo": "YOLO-only", "hybrid": "Hybrid"}
        display_names = [labels.get(method, method) for method in methods]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(display_names, summary.loc[methods, "grasp_success_rate"] * 100, color="#4c78a8")
        ax.set_title("Grasp Success Rate by Method")
        ax.set_ylabel("Grasp success (%)")
        ax.set_xlabel("Method")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        path = output_path / "grasp_success_rate_by_method.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(display_names, summary.loc[methods, "execution_time_s"], color="#f58518")
        ax.set_title("Mean Execution Time by Method")
        ax.set_ylabel("Execution time (s)")
        ax.set_xlabel("Method")
        fig.tight_layout()
        path = output_path / "mean_execution_time_by_method.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate plots from recorded experiment trials.")
    parser.add_argument("--input-csv", type=Path, default=Path(__file__).resolve().parent / "results" / "trial_results.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "plots")
    args = parser.parse_args()
    csv_path = args.input_csv
    if not csv_path.exists():
        print(f"No experiment CSV results found at {csv_path}. No plots generated.")
        return 0
    frame = pd.read_csv(csv_path)
    plots = plot_results(frame, args.output_dir)
    if not plots:
        print("No experiment data found; no plots generated.")
        return 0
    print(f"Saved {len(plots)} plots under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
