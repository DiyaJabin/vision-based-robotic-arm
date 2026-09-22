from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(frame: pd.DataFrame, output_dir: str | Path | None = None) -> list[Path]:
    if frame is None or frame.empty:
        print("No experiment data found; no plots generated.")
        return []

    output_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "results" / "plots"
    output_path.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    if "method" not in frame.columns:
        return []
    for column, default in (("placement_success", False), ("perception_latency_ms", 0.0), ("retry_count", 0)):
        if column not in frame.columns:
            frame[column] = default
    methods = sorted(frame["method"].dropna().unique().tolist())

    if methods:
        summary = frame.groupby("method", dropna=False).agg(
            success_rate=("placement_success", lambda s: float(s.fillna(0).astype(bool).mean()) if len(s) else 0.0),
            avg_latency=("perception_latency_ms", lambda s: float(s.fillna(0.0).mean()) if len(s) else 0.0),
            placement_rate=("placement_success", lambda s: float(s.fillna(0).astype(bool).mean()) if len(s) else 0.0),
            retry_count=("retry_count", lambda s: float(s.fillna(0).sum()) if len(s) else 0.0),
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(summary.index, summary["success_rate"], color="#4c78a8")
        ax.set_title("Success Rate by Method")
        ax.set_ylabel("Success Rate")
        ax.set_xlabel("Method")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        path = output_path / "success_rate_by_method.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(summary.index, summary["avg_latency"], color="#f58518")
        ax.set_title("Average Perception Latency by Method")
        ax.set_ylabel("Latency (ms)")
        ax.set_xlabel("Method")
        fig.tight_layout()
        path = output_path / "average_latency_by_method.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)

    return generated


def main():
    csv_path = Path(__file__).resolve().parent / "results" / "trial_results.csv"
    if not csv_path.exists():
        print("No experiment CSV results found in experiments/results/. No plots generated.")
        return 0
    frame = pd.read_csv(csv_path)
    plots = plot_results(frame)
    if not plots:
        print("No experiment data found; no plots generated.")
        return 0
    print(f"Saved {len(plots)} plots under {Path(__file__).resolve().parent / 'results' / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
