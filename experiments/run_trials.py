from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from experiments.logger import TrialLogger


@dataclass(frozen=True)
class TrialConfig:
    trials: int = 10
    seed: int | None = None
    method: str = "hybrid"
    randomize_rotation: bool = True
    output_dir: str | Path = "experiments/results"

    def __post_init__(self):
        if self.trials <= 0:
            raise ValueError("Trial count must be positive.")
        if self.method not in {"opencv", "yolo", "hybrid"}:
            raise ValueError("Method must be one of: opencv, yolo, hybrid.")


def describe_runtime_status(missing_runtime: bool = True) -> str:
    return "SKIPPED / NOT AVAILABLE" if missing_runtime else "AVAILABLE"


def runtime_available() -> bool:
    try:
        import pybullet  # noqa: F401
        from simulation import scene  # noqa: F401
        return True
    except Exception:
        return False


def randomized_positions(rng: random.Random, count: int):
    positions = []
    for _ in range(count):
        x = rng.uniform(0.35, 0.72)
        y = rng.uniform(-0.22, 0.22)
        yaw = rng.uniform(0.0, 3.14159 * 2.0) if True else 0.0
        positions.append((x, y, yaw))
    return positions


def run_trials(config: TrialConfig | None = None) -> list[dict]:
    cfg = config or TrialConfig()
    logger = TrialLogger(cfg.output_dir)
    rng = random.Random(cfg.seed)

    if not runtime_available():
        print(f"Method {cfg.method}: {describe_runtime_status(missing_runtime=True)}")
        return []

    rows = []
    for idx in range(cfg.trials):
        pos = randomized_positions(rng, 1)[0]
        row = {
            "trial_id": idx + 1,
            "method": cfg.method,
            "num_objects": 1,
            "objects_detected": 0,
            "valid_detections": 0,
            "selected_object": None,
            "selected_class": None,
            "detection_confidence": 0.0,
            "validation_score": None,
            "graspability_score": None,
            "perception_latency_ms": 0.0,
            "ik_feasible": False,
            "grasp_attempted": False,
            "grasp_success": False,
            "placement_success": False,
            "retry_count": 0,
            "failure_reason": "runtime_unavailable",
            "total_execution_time_s": 0.0,
        }
        # Actual PyBullet execution is deferred until the environment is configured.
        # This utility keeps the evaluation interface available without fabricating results.
        logger.log_trial(row)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run randomized evaluation trials for the existing simulation pipeline.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--method", choices=("opencv", "yolo", "hybrid"), default="hybrid")
    parser.add_argument("--randomize-rotation", action="store_true")
    args = parser.parse_args()

    config = TrialConfig(
        trials=args.trials,
        seed=args.seed,
        method=args.method,
        randomize_rotation=args.randomize_rotation,
    )
    rows = run_trials(config)
    if not rows:
        print(f"Evaluation status: {describe_runtime_status(missing_runtime=True)}")
        return 0
    print(f"Completed {len(rows)} trial rows for method '{config.method}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
