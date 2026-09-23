from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



@dataclass(frozen=True)
class TrialConfig:
    trials: int = 10
    seed: int | None = None
    method: str = "hybrid"
    randomize_rotation: bool = True
    output_dir: str | Path = "experiments/results"
    weights: str = "yolov8n.pt"
    custom_weights: str | Path | None = None

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
        import pandas  # noqa: F401
        from simulation import scene  # noqa: F401
        return True
    except Exception:
        return False


def randomized_positions(rng: random.Random, count: int):
    positions = []
    for _ in range(count):
        x = rng.uniform(0.35, 0.72)
        y = rng.uniform(-0.22, 0.22)
        yaw = rng.uniform(0.0, 3.141592653589793 * 2.0)
        positions.append((x, y, yaw))
    return positions


def run_trials(config: TrialConfig | None = None) -> list[dict]:
    cfg = config or TrialConfig()
    rng = random.Random(cfg.seed)

    if not runtime_available():
        print(f"Method {cfg.method}: {describe_runtime_status(missing_runtime=True)}")
        return []

    import pybullet as p
    from experiments.logger import TrialLogger
    from core.config import CLASS_IDS
    from simulation import scene
    from simulation.run_pick_and_place import PipelineConfig, run_scene

    logger = TrialLogger(cfg.output_dir)
    rows = []
    mode = cfg.method if cfg.method in {"yolo", "hybrid"} else "baseline"
    for idx in range(cfg.trials):
        client = scene.connect_simulation(use_gui=False)
        try:
            data = scene.load_complete_scene()
            targets = {body: name for name, body in data["objects"].items()}
            poses = randomized_positions(rng, len(targets))
            for (body, name), (x, y, yaw) in zip(targets.items(), poses):
                if not cfg.randomize_rotation:
                    yaw = 0.0
                dimensions = {
                    "cube": scene.OBJECT_SIZE,
                    "cylinder": scene.CYLINDER_HEIGHT,
                    "box": scene.BOX_HALF_EXTENTS[2] * 2,
                }
                z = scene.TABLETOP_Z + dimensions[name] / 2.0
                p.resetBasePositionAndOrientation(body, (x, y, z), p.getQuaternionFromEuler((0, 0, yaw)))
            started = time.perf_counter()
            result = run_scene(data, targets, PipelineConfig(mode=mode, weights=cfg.weights,
                custom_weights=cfg.custom_weights), client_id=client)
            elapsed_s = time.perf_counter() - started
            placed = [pick for pick in result.picks if pick.status == "placed"]
            row = {
                "trial_id": idx + 1, "method": cfg.method, "num_objects": len(targets),
                "objects_detected": len(placed) + len(result.rejections),
                "valid_detections": len(placed),
                "selected_object": placed[0].object_id if placed else None,
                "selected_class": placed[0].class_name if placed else None,
                "detection_confidence": 0.0, "validation_score": None,
                "graspability_score": None, "perception_latency_ms": None,
                "ik_feasible": bool(placed), "grasp_attempted": bool(result.picks),
                "grasp_success": bool(placed),
                "placement_success": result.status == "completed",
                "retry_count": max(len(result.picks) - len(placed), 0),
                "failure_reason": result.reason,
                "total_execution_time_s": elapsed_s,
            }
            logger.log_trial(row)
            rows.append(row)
        finally:
            if p.isConnected(client):
                p.disconnect(client)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run randomized evaluation trials for the existing simulation pipeline.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--method", choices=("opencv", "yolo", "hybrid"), default="hybrid")
    parser.add_argument("--no-randomize-rotation", action="store_false", dest="randomize_rotation")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--custom-weights")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.set_defaults(randomize_rotation=True)
    args = parser.parse_args()

    config = TrialConfig(
        trials=args.trials,
        seed=args.seed,
        method=args.method,
        randomize_rotation=args.randomize_rotation,
        weights=args.weights,
        custom_weights=args.custom_weights,
        output_dir=args.output_dir,
    )
    rows = run_trials(config)
    if not rows:
        print(f"Evaluation status: {describe_runtime_status(missing_runtime=True)}")
        return 0
    print(f"Completed {len(rows)} trial rows for method '{config.method}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
