"""Capture raw detector output from the integrated PyBullet scene."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import pybullet as p

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.yolo_detector import YoloDetector
from simulation import camera, scene
from simulation.run_pick_and_place import PipelineConfig, SimulationAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/yolo/best.pt")
    parser.add_argument("--output", type=Path, default=Path("runs/runtime_diagnostics"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = scene.connect_simulation(use_gui=False)
    try:
        data = scene.load_complete_scene()
        targets = {body: name for name, body in data["objects"].items()}
        adapter = SimulationAdapter(data, targets, PipelineConfig(mode="yolo", custom_weights=args.weights),
                                    client_id=client, detector=YoloDetector(custom_weights=args.weights))
        adapter.prepare_observation()
        frame = adapter.camera.capture_bgr()
        cv2.imwrite(str(args.output / "runtime_frame.png"), frame)
        detections = adapter.detector.detect(frame)
        print(f"frame={frame.shape[1]}x{frame.shape[0]} saved={args.output / 'runtime_frame.png'}")
        print("targets:")
        for body, name in targets.items():
            position, _ = p.getBasePositionAndOrientation(body)
            print(f"  body={body} class={name} world_xy=({position[0]:.3f},{position[1]:.3f})")
        print("detections:")
        for item in detections:
            print(f"  class={item.class_name} confidence={item.confidence:.4f} "
                  f"bbox=({item.bbox.x:.1f},{item.bbox.y:.1f},{item.bbox.width:.1f},{item.bbox.height:.1f}) "
                  f"centre=({item.centre.u:.1f},{item.centre.v:.1f}) "
                  f"pixels={item.bbox.width * item.bbox.height:.1f}")
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
