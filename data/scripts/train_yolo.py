"""Train and evaluate the project three-class YOLOv8 detector."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/yolo_dataset/dataset.yaml"))
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/yolo_student")
    parser.add_argument("--name", default="final")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[2] / "runs" / "ultralytics_config"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    config_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(args.pretrained)
    if args.test:
        metrics = model.val(data=str(args.data), split="test", imgsz=args.imgsz,
                            batch=args.batch, device=args.device, workers=0,
                            project=args.project, name=args.name, exist_ok=True, plots=True)
        print({"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
               "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)})
        return
    model.train(data=str(args.data), epochs=args.epochs, patience=5, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0, seed=42,
                project=args.project, name=args.name, exist_ok=True)


if __name__ == "__main__":
    main()
