"""Build a split YOLO dataset from visible PyBullet target pixels.

Labels come from the camera segmentation buffer and known target body IDs.
Robot, table, destination, and background pixels are never emitted as target
labels. The script reuses the existing randomized scene and settling logic.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import CLASS_IDS
from simulation import camera, scene
from data.scripts.generate_dataset import (
    DEFAULT_SEED, MAX_SETTLE_STEPS, SETTLE_CHECK_INTERVAL,
    CONSECUTIVE_STABLE_CHECKS, LINEAR_VELOCITY_THRESHOLD,
    ANGULAR_VELOCITY_THRESHOLD, create_random_scene, wait_for_objects_to_settle,
)


def body_pixels(segmentation: np.ndarray, body_id: int) -> np.ndarray:
    """Return visible pixel coordinates for a body, stripping link bits."""
    encoded = segmentation.astype(np.int64)
    visible_body = np.bitwise_and(encoded, 0xFFFFFF) == int(body_id)
    return np.argwhere(visible_body)


def yolo_box(pixels: np.ndarray, width: int, height: int) -> tuple[float, float, float, float] | None:
    if pixels.size == 0:
        return None
    y_min, x_min = pixels.min(axis=0)
    y_max, x_max = pixels.max(axis=0)
    x_min, x_max = max(0, int(x_min)), min(width - 1, int(x_max))
    y_min, y_max = max(0, int(y_min)), min(height - 1, int(y_max))
    if x_max <= x_min or y_max <= y_min:
        return None
    cx = ((x_min + x_max) / 2) / width
    cy = ((y_min + y_max) / 2) / height
    bw = (x_max - x_min) / width
    bh = (y_max - y_min) / height
    values = (cx, cy, bw, bh)
    if not all(0.0 <= value <= 1.0 for value in values) or bw <= 0 or bh <= 0:
        return None
    return values


def labels_for_scene(metadata, segmentation, width, height):
    labels = []
    for item in metadata:
        box = yolo_box(body_pixels(segmentation, int(item["object_id"])), width, height)
        if box is not None:
            labels.append((int(item["class_id"]), box))
    return labels


def write_label(path: Path, labels) -> None:
    path.write_text("".join(
        f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n"
        for class_id, (cx, cy, width, height) in labels
    ), encoding="utf-8")


def preview(image, labels, path):
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for class_id, (cx, cy, bw, bh) in labels:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(canvas, CLASS_IDS and {v: k for k, v in CLASS_IDS.items()}[class_id],
                    (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    cv2.imwrite(str(path), canvas)


def build_dataset(count: int, output_dir: Path, seed: int, min_objects: int, max_objects: int,
                  preview_count: int = 3) -> dict:
    if count <= 0 or min_objects <= 0 or max_objects < min_objects:
        raise ValueError("count and object bounds must be positive and ordered")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []
    client = scene.connect_simulation(use_gui=False)
    try:
        for attempt in range(count):
            p.resetSimulation()
            p.setAdditionalSearchPath(scene.pybullet_data.getDataPath())
            p.setGravity(0, 0, scene.GRAVITY)
            p.setTimeStep(scene.TIME_STEP)
            metadata = create_random_scene(rng, min_objects, max_objects)
            try:
                wait_for_objects_to_settle(metadata, LINEAR_VELOCITY_THRESHOLD,
                    ANGULAR_VELOCITY_THRESHOLD, SETTLE_CHECK_INTERVAL,
                    CONSECUTIVE_STABLE_CHECKS, MAX_SETTLE_STEPS)
            except Exception as error:
                print(f"skip scene {attempt}: {error}")
                continue
            image, segmentation = camera.capture_bgr_and_segmentation()
            labels = labels_for_scene(metadata, segmentation, image.shape[1], image.shape[0])
            if not labels:
                print(f"skip scene {attempt}: no visible target pixels")
                continue
            records.append((image, labels, metadata))
    finally:
        if p.isConnected(client):
            p.disconnect(client)
    total = len(records)
    order = list(range(total))
    random.Random(seed).shuffle(order)
    train_count = max(1, math.floor(total * 0.70))
    val_count = max(1, math.floor(total * 0.15)) if total >= 3 else 0
    if train_count + val_count >= total:
        val_count = max(0, total - train_count - 1)
    train_end = train_count
    val_end = train_end + val_count
    splits = {"train": order[:train_end], "val": order[train_end:val_end], "test": order[val_end:]}
    instance_counts = {name: 0 for name in CLASS_IDS}
    image_counts = {}
    for split, indices in splits.items():
        image_counts[split] = len(indices)
        for output_index, record_index in enumerate(indices):
            image, labels, metadata = records[record_index]
            stem = f"frame_{output_index:05d}"
            cv2.imwrite(str(output_dir / "images" / split / f"{stem}.png"), image)
            write_label(output_dir / "labels" / split / f"{stem}.txt", labels)
            if output_index < preview_count:
                preview(image, labels, preview_dir / f"{split}_{stem}.png")
            for class_id, _ in labels:
                instance_counts[{v: k for k, v in CLASS_IDS.items()}[class_id]] += 1
    yaml = output_dir / "dataset.yaml"
    relative_root = output_dir.as_posix()
    yaml.write_text(f"path: {relative_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cube\n  1: cylinder\n  2: box\n", encoding="utf-8")
    summary = {"images": total, "splits": image_counts, "instances": instance_counts, "seed": seed}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("data/yolo_dataset"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--preview-count", type=int, default=3)
    args = parser.parse_args()
    build_dataset(args.count, args.output_dir, args.seed, args.min_objects, args.max_objects, args.preview_count)


if __name__ == "__main__":
    main()
