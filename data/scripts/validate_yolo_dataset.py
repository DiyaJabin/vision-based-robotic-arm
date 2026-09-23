"""Validate YOLO image/label pairs, classes, bounds, and split counts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

VALID_CLASSES = {0, 1, 2}


def validate_dataset(root: Path) -> dict:
    counts = {split: {"images": 0, "labels": 0, "instances": {"0": 0, "1": 0, "2": 0}}
              for split in ("train", "val", "test")}
    errors = []
    for split in counts:
        images = sorted((root / "images" / split).glob("*.png"))
        counts[split]["images"] = len(images)
        for image in images:
            label = root / "labels" / split / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"missing label: {label}"); continue
            counts[split]["labels"] += 1
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{label}:{line_no}: expected 5 fields"); continue
                try:
                    cls = int(fields[0]); values = [float(x) for x in fields[1:]]
                except ValueError:
                    errors.append(f"{label}:{line_no}: nonnumeric label"); continue
                if cls not in VALID_CLASSES or not all(0 <= x <= 1 for x in values) or values[2] <= 0 or values[3] <= 0:
                    errors.append(f"{label}:{line_no}: invalid class or bounds")
                else:
                    counts[split]["instances"][str(cls)] += 1
    result = {"splits": counts, "errors": errors, "valid": not errors}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path, nargs="?", default=Path("data/yolo_dataset"))
    result = validate_dataset(parser.parse_args().root)
    raise SystemExit(0 if result["valid"] else 1)
