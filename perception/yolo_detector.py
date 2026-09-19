"""Ultralytics adapter. No training and no implicit COCO-to-project mapping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import math

from core.config import CLASS_IDS
from core.contracts import BoundingBox, DetectionResult, ImagePoint


class ModelUnavailableError(RuntimeError):
    """Weights or the inference backend could not be loaded/executed."""


@dataclass(frozen=True)
class GenericDetection:
    """Native model class; deliberately separate from project DetectionResult."""
    model_class_id: int
    model_class_name: str
    confidence: float
    bbox: BoundingBox
    centre: ImagePoint


def check_bgr(image: np.ndarray) -> None:
    import numpy as np
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 NumPy BGR image.")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Expected a nonempty H x W x 3 BGR image.")


class YoloDetector:
    """Load YOLOv8n or a supplied custom model; injected models aid testing.

    detect() returns only exact supported names (or explicit name mappings).
    detect_generic() retains native model classes for observation only. COCO
    pretrained weights normally have no cube/cylinder/box classes. Passing a
    custom path requires an existing file. A named pretrained weight may be
    downloaded by Ultralytics when the caller explicitly constructs this class.
    """
    def __init__(self, weights: str = "yolov8n.pt", *, custom_weights: str | Path | None = None,
                 class_mapping: Mapping[str, str] | None = None, device: str = "cpu",
                 min_confidence: float = 0.01, model=None):
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1].")
        self.class_mapping = dict(class_mapping or {})
        if any(name not in CLASS_IDS for name in self.class_mapping.values()):
            raise ValueError("Class mappings must target supported project names.")
        self.device, self.min_confidence = device, min_confidence
        if custom_weights is not None:
            path = Path(custom_weights)
            if not path.is_file():
                raise ModelUnavailableError(f"Custom weights do not exist: {path}")
            weights = str(path)
        if model is None:
            try:
                from ultralytics import YOLO
                model = YOLO(weights, task="detect")
            except Exception as error:
                raise ModelUnavailableError(f"Cannot load YOLO weights {weights!r}: {error}") from error
        self.model = model

    def detect_generic(self, image: np.ndarray) -> list[GenericDetection]:
        check_bgr(image)
        import numpy as np
        try:
            results = self.model.predict(source=image, conf=self.min_confidence,
                                         device=self.device, verbose=False, save=False)
            output = []
            height, width = image.shape[:2]
            for result in results:
                if result.boxes is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for coords, score, class_id in zip(boxes, scores, classes):
                    if not np.isfinite(coords).all() or not math.isfinite(float(score)) or not 0 <= score <= 1:
                        continue
                    if not math.isfinite(float(class_id)) or float(class_id) != int(class_id):
                        continue
                    x1, y1, x2, y2 = map(float, coords)
                    x1, x2 = max(0., x1), min(float(width), x2)
                    y1, y2 = max(0., y1), min(float(height), y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    name = str(result.names[int(class_id)])
                    bbox = BoundingBox(x1, y1, x2-x1, y2-y1)
                    output.append(GenericDetection(int(class_id), name, float(score), bbox,
                                                   ImagePoint((x1+x2)/2, (y1+y2)/2)))
            return output
        except Exception as error:
            raise ModelUnavailableError(f"YOLO inference failed: {error}") from error

    def map_project_detections(self, detections) -> list[DetectionResult]:
        """Filter native classes and map names, never model numeric IDs."""
        output = []
        for item in detections:
            name = self.class_mapping.get(item.model_class_name, item.model_class_name)
            if name in CLASS_IDS:
                output.append(DetectionResult(CLASS_IDS[name], name, item.confidence,
                                              item.bbox, item.centre))
        return output

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        return self.map_project_detections(self.detect_generic(image))
