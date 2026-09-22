"""Confidence gating with configurable, experimental ROI colour/shape checks."""
from __future__ import annotations
from dataclasses import dataclass
import math
from core.config import CLASS_IDS, ConfidenceThresholds
from core.contracts import BoundingBox, DetectionResult, ValidatedDetection
from perception.yolo_detector import check_bgr


@dataclass(frozen=True)
class ValidationConfig:
    """Initial heuristics, not tuned or scientifically validated values."""
    min_colour_fraction: float = 0.20
    min_contour_area: float = 25.0
    min_solidity: float = 0.80
    min_circularity: float = 0.65
    min_rectangularity: float = 0.65
    max_cube_aspect: float = 1.40
    max_box_aspect: float = 3.0

    def __post_init__(self):
        for value in (self.min_colour_fraction, self.min_solidity, self.min_circularity, self.min_rectangularity):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Validation fractions must be in (0, 1].")
        if not math.isfinite(self.min_contour_area) or self.min_contour_area <= 0:
            raise ValueError("Contour area must be positive.")
        if any(not math.isfinite(v) or v < 1 for v in (self.max_cube_aspect, self.max_box_aspect)):
            raise ValueError("Aspect limits must be finite and >= 1.")


@dataclass(frozen=True)
class ContourMetrics:
    colour_fraction: float
    area: float
    solidity: float
    rectangularity: float
    circularity: float
    aspect: float


def validate_metrics(detection: DetectionResult, metrics: ContourMetrics,
                     config: ValidationConfig | None = None) -> ValidatedDetection:
    """Pure medium-band decision from measured ROI metrics, no model score reuse."""
    cfg=config or ValidationConfig()
    if (CLASS_IDS.get(detection.class_name)!=detection.class_id
            or any(not math.isfinite(v) or v<0 for v in vars(metrics).values())):
        return ValidatedDetection(detection,False,0.,"invalid_contour_metrics")
    colour,area,solidity=metrics.colour_fraction,metrics.area,metrics.solidity
    shape=metrics.circularity if detection.class_name=="cylinder" else metrics.rectangularity
    checks=[(colour>=cfg.min_colour_fraction,"colour_fraction"),
            (area>=cfg.min_contour_area,"contour_area"),
            (solidity>=cfg.min_solidity,"solidity")]
    if detection.class_name=="cylinder":
        checks.append((shape>=cfg.min_circularity,"circularity"))
    else:
        limit=cfg.max_cube_aspect if detection.class_name=="cube" else cfg.max_box_aspect
        checks += [(shape>=cfg.min_rectangularity,"rectangularity"),(metrics.aspect<=limit,"aspect_ratio")]
    score=max(0.,min(1.,(colour+solidity+shape)/3))
    failures=[reason for passed,reason in checks if not passed]
    return ValidatedDetection(detection,not failures,score,
                              "opencv_pass" if not failures else "opencv_fail:"+",".join(failures))


def roi_contour(frame: np.ndarray, bbox: BoundingBox, class_name: str):
    """Largest cleaned class-colour contour in a clipped ROI, in full pixels."""
    import cv2
    import numpy as np
    from perception.baseline_hsv import OBJECT_CONFIG, create_mask
    check_bgr(frame)
    if class_name not in CLASS_IDS:
        raise ValueError("Unsupported project class.")
    values = (bbox.x, bbox.y, bbox.width, bbox.height)
    if not all(math.isfinite(v) for v in values) or bbox.width <= 0 or bbox.height <= 0:
        raise ValueError("Invalid bounding box.")
    height, width = frame.shape[:2]
    x1, y1 = max(0, math.floor(bbox.x)), max(0, math.floor(bbox.y))
    x2, y2 = min(width, math.ceil(bbox.x+bbox.width)), min(height, math.ceil(bbox.y+bbox.height))
    if x1 >= x2 or y1 >= y2:
        return None, 0.0
    config = OBJECT_CONFIG[class_name]
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    mask = create_mask(hsv, config["lower"], config["upper"])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea) + np.array([[[x1, y1]]], dtype=np.int32)
    return contour, float(np.count_nonzero(mask)/mask.size)


def confidence_band(confidence: float, thresholds: ConfidenceThresholds | None = None) -> str:
    """Pure confidence gate; no image processing or baseline score substitution."""
    limits=thresholds or ConfidenceThresholds()
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Detector confidence must be in [0, 1].")
    return "high" if confidence >= limits.high else "medium" if confidence >= limits.medium else "low"


class HybridValidator:
    def __init__(self, thresholds: ConfidenceThresholds | None = None,
                 config: ValidationConfig | None = None):
        self.thresholds = thresholds or ConfidenceThresholds()
        self.config = config or ValidationConfig()

    def validate(self, frame: np.ndarray, detection: DetectionResult) -> ValidatedDetection:
        check_bgr(frame)
        reject = lambda reason: ValidatedDetection(detection, False, None, reason)
        if CLASS_IDS.get(detection.class_name) != detection.class_id:
            return reject("unsupported_class")
        score = detection.confidence
        if not math.isfinite(score) or not 0 <= score <= 1:
            return reject("invalid_detector_confidence")
        bbox = detection.bbox
        if (not all(math.isfinite(v) for v in (bbox.x, bbox.y, bbox.width, bbox.height))
                or bbox.width <= 0 or bbox.height <= 0 or bbox.x+bbox.width <= 0
                or bbox.y+bbox.height <= 0 or bbox.x >= frame.shape[1] or bbox.y >= frame.shape[0]):
            return reject("invalid_bbox")
        band = confidence_band(score, self.thresholds)
        if band == "high":
            return ValidatedDetection(detection, True, None, "high_confidence")
        if band == "low":
            return reject("low_confidence")
        import cv2
        contour, colour = roi_contour(frame, bbox, detection.class_name)
        if contour is None:
            return ValidatedDetection(detection, False, 0.0, "no_colour_contour")
        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area/hull_area if hull_area else 0.0
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        rectangularity = area/(w*h) if w*h else 0.0
        perimeter = cv2.arcLength(contour, True)
        circularity = min(1.0, 4*math.pi*area/(perimeter*perimeter)) if perimeter else 0.0
        aspect = max(w,h)/min(w,h) if min(w,h) else math.inf
        return validate_metrics(detection,ContourMetrics(colour,area,solidity,rectangularity,circularity,aspect),self.config)
