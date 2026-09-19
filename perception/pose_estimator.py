"""Contour-based image pose, preserving object symmetry and explicit units."""
from __future__ import annotations
import math
from core.config import CLASS_IDS
from core.contracts import BoundingBox, ImagePoint, ObjectPose2D
from perception.hybrid_validator import roi_contour


class PoseEstimationError(ValueError):
    pass


def normalise_orientation(width: float, height: float, angle: float, class_name: str):
    """Return axis-aligned cylinder dimensions unchanged and no cylinder yaw.

    For rectangles width follows the longer axis; cubes use equivalent axes
    modulo 90 degrees, swapping dimensions when selecting that equivalent axis.
    """
    if class_name not in CLASS_IDS or not all(math.isfinite(v) for v in (width,height,angle)) or min(width,height)<=0:
        raise PoseEstimationError("Invalid geometry or unsupported class.")
    if class_name == "cylinder":
        return width,height,None
    if width < height:
        width,height,angle=height,width,angle+90
    angle %= 180
    if class_name == "cube" and angle >= 90:
        width,height,angle=height,width,angle-90
    return width,height,angle


def pose_from_contour(contour: np.ndarray, class_name: str) -> ObjectPose2D:
    import cv2
    import numpy as np
    if class_name not in CLASS_IDS:
        raise PoseEstimationError("Unsupported project class.")
    points = np.asarray(contour, dtype=np.float32)
    if points.size < 6 or not np.isfinite(points).all():
        raise PoseEstimationError("Contour needs finite nondegenerate points.")
    points = points.reshape(-1, 1, 2)
    moments = cv2.moments(points)
    if moments["m00"] <= 0:
        raise PoseEstimationError("Contour has zero area.")
    centre = ImagePoint(moments["m10"]/moments["m00"], moments["m01"]/moments["m00"])
    (_, _), (width, height), angle = cv2.minAreaRect(points)
    if class_name == "cylinder":
        extents = np.ptp(points[:, 0, :], axis=0)
        return ObjectPose2D(centre, float(extents[0]), float(extents[1]), None)
    width,height,angle = normalise_orientation(width,height,angle,class_name)
    return ObjectPose2D(centre, float(width), float(height), float(angle))


def estimate_pose(frame: np.ndarray, bbox: BoundingBox, class_name: str) -> ObjectPose2D:
    contour, _ = roi_contour(frame, bbox, class_name)
    if contour is None:
        raise PoseEstimationError("No class-colour contour available for image pose.")
    return pose_from_contour(contour, class_name)
