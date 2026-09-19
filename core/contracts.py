"""Lightweight perception and manipulation contracts; no execution logic.

World positions use metres in the PyBullet world frame (Z upward), with
Euler angles in radians. The contracts below describe image geometry only:
pixels, top-left origin, u/x rightward and v/y downward. Image angles use
degrees, positive clockwise on the image; they are not world-frame yaw.
The existing OpenCV dictionary API is unchanged and is not implicitly adapted.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePoint:
    """Image location in pixels; fractional coordinates are permitted."""

    u: float
    v: float


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned pixel box: top-left (x, y), positive width and height.

    Geometric bounds are [x, x + width) and [y, y + height); values are not
    normalized YOLO label coordinates. Producers must provide valid geometry.
    """

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class DetectionResult:
    """Learned detection with project class ID/name and confidence in [0, 1].

    Producers map model classes to core.config's supported project classes.
    confidence is a detector score, never the OpenCV baseline_score. centre
    denotes the bounding-box centre in pixels, not a contour centroid.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    centre: ImagePoint


@dataclass(frozen=True)
class ValidatedDetection:
    """Validation outcome, without implementing validation.

    validation_score is a heuristic in [0, 1], distinct from detector
    confidence, or None when validation was not performed. reason explains
    the outcome, including rejection or bypass of validation.
    """

    detection: DetectionResult
    accepted: bool
    validation_score: float | None
    reason: str


@dataclass(frozen=True)
class ObjectPose2D:
    """Image-plane object centroid and oriented dimensions, all in pixels.

    width follows orientation_deg; height is perpendicular. Defined angles
    use [0, 180) degrees from image +x, clockwise-positive, for an undirected
    rectangle axis. Squares have equivalent orientations modulo 90 degrees.
    Use None when orientation is unobservable, such as a circular cylinder
    top view; dimensions then follow image x/y. Producers must normalize
    angles explicitly: this contract does not alter the baseline's output.
    """

    centre: ImagePoint
    width: float
    height: float
    orientation_deg: float | None


@dataclass(frozen=True)
class ObjectObservation:
    """Accepted image observation; classical scores never fill confidence.

    source is 'opencv' or 'yolo'. The baseline dictionary adapter retains its
    contour centroid; a YOLO observation initially uses the box centre.
    """
    class_id: int
    class_name: str
    bbox: BoundingBox
    centre: ImagePoint
    source: str
    detection_confidence: float | None = None
    baseline_score: float | None = None
    validation_score: float | None = None


@dataclass(frozen=True)
class CartesianPose:
    """World metres and a unit quaternion in PyBullet (x,y,z,w) order."""
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class PickOutcome:
    """Consumable execution result; no file logging or experiment metrics."""
    object_id: int | None
    class_name: str
    status: str
    reason: str
    destination: str | None = None


@dataclass(frozen=True)
class RunOutcome:
    status: str
    reason: str
    picks: tuple[PickOutcome, ...]
    observation_count: int
    remaining_object_ids: tuple[int, ...]
    rejections: tuple[str, ...] = ()
