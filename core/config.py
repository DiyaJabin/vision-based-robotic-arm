"""Shared defaults for the initial three-class project.

World positions use metres; image coordinates use pixels. PyBullet Euler
angles use radians; image-plane contract angles use degrees (see contracts).
Scene geometry and module-specific tuning remain in their owning modules.
"""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping


CLASS_NAMES: Mapping[int, str] = MappingProxyType({
    0: "cube",
    1: "cylinder",
    2: "box",
})
CLASS_IDS: Mapping[str, int] = MappingProxyType({
    name: class_id for class_id, name in CLASS_NAMES.items()
})
CLASS_DESTINATIONS: Mapping[str, str] = MappingProxyType({
    "cube": "A",
    "cylinder": "B",
    "box": "C",
})
BIN_SCENE_KEYS: Mapping[str, str] = MappingProxyType({
    "A": "destination_1",
    "B": "destination_2",
    "C": "destination_3",
})

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Configurable experimental defaults, not validated scientific thresholds.

    Proposed bands: high >= high, medium <= score < high, low < medium.
    These apply to learned detector confidence, never OpenCV baseline_score.
    Instantiate with different values to configure a future consumer.
    """

    high: float = 0.80
    medium: float = 0.50

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.high)
            and math.isfinite(self.medium)
            and 0.0 <= self.medium < self.high <= 1.0
        ):
            raise ValueError("Thresholds must satisfy 0 <= medium < high <= 1.")
