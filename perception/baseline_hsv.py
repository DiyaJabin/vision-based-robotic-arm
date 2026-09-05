"""
Classical OpenCV baseline for detecting colored tabletop objects.

Classes:
    0 - cube     - red
    1 - cylinder - green
    2 - box      - blue

This baseline uses HSV color segmentation and contour analysis.
It does not perform YOLO detection, pixel-to-world conversion,
grasp planning, inverse kinematics, or robot control.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Object configuration
# ---------------------------------------------------------------------------

OBJECT_CONFIG = {
    "cube": {
        "class_id": 0,
        "color": "red",
        "lower": [(0, 100, 80), (170, 100, 80)],
        "upper": [(10, 255, 255), (180, 255, 255)],
    },
    "cylinder": {
        "class_id": 1,
        "color": "green",
        "lower": [(35, 70, 60)],
        "upper": [(85, 255, 255)],
    },
    "box": {
        "class_id": 2,
        "color": "blue",
        "lower": [(90, 70, 60)],
        "upper": [(135, 255, 255)],
    },
}


DEFAULT_MIN_AREA = 150
DEFAULT_MAX_AREA = 5000


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_mask(
    hsv_image: np.ndarray,
    lower_ranges: List[Tuple[int, int, int]],
    upper_ranges: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Create a cleaned binary mask for one HSV color range."""

    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for lower, upper in zip(lower_ranges, upper_ranges):
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)

        mask |= cv2.inRange(hsv_image, lower_np, upper_np)

    kernel = np.ones((5, 5), np.uint8)

    # Remove small noise.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # Fill small gaps.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return mask


def calculate_orientation(contour: np.ndarray) -> float:
    """
    Estimate the dominant contour orientation in degrees.

    The angle corresponds approximately to the direction of the
    longer side of the minimum-area bounding rectangle.
    """

    rectangle = cv2.minAreaRect(contour)

    (_, _), (width, height), angle = rectangle

    if width < height:
        angle += 90.0

    return float(angle)


def detect_objects(
    frame: np.ndarray,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float = DEFAULT_MAX_AREA,
) -> List[Dict]:
    """
    Detect colored objects in a BGR image.

    Returns a list of dictionaries containing:
        class_id
        class_name
        color
        confidence
        bbox
        center
        area
        orientation
    """

    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty.")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detections = []

    for class_name, config in OBJECT_CONFIG.items():

        lower_ranges = config["lower"]
        upper_ranges = config["upper"]

        mask = create_mask(
            hsv,
            lower_ranges,
            upper_ranges,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            moments = cv2.moments(contour)

            if moments["m00"] == 0:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            orientation = calculate_orientation(contour)

            # This is a simple classical-baseline confidence proxy,
            # not a learned probability.
            image_area = frame.shape[0] * frame.shape[1]
            confidence = min(1.0, area / max(image_area * 0.02, 1.0))

            detections.append(
                {
                    "class_id": config["class_id"],
                    "class_name": class_name,
                    "color": config["color"],
                    "baseline_score": round(float(confidence), 3),
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(width),
                        "height": int(height),
                    },
                    "center": {
                        "x": center_x,
                        "y": center_y,
                    },
                    "area": round(float(area), 2),
                    "orientation": round(float(orientation), 2),
                }
            )

    # Consistent ordering for easier debugging/testing.
    detections.sort(
        key=lambda detection: (
            detection["class_id"],
            detection["center"]["y"],
            detection["center"]["x"],
        )
    )

    return detections


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    detections: List[Dict],
) -> np.ndarray:
    """Draw bounding boxes, centers, labels and orientation on a frame."""

    output = frame.copy()

    for detection in detections:

        bbox = detection["bbox"]
        center = detection["center"]

        x = bbox["x"]
        y = bbox["y"]
        width = bbox["width"]
        height = bbox["height"]

        center_x = center["x"]
        center_y = center["y"]

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (255, 255, 255),
            2,
        )

        cv2.circle(
            output,
            (center_x, center_y),
            5,
            (255, 255, 255),
            -1,
        )

        label = (
            f'{detection["class_name"]} '
            f'({detection["baseline_score"]:.2f})'
        )

        cv2.putText(
            output,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw an approximate orientation axis.
        angle_rad = np.deg2rad(detection["orientation"])

        line_length = max(width, height) / 2.0

        end_x = int(
            center_x + line_length * np.cos(angle_rad)
        )

        end_y = int(
            center_y + line_length * np.sin(angle_rad)
        )

        cv2.line(
            output,
            (center_x, center_y),
            (end_x, end_y),
            (255, 255, 255),
            2,
        )

    return output


# ---------------------------------------------------------------------------
# Simple standalone image test
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the detector on an existing camera image."""

    image_path = "data/generated/images/frame_00000.png"

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Could not read image: {image_path}")
        print("Generate a dataset first.")
        return

    detections = detect_objects(frame)

    print(f"Detected objects: {len(detections)}")

    for index, detection in enumerate(detections, start=1):
        print(f"\nObject {index}")
        print(f"  Class       : {detection['class_name']}")
        print(f"  Class ID    : {detection['class_id']}")
        print(f"  Center      : {detection['center']}")
        print(f"  Bounding box: {detection['bbox']}")
        print(f"  Orientation : {detection['orientation']} degrees")
        print(f"  Area        : {detection['area']}")

    annotated = draw_detections(frame, detections)

    cv2.imshow("OpenCV HSV Baseline", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()