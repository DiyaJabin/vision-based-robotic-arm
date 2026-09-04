"""
Demo for the classical OpenCV vision baseline.

Pipeline:
    PyBullet scene
        ↓
    Virtual RGB camera
        ↓
    OpenCV HSV detection
        ↓
    Bounding boxes + centers + orientation
"""

import cv2
import pybullet as p

from perception.baseline_hsv import detect_objects, draw_detections
from simulation import camera
from simulation import scene


def main() -> None:
    """Run the PyBullet camera and OpenCV detection demo."""

    print("Starting PyBullet simulation...")

    physics_client = scene.connect_simulation(use_gui=True)

    try:
        scene.load_complete_scene()

        # Allow the objects to settle.
        for _ in range(60):
            p.stepSimulation()

        print("Capturing camera frame...")

        frame = camera.capture_rgb()

        print("Running OpenCV HSV detection...")

        detections = detect_objects(frame)

        print(f"\nDetected objects: {len(detections)}")

        for index, detection in enumerate(detections, start=1):
            print(f"\nObject {index}")
            print(f"  Class       : {detection['class_name']}")
            print(f"  Class ID    : {detection['class_id']}")
            print(f"  Center      : {detection['center']}")
            print(f"  Bounding box: {detection['bbox']}")
            print(f"  Orientation : {detection['orientation']} degrees")
            print(f"  Area        : {detection['area']}")

        annotated = draw_detections(frame, detections)

        cv2.imshow(
            "PyBullet + OpenCV HSV Baseline",
            annotated,
        )

        print("\nPress any key in the image window to close.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        p.disconnect(physics_client)
        print("Simulation closed.")


if __name__ == "__main__":
    main()