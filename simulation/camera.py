"""Virtual overhead RGB-D camera for the PyBullet simulation.

This module captures RGB and depth images from a fixed overhead camera
inside the PyBullet tabletop environment.

It reuses the scene setup from simulation.scene and does not create a
separate simulation environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pybullet as p

from simulation import scene


# ============================================================
# Camera configuration
# ============================================================

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 640

CAMERA_EYE = (0.50, 0.00, 2.20)
CAMERA_TARGET = (0.50, 0.00, 0.65)
CAMERA_UP = (0.00, 1.00, 0.00)

FIELD_OF_VIEW = 55.0
NEAR_PLANE = 0.01
FAR_PLANE = 5.0


def _camera_matrices(
    width: int,
    height: int,
) -> Tuple[list[float], list[float]]:
    """Create the PyBullet view and projection matrices.

    Args:
        width: Camera image width in pixels.
        height: Camera image height in pixels.

    Returns:
        Tuple containing the view matrix and projection matrix.
    """
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=CAMERA_EYE,
        cameraTargetPosition=CAMERA_TARGET,
        cameraUpVector=CAMERA_UP,
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=FIELD_OF_VIEW,
        aspect=float(width) / float(height),
        nearVal=NEAR_PLANE,
        farVal=FAR_PLANE,
    )

    return view_matrix, projection_matrix


def capture_rgb(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> np.ndarray:
    """Capture an RGB image from the virtual camera.

    The returned image is converted to BGR format so it can be
    directly processed by OpenCV.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        NumPy array with shape (height, width, 3) in BGR format.
    """
    view_matrix, projection_matrix = _camera_matrices(width, height)

    _, _, rgba_image, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    rgba_array = np.asarray(
        rgba_image,
        dtype=np.uint8,
    ).reshape((height, width, 4))

    rgb_array = cv2.cvtColor(
        rgba_array,
        cv2.COLOR_RGBA2RGB,
    )

    bgr_array = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2BGR,
    )

    return bgr_array


def capture_depth(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> np.ndarray:
    """Capture a depth image and convert it to metres.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Floating-point NumPy array containing depth values in metres.
    """
    view_matrix, projection_matrix = _camera_matrices(
        width,
        height,
    )

    _, _, _, depth_buffer, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    depth_buffer = np.asarray(
        depth_buffer,
        dtype=np.float32,
    ).reshape((height, width))

    # Convert PyBullet's normalized depth buffer to metric depth.
    depth_image = (
        NEAR_PLANE
        * FAR_PLANE
        / (
            FAR_PLANE
            - (FAR_PLANE - NEAR_PLANE) * depth_buffer
        )
    )

    return depth_image


def save_frame(
    frame: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a BGR camera frame as an image.

    Args:
        frame: OpenCV-compatible BGR image.
        output_path: Destination file path.

    Returns:
        Absolute path of the saved image.

    Raises:
        ValueError: If OpenCV cannot write the image.
    """
    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(path),
        frame,
    )

    if not success:
        raise ValueError(
            f"Could not save camera frame: {path}"
        )

    return path.resolve()


def show_frame(
    frame: np.ndarray,
    window_name: str = "PyBullet Virtual Camera",
    wait_ms: int = 30,
) -> bool:
    """Display a camera frame using OpenCV.

    Press ``q`` or ``Esc`` to close the display.

    Args:
        frame: OpenCV-compatible BGR image.
        window_name: Name of the OpenCV window.
        wait_ms: Keyboard wait time in milliseconds.

    Returns:
        True if the user requested exit; otherwise False.
    """
    cv2.imshow(
        window_name,
        frame,
    )

    key = cv2.waitKey(wait_ms) & 0xFF

    return key in (ord("q"), 27)


def capture_and_save(
    output_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Path:
    """Capture an RGB frame and save it to disk.

    Args:
        output_path: Destination image path.
        width: Image width.
        height: Image height.

    Returns:
        Absolute path of the saved image.
    """
    frame = capture_rgb(
        width=width,
        height=height,
    )

    return save_frame(
        frame,
        output_path,
    )


def main() -> None:
    """Run a simple virtual-camera demonstration."""
    client_id = scene.connect_simulation(
        use_gui=True,
    )

    scene.load_complete_scene()

    try:
        frame = capture_rgb()

        output_path = capture_and_save(
            "data/generated/camera_preview.png"
        )

        depth = capture_depth()

        print(f"RGB frame shape: {frame.shape}")
        print(f"Depth frame shape: {depth.shape}")
        print(f"RGB image saved to: {output_path}")

        while p.isConnected(client_id):
            if show_frame(frame):
                break

            p.stepSimulation()

    except KeyboardInterrupt:
        print("\nCamera demonstration interrupted.")

    finally:
        cv2.destroyAllWindows()

        if p.isConnected(client_id):
            p.disconnect(client_id)
            print("PyBullet simulation disconnected cleanly.")


if __name__ == "__main__":
    main()
