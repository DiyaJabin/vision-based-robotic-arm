"""Capture a camera frame and return it in OpenCV BGR format.

This module captures color and depth images from a fixed overhead camera
inside the PyBullet tabletop environment. Color frames are returned as BGR
arrays for OpenCV.

It reuses the scene setup from simulation.scene and does not create a
separate simulation environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pybullet as p

from core.config import CAMERA_HEIGHT, CAMERA_WIDTH
from simulation import scene


# ============================================================
# Camera configuration
# ============================================================

DEFAULT_WIDTH = CAMERA_WIDTH
DEFAULT_HEIGHT = CAMERA_HEIGHT

CAMERA_EYE = (0.50, 0.00, 2.20)
CAMERA_TARGET = (0.50, 0.00, scene.TABLETOP_Z)
CAMERA_UP = (0.00, 1.00, 0.00)

FIELD_OF_VIEW = 55.0
NEAR_PLANE = 0.01
FAR_PLANE = 5.0


def camera_matrices(
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


def capture_bgr(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> np.ndarray:
    """Capture a virtual-camera color image and return a BGR array.

    The returned image is converted to BGR format so it can be
    directly processed by OpenCV.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        NumPy array with shape (height, width, 3) in BGR format.
    """
    view_matrix, projection_matrix = camera_matrices(width, height)

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
    view_matrix, projection_matrix = camera_matrices(
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
    """Capture a BGR frame and save it with OpenCV.

    Args:
        output_path: Destination image path.
        width: Image width.
        height: Image height.

    Returns:
        Absolute path of the saved image.
    """
    frame = capture_bgr(
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
        frame = capture_bgr()

        output_path = capture_and_save(
            "data/generated/camera_preview.png"
        )

        depth = capture_depth()

        print(f"BGR frame shape: {frame.shape}")
        print(f"Depth frame shape: {depth.shape}")
        print(f"Camera image saved to: {output_path}")

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


def project_world_points(points, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
    """Project known world reference points using the configured virtual camera.

    This supports calibration correspondences, not synthetic YOLO annotations.
    Output is continuous (u,v) pixels with top-left origin. PyBullet/OpenGL
    matrices are column-major; image v reverses the NDC vertical direction.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    world=np.asarray(points,dtype=float)
    if world.ndim!=2 or world.shape[1]!=3 or not np.isfinite(world).all():
        raise ValueError("Expected finite N x 3 world points.")
    view,projection=camera_matrices(width,height)
    transform=np.asarray(projection).reshape(4,4,order="F") @ np.asarray(view).reshape(4,4,order="F")
    clip=(transform @ np.column_stack((world,np.ones(len(world)))).T).T
    if np.any(clip[:,3]<=0):
        raise ValueError("Reference point is behind the camera.")
    ndc=clip[:,:3]/clip[:,3,None]
    return np.column_stack(((ndc[:,0]+1)*width/2,(1-ndc[:,1])*height/2))


# Compatibility for existing users of the original private helper.
_camera_matrices = camera_matrices
