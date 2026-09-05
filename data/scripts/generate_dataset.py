"""Generate synthetic tabletop images from the PyBullet simulation.

The generator creates randomized scenes containing cube, cylinder, and box
objects, captures RGB images using the virtual camera, and stores JSON
metadata containing the ground-truth simulation state.

YOLO bounding-box labels are intentionally not generated here. Reliable
image-space projection should be implemented and validated separately
before producing training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pybullet as p

# Add the repository root to Python's import path when this file
# is executed directly from data/scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation import camera
from simulation import scene


# ============================================================
# Dataset configuration
# ============================================================

DEFAULT_IMAGE_COUNT = 20
DEFAULT_SEED = 42

DEFAULT_OUTPUT_DIR = Path("data/generated")

OBJECT_TYPES = ("cube", "cylinder", "box")

# Randomized tabletop workspace.
RANDOM_X_RANGE = (0.30, 0.72)
RANDOM_Y_RANGE = (-0.22, 0.22)

# Minimum horizontal distance between object centres.
MIN_OBJECT_DISTANCE = 0.10

# Number of attempts allowed when searching for a non-overlapping position.
MAX_POSITION_ATTEMPTS = 100

# Number of physics steps before image capture.
SETTLE_STEPS = 30


def object_surface_height(object_type: str) -> float:
    """Return the tabletop-relative centre height for an object.

    Args:
        object_type: Object class name.

    Returns:
        Object centre height above the tabletop.

    Raises:
        ValueError: If the object type is unsupported.
    """
    if object_type == "cube":
        return scene.OBJECT_SIZE / 2.0

    if object_type == "cylinder":
        return scene.CYLINDER_HEIGHT / 2.0

    if object_type == "box":
        return scene.BOX_HALF_EXTENTS[2]

    raise ValueError(f"Unsupported object type: {object_type}")


def choose_object_types(
    rng: random.Random,
    min_objects: int,
    max_objects: int,
) -> List[str]:
    """Randomly choose object classes for one scene.

    Args:
        rng: Random number generator.
        min_objects: Minimum number of objects.
        max_objects: Maximum number of objects.

    Returns:
        List of selected object class names.
    """
    count = rng.randint(min_objects, max_objects)

    # Ensure all three classes appear whenever three or more objects
    # are requested.
    if count >= len(OBJECT_TYPES):
        selected = list(OBJECT_TYPES)

        if count > len(OBJECT_TYPES):
            selected.extend(
                rng.choice(OBJECT_TYPES)
                for _ in range(count - len(OBJECT_TYPES))
            )

        rng.shuffle(selected)
        return selected

    return rng.sample(list(OBJECT_TYPES), count)


def generate_object_positions(
    rng: random.Random,
    object_types: List[str],
) -> List[Tuple[float, float]]:
    """Generate non-overlapping random XY positions.

    Args:
        rng: Random number generator.
        object_types: Object classes that will occupy the scene.

    Returns:
        List of XY coordinates corresponding to object_types.

    Raises:
        RuntimeError: If a valid non-overlapping layout cannot be found.
    """
    positions: List[Tuple[float, float]] = []

    for _ in object_types:
        position_found = False

        for _ in range(MAX_POSITION_ATTEMPTS):
            x = rng.uniform(*RANDOM_X_RANGE)
            y = rng.uniform(*RANDOM_Y_RANGE)

            candidate = (x, y)

            if all(
                math.dist(candidate, existing) >= MIN_OBJECT_DISTANCE
                for existing in positions
            ):
                positions.append(candidate)
                position_found = True
                break

        if not position_found:
            raise RuntimeError(
                "Could not generate a non-overlapping object layout."
            )

    return positions


def create_random_scene(
    rng: random.Random,
    min_objects: int,
    max_objects: int,
) -> List[Dict[str, object]]:
    """Create one randomized PyBullet tabletop scene.

    Args:
        rng: Random number generator.
        min_objects: Minimum number of objects.
        max_objects: Maximum number of objects.

    Returns:
        Ground-truth metadata for all spawned target objects.
    """
    scene.load_environment()
    scene.load_robot()
    scene.create_destination_zones()

    object_types = choose_object_types(
        rng,
        min_objects,
        max_objects,
    )

    positions = generate_object_positions(
        rng,
        object_types,
    )

    metadata: List[Dict[str, object]] = []

    for object_index, (object_type, xy_position) in enumerate(
        zip(object_types, positions)
    ):
        yaw = rng.uniform(
            0.0,
            2.0 * math.pi,
        )

        z = (
            scene.TABLETOP_Z
            + object_surface_height(object_type)
        )

        body_id = scene.spawn_object(
            object_type=object_type,
            position=(xy_position[0], xy_position[1], z),
            yaw=yaw,
        )

        metadata.append(
            {
                "class_id": scene.OBJECT_CLASS_IDS[object_type],
                "class_name": object_type,
                "object_id": body_id,
                "world_position": {
                    "x": xy_position[0],
                    "y": xy_position[1],
                    "z": z,
                },
                "world_orientation": {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": yaw,
                },
                "object_index": object_index,
            }
        )

    return metadata


def step_simulation(num_steps: int = SETTLE_STEPS) -> None:
    """Advance the physics simulation for a fixed number of steps."""
    for _ in range(num_steps):
        p.stepSimulation()

def update_object_poses(objects):
    """Update metadata with the actual poses after physics settling."""

    for obj in objects:
        position, orientation = p.getBasePositionAndOrientation(
            obj["object_id"]
        )

        roll, pitch, yaw = p.getEulerFromQuaternion(
            orientation
        )

        obj["world_position"] = {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        }

        obj["world_orientation"] = {
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        }

def save_metadata(
    metadata: List[Dict[str, object]],
    image_filename: str,
    frame_index: int,
    output_dir: Path,
) -> Path:
    """Save ground-truth metadata for one generated frame.

    Args:
        metadata: Object metadata for the scene.
        image_filename: Name of the corresponding RGB image.
        frame_index: Numeric frame identifier.
        output_dir: Dataset output directory.

    Returns:
        Path to the generated JSON metadata file.
    """
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = metadata_dir / f"frame_{frame_index:05d}.json"

    document = {
        "image_filename": image_filename,
        "image_width": camera.DEFAULT_WIDTH,
        "image_height": camera.DEFAULT_HEIGHT,
        "objects": metadata,
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            document,
            metadata_file,
            indent=2,
        )

    return metadata_path


def generate_dataset(
    image_count: int,
    output_dir: Path,
    seed: int,
    min_objects: int,
    max_objects: int,
) -> None:
    """Generate a complete synthetic RGB dataset.

    Args:
        image_count: Number of images to generate.
        output_dir: Dataset output directory.
        seed: Random seed for reproducibility.
        min_objects: Minimum objects per scene.
        max_objects: Maximum objects per scene.
    """
    if image_count <= 0:
        raise ValueError("image_count must be greater than zero.")

    if min_objects < 1:
        raise ValueError("min_objects must be at least 1.")

    if max_objects < min_objects:
        raise ValueError(
            "max_objects must be greater than or equal to min_objects."
        )

    rng = random.Random(seed)

    images_dir = output_dir / "images"
    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client_id = scene.connect_simulation(
        use_gui=False,
    )

    try:
        for frame_index in range(image_count):
            print(
                f"Generating frame "
                f"{frame_index + 1}/{image_count}..."
            )

            # Each frame is generated in a fresh physics client state.
            p.resetSimulation()
            p.setAdditionalSearchPath(
                scene.pybullet_data.getDataPath()
                if hasattr(scene, "pybullet_data")
                else ""
            )
            p.setGravity(
                0.0,
                0.0,
                scene.GRAVITY,
            )
            p.setTimeStep(
                scene.TIME_STEP,
            )

            metadata = create_random_scene(
                rng,
                min_objects,
                max_objects,
            )

            step_simulation()

            # Update metadata with the actual object poses
            # after physics settling.
            for obj in metadata:
                position, orientation = p.getBasePositionAndOrientation(
                    obj["object_id"]
                )

                roll, pitch, yaw = p.getEulerFromQuaternion(
                    orientation
                )

                obj["world_position"] = {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                }

                obj["world_orientation"] = {
                    "roll": float(roll),
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                }

            image_filename = f"frame_{frame_index:05d}.png"
            image_path = images_dir / image_filename

            frame = camera.capture_bgr(
                width=camera.DEFAULT_WIDTH,
                height=camera.DEFAULT_HEIGHT,
            )

            camera.save_frame(
                frame,
                image_path,
            )

            metadata_path = save_metadata(
                metadata,
                image_filename,
                frame_index,
                output_dir,
            )

            print(
                f"  Image   : {image_path}"
            )
            print(
                f"  Metadata: {metadata_path}"
            )
            print(
                f"  Objects : {len(metadata)}"
            )

    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)

    print("\nDataset generation complete.")
    print(f"Images directory  : {images_dir.resolve()}")
    print(
        f"Metadata directory: "
        f"{(output_dir / 'metadata').resolve()}"
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic PyBullet tabletop images "
            "and ground-truth JSON metadata."
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_IMAGE_COUNT,
        help=(
            f"Number of images to generate "
            f"(default: {DEFAULT_IMAGE_COUNT})."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Dataset output directory "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )

    parser.add_argument(
        "--min-objects",
        type=int,
        default=3,
        help="Minimum number of objects per scene.",
    )

    parser.add_argument(
        "--max-objects",
        type=int,
        default=3,
        help="Maximum number of objects per scene.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the synthetic dataset generator."""
    args = parse_arguments()

    generate_dataset(
        image_count=args.count,
        output_dir=args.output_dir,
        seed=args.seed,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
    )


if __name__ == "__main__":
    main()