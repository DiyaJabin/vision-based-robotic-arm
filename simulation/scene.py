"""PyBullet tabletop simulation foundation.

Provides reusable functions for creating the simulation environment,
loading the robotic arm, spawning coloured tabletop objects, and
creating destination zones.

This module intentionally does not implement robot motion, inverse
kinematics, grasping, or pick-and-place control.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Tuple

import pybullet as p
import pybullet_data


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

GRAVITY = -9.81
TIME_STEP = 1.0 / 240.0

TABLE_POSITION = (0.5, 0.0, 0.0)
ROBOT_POSITION = (0.0, 0.0, 0.62)

TABLETOP_Z = 0.65

OBJECT_SIZE = 0.05
CYLINDER_RADIUS = 0.025
CYLINDER_HEIGHT = 0.08
BOX_HALF_EXTENTS = (0.04, 0.03, 0.025)

# Object class IDs used by the initial synthetic dataset.
OBJECT_CLASS_IDS = {
    "cube": 0,
    "cylinder": 1,
    "box": 2,
}

OBJECT_COLORS = {
    "cube": (0.90, 0.15, 0.15, 1.0),       # red
    "cylinder": (0.15, 0.80, 0.20, 1.0),   # green
    "box": (0.15, 0.35, 0.90, 1.0),        # blue
}

DESTINATION_COLORS = (
    (0.90, 0.70, 0.10, 0.75),
    (0.70, 0.20, 0.80, 0.75),
    (0.10, 0.75, 0.75, 0.75),
)

# Workspace limits used by the scene and later dataset randomization.
WORKSPACE_X = (0.25, 0.75)
WORKSPACE_Y = (-0.30, 0.30)


def connect_simulation(use_gui: bool = True) -> int:
    """Connect to PyBullet and configure basic simulation parameters.

    Args:
        use_gui: If True, open the PyBullet GUI. If False, use DIRECT mode.

    Returns:
        The PyBullet physics client ID.

    Raises:
        RuntimeError: If PyBullet cannot establish a connection.
    """
    connection_mode = p.GUI if use_gui else p.DIRECT
    physics_client = p.connect(connection_mode)

    if physics_client < 0 and use_gui:
        print(
            "Warning: PyBullet GUI could not be opened. "
            "Falling back to DIRECT mode."
        )
        physics_client = p.connect(p.DIRECT)

    if physics_client < 0:
        raise RuntimeError("Could not connect to the PyBullet physics server.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, GRAVITY)
    p.setTimeStep(TIME_STEP)

    if use_gui and p.getConnectionInfo(physics_client)["connectionMethod"] == p.GUI:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.55,
            cameraYaw=45.0,
            cameraPitch=-45.0,
            cameraTargetPosition=[0.5, 0.0, TABLETOP_Z],
        )

    return physics_client


def load_environment() -> Dict[str, int]:
    """Load the ground plane and tabletop.

    Returns:
        Dictionary containing PyBullet body IDs for the environment.
    """
    bodies: Dict[str, int] = {}

    bodies["plane"] = p.loadURDF(
        "plane.urdf",
        basePosition=[0.0, 0.0, 0.0],
    )

    table_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    bodies["table"] = p.loadURDF(
        "table/table.urdf",
        basePosition=list(TABLE_POSITION),
        baseOrientation=table_orientation,
    )

    return bodies


def load_robot() -> int:
    """Load the fixed-base KUKA iiwa robotic arm.

    Returns:
        PyBullet body ID of the robot.
    """
    robot_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    robot_id = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=list(ROBOT_POSITION),
        baseOrientation=robot_orientation,
        useFixedBase=True,
    )

    return robot_id


def spawn_object(
    object_type: str,
    position: Tuple[float, float, float],
    yaw: float = 0.0,
    mass: float = 0.1,
) -> int:
    """Create a coloured tabletop object.

    Args:
        object_type: One of ``cube``, ``cylinder``, or ``box``.
        position: Object centre position in world coordinates.
        yaw: Rotation around the vertical axis in radians.
        mass: Object mass in kilograms.

    Returns:
        PyBullet body ID.

    Raises:
        ValueError: If an unsupported object type is requested.
    """
    if object_type not in OBJECT_CLASS_IDS:
        supported = ", ".join(OBJECT_CLASS_IDS)
        raise ValueError(
            f"Unsupported object type '{object_type}'. "
            f"Supported types: {supported}"
        )

    color = OBJECT_COLORS[object_type]
    orientation = p.getQuaternionFromEuler([0.0, 0.0, yaw])

    if object_type == "cube":
        half_extent = OBJECT_SIZE / 2.0

        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_extent] * 3,
            rgbaColor=color,
        )

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half_extent] * 3,
        )

    elif object_type == "cylinder":
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=CYLINDER_RADIUS,
            length=CYLINDER_HEIGHT,
            rgbaColor=color,
        )

        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=CYLINDER_RADIUS,
            height=CYLINDER_HEIGHT,
        )

    else:
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(BOX_HALF_EXTENTS),
            rgbaColor=color,
        )

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=list(BOX_HALF_EXTENTS),
        )

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=list(position),
        baseOrientation=orientation,
    )

    return body_id


def create_default_objects() -> Dict[str, int]:
    """Spawn the three initial coloured target objects.

    Returns:
        Dictionary mapping object names to PyBullet body IDs.
    """
    bodies: Dict[str, int] = {}

    cube_height = OBJECT_SIZE / 2.0
    cylinder_height = CYLINDER_HEIGHT / 2.0
    box_height = BOX_HALF_EXTENTS[2]

    bodies["cube"] = spawn_object(
        "cube",
        (0.50, 0.15, TABLETOP_Z + cube_height),
        yaw=0.0,
    )

    bodies["cylinder"] = spawn_object(
        "cylinder",
        (0.58, -0.15, TABLETOP_Z + cylinder_height),
        yaw=0.0,
    )

    bodies["box"] = spawn_object(
        "box",
        (0.68, 0.05, TABLETOP_Z + box_height),
        yaw=0.35,
    )

    return bodies


def create_destination_zones() -> Dict[str, int]:
    """Create three flat destination zones on the tabletop.

    Returns:
        Dictionary mapping destination names to PyBullet body IDs.
    """
    zones: Dict[str, int] = {}

    zone_positions = (
        (0.35, 0.30),
        (0.50, 0.30),
        (0.65, 0.30),
    )

    for index, (x, y) in enumerate(zone_positions, start=1):
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.055, 0.045, 0.003],
            rgbaColor=DESTINATION_COLORS[index - 1],
        )

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.055, 0.045, 0.003],
        )

        zones[f"destination_{index}"] = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, TABLETOP_Z + 0.003],
        )

    return zones


def print_joint_info(robot_id: int) -> None:
    """Print robot joint indices, names, types, and link names.

    Args:
        robot_id: PyBullet body ID of the robotic arm.
    """
    joint_type_names = {
        p.JOINT_REVOLUTE: "Revolute",
        p.JOINT_PRISMATIC: "Prismatic",
        p.JOINT_SPHERICAL: "Spherical",
        p.JOINT_PLANAR: "Planar",
        p.JOINT_FIXED: "Fixed",
    }

    num_joints = p.getNumJoints(robot_id)

    print("\n" + "=" * 75)
    print(
        f"Robotic Arm Joint Information "
        f"(Body ID: {robot_id}, Joints: {num_joints})"
    )
    print("=" * 75)
    print(
        f"{'Index':<8}"
        f"{'Joint Name':<30}"
        f"{'Joint Type':<15}"
        f"{'Link Name':<20}"
    )
    print("-" * 75)

    for index in range(num_joints):
        info = p.getJointInfo(robot_id, index)

        joint_name = info[1].decode("utf-8")
        joint_type = joint_type_names.get(
            info[2],
            f"Unknown ({info[2]})",
        )
        link_name = info[12].decode("utf-8")

        print(
            f"{index:<8}"
            f"{joint_name:<30}"
            f"{joint_type:<15}"
            f"{link_name:<20}"
        )

    print("=" * 75 + "\n")


def load_complete_scene() -> Dict[str, object]:
    """Load the complete reusable simulation scene.

    Returns:
        Dictionary containing environment, robot, objects, and destinations.
    """
    environment = load_environment()
    robot_id = load_robot()
    objects = create_default_objects()
    destinations = create_destination_zones()

    print_joint_info(robot_id)

    return {
        "environment": environment,
        "robot": robot_id,
        "objects": objects,
        "destinations": destinations,
    }


def run_simulation(physics_client: int) -> None:
    """Run the PyBullet simulation until the user exits.

    Args:
        physics_client: Active PyBullet physics client ID.
    """
    print(
        "Simulation active. Close the PyBullet window "
        "or press Ctrl+C to exit."
    )

    try:
        while p.isConnected(physics_client):
            p.stepSimulation()
            time.sleep(TIME_STEP)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    finally:
        if p.isConnected(physics_client):
            p.disconnect(physics_client)
            print("PyBullet simulation disconnected cleanly.")


def main() -> None:
    """Launch the interactive tabletop simulation."""
    physics_client: Optional[int] = None

    try:
        physics_client = connect_simulation(use_gui=True)
        load_complete_scene()
        run_simulation(physics_client)

    except Exception as error:
        print(
            f"Error executing PyBullet simulation: {error}",
            file=sys.stderr,
        )

        if physics_client is not None and p.isConnected(physics_client):
            p.disconnect(physics_client)

        sys.exit(1)


if __name__ == "__main__":
    main()