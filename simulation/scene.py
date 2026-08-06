"""PyBullet simulation feasibility scene for tabletop robotic arm setup.

This module initializes a PyBullet GUI simulation environment, loads a plane, table,
robotic arm model, sample tabletop target objects, and a destination tray. It prints joint
information to stdout and runs the physics step loop.
"""

import sys
import time
from typing import Dict, List, Optional, Tuple

import pybullet as p
import pybullet_data


def initialize_simulation(use_gui: bool = True) -> int:
    """Connect to PyBullet server and set up environment parameters.

    Args:
        use_gui (bool): Whether to attempt connecting in GUI mode (default: True).

    Returns:
        int: PyBullet client physics ID.

    Raises:
        RuntimeError: If connection to PyBullet server fails.
    """
    connection_mode = p.GUI if use_gui else p.DIRECT
    physics_client = p.connect(connection_mode)

    if physics_client < 0:
        # Fallback attempt if GUI connection fails
        print("Warning: Failed to open PyBullet GUI window. Falling back to DIRECT mode.")
        physics_client = p.connect(p.DIRECT)

    if physics_client < 0:
        raise RuntimeError("Could not connect to PyBullet physics server.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # Configure camera view for optimal tabletop perspective
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0.0, 0.6],
    )

    return physics_client


def build_tabletop_scene() -> Dict[str, int]:
    """Load plane, table, robotic arm, target objects, and destination tray into the scene.

    Returns:
        Dict[str, int]: Mapping of body names to PyBullet body IDs.
    """
    bodies: Dict[str, int] = {}

    # Load ground plane
    bodies["plane"] = p.loadURDF("plane.urdf", [0, 0, 0])

    # Load workspace table
    table_pos = [0.5, 0, 0]
    table_orientation = p.getQuaternionFromEuler([0, 0, 0])
    bodies["table"] = p.loadURDF("table/table.urdf", table_pos, table_orientation)

    # Load robotic arm (KUKA iiwa)
    robot_pos = [0.0, 0.0, 0.62]  # Positioned at table height boundary
    robot_orientation = p.getQuaternionFromEuler([0, 0, 0])
    bodies["robot"] = p.loadURDF("kuka_iiwa/model.urdf", robot_pos, robot_orientation, useFixedBase=True)

    # Load tabletop target object 1 (cube)
    cube_pos = [0.5, 0.15, 0.65]
    cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025], rgbaColor=[0.9, 0.2, 0.2, 1.0])
    cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
    bodies["target_cube"] = p.createMultiBody(
        baseMass=0.1, baseCollisionShapeIndex=cube_collision, baseVisualShapeIndex=cube_visual, basePosition=cube_pos
    )

    # Load tabletop target object 2 (cylinder / bottle simulation)
    cyl_pos = [0.55, -0.15, 0.65]
    cyl_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.08, rgbaColor=[0.2, 0.8, 0.2, 1.0])
    cyl_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.08)
    bodies["target_cylinder"] = p.createMultiBody(
        baseMass=0.1, baseCollisionShapeIndex=cyl_collision, baseVisualShapeIndex=cyl_visual, basePosition=cyl_pos
    )

    # Load destination area / tray
    tray_pos = [0.4, 0.35, 0.63]
    tray_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.01], rgbaColor=[0.2, 0.4, 0.9, 0.8])
    tray_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.01])
    bodies["destination_tray"] = p.createMultiBody(
        baseMass=0.0, baseCollisionShapeIndex=tray_collision, baseVisualShapeIndex=tray_visual, basePosition=tray_pos
    )

    return bodies


def print_robot_joint_info(robot_id: int) -> None:
    """Extract and print robot joint index table, joint names, and types.

    Args:
        robot_id (int): PyBullet body ID of the robotic arm.
    """
    num_joints = p.getNumJoints(robot_id)
    joint_type_names = {
        p.JOINT_REVOLUTE: "Revolute",
        p.JOINT_PRISMATIC: "Prismatic",
        p.JOINT_SPHERICAL: "Spherical",
        p.JOINT_PLANAR: "Planar",
        p.JOINT_FIXED: "Fixed",
    }

    print("\n" + "=" * 65)
    print(f"Robotic Arm Joint Topology (Body ID: {robot_id}, Total Joints: {num_joints})")
    print("=" * 65)
    print(f"{'Index':<8} {'Joint Name':<30} {'Joint Type':<15} {'Link Name':<15}")
    print("-" * 65)

    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_type = joint_type_names.get(info[2], f"Unknown ({info[2]})")
        link_name = info[12].decode("utf-8")
        print(f"{i:<8} {joint_name:<30} {joint_type:<15} {link_name:<15}")

    print("=" * 65 + "\n")


def run_simulation_loop(physics_client: int) -> None:
    """Run physics steps in a loop until the PyBullet GUI window is closed by the user.

    Args:
        physics_client (int): Active physics client ID.
    """
    print("Simulation active. Press Ctrl+C or close the GUI window to exit.")
    try:
        while p.isConnected(physics_client):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        if p.isConnected(physics_client):
            p.disconnect(physics_client)
            print("PyBullet simulation disconnected cleanly.")


def main() -> None:
    """Main execution function for PyBullet feasibility scene."""
    try:
        client_id = initialize_simulation(use_gui=True)
        bodies = build_tabletop_scene()
        print_robot_joint_info(bodies["robot"])
        run_simulation_loop(client_id)
    except Exception as err:
        print(f"Error executing PyBullet feasibility scene: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
