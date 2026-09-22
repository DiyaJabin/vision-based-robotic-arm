"""Simulation-only proximity gripper for the fingerless KUKA iiwa.

No physical jaws, friction closure or force-closure claim: a fixed constraint
preserves the current relative pose after explicit grasp-condition checks.
"""
from dataclasses import dataclass
import math
import importlib


class GraspError(RuntimeError):
    pass


@dataclass(frozen=True)
class GripperConfig:
    """Experimental virtual attachment tolerances, metres and m/s."""
    standoff: float = 0.025
    xy_tolerance: float = 0.020
    z_tolerance: float = 0.015
    max_speed: float = 0.08
    max_angular_speed: float = 0.2
    max_force: float = 100.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Gripper parameters must be finite and positive.")


class ConstraintGripper:
    def __init__(self, robot_id: int, end_effector_link: int, client_id: int = 0,
                 config: GripperConfig | None = None, *, backend=None):
        self._p = backend if backend is not None else importlib.import_module("pybullet")
        self.robot_id, self.link, self.client_id = robot_id, end_effector_link, client_id
        self.config = config or GripperConfig()
        self.constraint_id = None
        self.object_id = None
        self.closed = False
        self.relative_flange_pose = None

    def open_gripper(self):
        self.release_object()
        self.closed = False

    def close_gripper(self, object_id: int | None = None):
        self.closed = True
        if object_id is not None:
            self.attach_object(object_id)

    def attach_object(self, object_id: int):
        if not self.closed:
            raise GraspError("gripper_is_open")
        if self.constraint_id is not None:
            raise GraspError("already_holding_object")
        if object_id == self.robot_id or self._p.getDynamicsInfo(object_id,-1,physicsClientId=self.client_id)[0] <= 0:
            raise GraspError("object_is_not_dynamic")
        state = self._p.getLinkState(self.robot_id, self.link, computeLinkVelocity=True,
                               computeForwardKinematics=True, physicsClientId=self.client_id)
        ee = state[4]
        lower, upper = self._p.getAABB(object_id,physicsClientId=self.client_id)
        centre = ((lower[0]+upper[0])/2, (lower[1]+upper[1])/2)
        cfg = self.config
        if math.dist(ee[:2], centre) > cfg.xy_tolerance:
            raise GraspError("object_outside_attachment_radius")
        if abs(ee[2]-(upper[2]+cfg.standoff)) > cfg.z_tolerance:
            raise GraspError("invalid_attachment_height")
        matrix = self._p.getMatrixFromQuaternion(state[5])
        if matrix[8] > -0.90:
            raise GraspError("end_effector_not_pointing_down")
        velocity, angular = self._p.getBaseVelocity(object_id,physicsClientId=self.client_id)
        if (math.hypot(*velocity) > cfg.max_speed or math.hypot(*state[6]) > cfg.max_speed
                or math.hypot(*angular)>cfg.max_angular_speed or math.hypot(*state[7])>cfg.max_angular_speed):
            raise GraspError("attachment_requires_slow_motion")
        position, orientation = self._p.getBasePositionAndOrientation(object_id,physicsClientId=self.client_id)
        inverse = self._p.invertTransform(state[0],state[1])
        relative = self._p.multiplyTransforms(*inverse, position,orientation)
        constraint = self._p.createConstraint(self.robot_id,self.link,object_id,-1,self._p.JOINT_FIXED,
            [0,0,0],relative[0],[0,0,0],parentFrameOrientation=relative[1],
            childFrameOrientation=[0,0,0,1],physicsClientId=self.client_id)
        self._p.changeConstraint(constraint,maxForce=cfg.max_force,physicsClientId=self.client_id)
        self.constraint_id, self.object_id = constraint, object_id
        inverse_flange = self._p.invertTransform(state[4],state[5])
        self.relative_flange_pose = self._p.multiplyTransforms(*inverse_flange,position,orientation)

    def flange_pose_for_object(self, position, orientation=(0,0,0,1)):
        """Use the actual attachment transform to place an upright object.

        This simulator-only transform does not replace image-based target XY.
        """
        from core.contracts import CartesianPose
        if self.relative_flange_pose is None:
            raise GraspError("no_attachment_transform")
        inverse = self._p.invertTransform(*self.relative_flange_pose)
        flange = self._p.multiplyTransforms(position,orientation,*inverse)
        return CartesianPose(tuple(flange[0]),tuple(flange[1]))

    def release_object(self):
        if self.constraint_id is not None:
            self._p.removeConstraint(self.constraint_id,physicsClientId=self.client_id)
        released = self.object_id
        self.constraint_id = self.object_id = None
        self.relative_flange_pose = None
        self.closed = False
        return released
