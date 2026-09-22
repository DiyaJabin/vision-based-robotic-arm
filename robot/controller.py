"""Bounded motor-driven motion; no perception or instantaneous robot resets."""
from dataclasses import dataclass
import math
import time
import numpy as np
import pybullet as p
from robot.kinematics import CartesianPose, Kinematics


class MotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class MotionConfig:
    time_step: float = 1/240
    joint_speed: float = 0.65
    joint_tolerance: float = 0.008
    settle_timeout: float = 3.0
    cartesian_step: float = 0.025
    realtime: bool = False

    def __post_init__(self):
        if any(not math.isfinite(v) or v<=0 for k,v in vars(self).items() if k!='realtime'):
            raise ValueError("Motion parameters must be positive and finite.")


class RobotController:
    def __init__(self, kinematics: Kinematics, config: MotionConfig | None = None, obstacles=()):
        self.kinematics=kinematics
        self.config=config or MotionConfig()
        self.obstacles=tuple(obstacles)
        self.payload_id=None

    def _command(self,positions):
        kin=self.kinematics
        p.setJointMotorControlArray(kin.robot_id,kin.joints,p.POSITION_CONTROL,
            targetPositions=list(positions),forces=list(kin.forces),
            positionGains=[0.3]*len(kin.joints),velocityGains=[1.]*len(kin.joints),
            physicsClientId=kin.client_id)

    def _step(self, allowed_bodies=()):
        p.stepSimulation(physicsClientId=self.kinematics.client_id)
        reason=self.kinematics.current_collision_reason(self.obstacles,
            allowed_bodies=allowed_bodies,payload_id=self.payload_id)
        if reason:
            raise MotionError(reason)
        if self.config.realtime:
            time.sleep(self.config.time_step)

    def hold(self):
        self._command(self.kinematics.current_joints())

    def move_to_joint_positions(self, target, *, duration=None, allowed_bodies=()):
        kin,cfg=self.kinematics,self.config
        target=np.asarray(target,dtype=float)
        if not kin.valid_joints(target):
            raise MotionError("invalid_joint_target")
        start=np.asarray(kin.current_joints())
        speed=min(cfg.joint_speed,min(kin.velocity_limits))
        minimum=max(0.15,1.5*float(np.max(np.abs(target-start)))/speed)
        duration=minimum if duration is None else max(minimum,float(duration))
        if not math.isfinite(duration) or duration>30:
            raise MotionError("invalid_motion_duration")
        steps=math.ceil(duration/cfg.time_step)
        # Check interpolated joint path before any motor movement.
        for alpha in np.linspace(0,1,max(2,math.ceil(float(np.max(np.abs(target-start)))/0.08)+1)):
            reason=kin.collision_reason(start+(target-start)*alpha,self.obstacles,
                allowed_bodies=allowed_bodies,payload_id=self.payload_id)
            if reason:
                raise MotionError(reason)
        try:
            for i in range(1,steps+1):
                alpha=i/steps
                smooth=alpha*alpha*(3-2*alpha)
                self._command(start+(target-start)*smooth)
                self._step(allowed_bodies)
            self._command(target)
            for _ in range(math.ceil(cfg.settle_timeout/cfg.time_step)):
                self._step(allowed_bodies)
                states=p.getJointStates(kin.robot_id,kin.joints,physicsClientId=kin.client_id)
                if max(abs(s[0]-q) for s,q in zip(states,target))<cfg.joint_tolerance and max(abs(s[1]) for s in states)<0.03:
                    return
            raise MotionError("joint_motion_timeout")
        except Exception:
            self.hold()
            raise

    def move_end_effector(self,pose: CartesianPose, *, allowed_bodies=(), linear=True):
        kin=self.kinematics
        if len(pose.position)!=3 or not all(math.isfinite(v) for v in pose.position):
            raise MotionError("invalid_cartesian_position")
        if len(pose.orientation)!=4 or not all(math.isfinite(v) for v in pose.orientation) or math.hypot(*pose.orientation)<1e-9:
            raise MotionError("invalid_cartesian_orientation")
        orientation=tuple(v/math.hypot(*pose.orientation) for v in pose.orientation)
        pose=CartesianPose(pose.position,orientation)
        start=kin.current_pose()
        count=max(1,math.ceil(math.dist(start.position,pose.position)/self.config.cartesian_step)) if linear else 1
        # Plan the full segment before executing it, maintaining IK seed continuity.
        targets=[]
        seed=kin.current_joints()
        for i in range(1,count+1):
            alpha=i/count
            position=tuple(a+(b-a)*alpha for a,b in zip(start.position,pose.position))
            # Quaternion normalized lerp follows the shortest hemisphere.
            q0=np.asarray(start.orientation); q1=np.asarray(pose.orientation)
            if np.dot(q0,q1)<0: q1=-q1
            q=q0+(q1-q0)*alpha; q=q/np.linalg.norm(q)
            result=kin.solve(CartesianPose(position,tuple(q)),seed=seed)
            if not result.feasible:
                raise MotionError(result.reason)
            targets.append(result.joints); seed=result.joints
        for joints in targets:
            self.move_to_joint_positions(joints,allowed_bodies=allowed_bodies)
        actual=kin.current_pose()
        orientation_error=2*math.acos(min(1.,abs(float(np.dot(actual.orientation,pose.orientation)))))
        if math.dist(actual.position,pose.position)>kin.position_tolerance*2 or orientation_error>kin.orientation_tolerance*2:
            raise MotionError("end_effector_tracking_error")

    def move_home(self):
        self.move_to_joint_positions(self.kinematics.home)

    def execute_waypoints(self,waypoints, *, allowed_bodies=()):
        for pose in waypoints:
            self.move_end_effector(pose,allowed_bodies=allowed_bodies)
