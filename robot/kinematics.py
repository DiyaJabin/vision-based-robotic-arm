"""KUKA joint discovery and validated IK. All positions metres, angles radians."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import numpy as np
import pybullet as p


from core.contracts import CartesianPose


@dataclass(frozen=True)
class IKResult:
    feasible: bool
    joints: tuple[float, ...] | None
    reason: str
    position_error: float = math.inf
    orientation_error: float = math.inf


def downward_pose(x: float, y: float, z: float, yaw: float = 0.0) -> CartesianPose:
    return CartesianPose((x,y,z),tuple(p.getQuaternionFromEuler((0,math.pi,yaw))))


def manipulation_poses(x, y, object_top, place_xy, place_top, *, standoff=0.025, lift=0.16):
    """Named top-down flange poses; z uses known geometry, never RGB depth."""
    return {
        "pre_grasp": downward_pose(x,y,object_top+standoff+lift),
        "grasp": downward_pose(x,y,object_top+standoff),
        "lift": downward_pose(x,y,object_top+standoff+lift),
        "pre_place": downward_pose(*place_xy,place_top+standoff+lift),
        "place": downward_pose(*place_xy,place_top+standoff),
    }


class Kinematics:
    def __init__(self, robot_id: int, client_id: int = 0, end_effector_name="lbr_iiwa_link_7",
                 position_tolerance=0.004, orientation_tolerance=0.06):
        self.robot_id, self.client_id = robot_id, client_id
        infos=[p.getJointInfo(robot_id,i,physicsClientId=client_id)
               for i in range(p.getNumJoints(robot_id,physicsClientId=client_id))]
        matches=[info[0] for info in infos if info[12].decode()==end_effector_name]
        if len(matches)!=1:
            raise ValueError(f"Cannot identify end effector {end_effector_name!r}.")
        self.end_effector_link=matches[0]
        self.joints=tuple(info[0] for info in infos if info[2]==p.JOINT_REVOLUTE)
        if len(self.joints)!=7:
            raise ValueError("This integration expects the seven-joint KUKA iiwa arm.")
        self.lower=tuple(infos[i][8] for i in self.joints)
        self.upper=tuple(infos[i][9] for i in self.joints)
        self.forces=tuple(infos[i][10] for i in self.joints)
        self.velocity_limits=tuple(infos[i][11] for i in self.joints)
        # Initial bent-arm rest configuration, within the URDF limits.
        self.home=(-math.pi/2,0.4,0.,-1.2,0.,1.2,0.)
        if any(not math.isfinite(v) or v<=0 for v in (position_tolerance,orientation_tolerance)):
            raise ValueError("IK tolerances must be positive and finite.")
        self.position_tolerance=position_tolerance
        self.orientation_tolerance=orientation_tolerance
        base_com,base_q=p.getBasePositionAndOrientation(robot_id,physicsClientId=client_id)
        dynamics=p.getDynamicsInfo(robot_id,-1,physicsClientId=client_id)
        inverse=p.invertTransform(dynamics[3],dynamics[4])
        self.base_position=tuple(p.multiplyTransforms(base_com,base_q,*inverse)[0])
        # Conservative chain-length bound measured from loaded FK link frames.
        positions=[self.base_position]+[p.getLinkState(robot_id,j,computeForwardKinematics=True,
                       physicsClientId=client_id)[4] for j in self.joints]
        self.max_radius=sum(math.dist(a,b) for a,b in zip(positions,positions[1:]))

    def current_joints(self):
        return tuple(p.getJointState(self.robot_id,j,physicsClientId=self.client_id)[0] for j in self.joints)

    def current_pose(self):
        state=p.getLinkState(self.robot_id,self.end_effector_link,computeForwardKinematics=True,
                             physicsClientId=self.client_id)
        return CartesianPose(tuple(state[4]),tuple(state[5]))

    def valid_joints(self, joints):
        return len(joints)==len(self.joints) and all(math.isfinite(q) and lo-1e-6<=q<=hi+1e-6
                                                    for q,lo,hi in zip(joints,self.lower,self.upper))

    @contextmanager
    def inspection_configuration(self,joints):
        """Temporary FK/collision inspection; restored without stepping physics.

        This is not execution. Controllers use motor commands exclusively.
        """
        if not self.valid_joints(joints):
            raise ValueError("Invalid joint configuration.")
        state=p.saveState(physicsClientId=self.client_id)
        try:
            for j,q in zip(self.joints,joints):
                p.resetJointState(self.robot_id,j,q,physicsClientId=self.client_id)
            p.performCollisionDetection(physicsClientId=self.client_id)
            yield
        finally:
            p.restoreState(stateId=state,physicsClientId=self.client_id)
            p.removeState(state,physicsClientId=self.client_id)

    def solve(self, pose: CartesianPose, seed=None) -> IKResult:
        if not np.isfinite(pose.position).all() or len(pose.position)!=3:
            return IKResult(False,None,"invalid_target")
        orientation=np.asarray(pose.orientation,dtype=float)
        if orientation.shape!=(4,) or not np.isfinite(orientation).all() or np.linalg.norm(orientation)<1e-9:
            return IKResult(False,None,"invalid_orientation")
        orientation=orientation/np.linalg.norm(orientation)
        if math.dist(pose.position,self.base_position)>self.max_radius+1e-3:
            return IKResult(False,None,"outside_chain_radius")
        rest=tuple(seed) if seed is not None else self.current_joints()
        if not self.valid_joints(rest):
            return IKResult(False,None,"invalid_seed")
        best=IKResult(False,None,"ik_failed")
        # Use the supplied/current seed first, then a deterministic bent-arm seed.
        for start in (rest,self.home):
            with self.inspection_configuration(start):
                solution=p.calculateInverseKinematics(self.robot_id,self.end_effector_link,
                    targetPosition=pose.position,targetOrientation=orientation.tolist(),
                    lowerLimits=self.lower,upperLimits=self.upper,
                    jointRanges=tuple(hi-lo for lo,hi in zip(self.lower,self.upper)),
                    restPoses=start,maxNumIterations=400,residualThreshold=1e-6,
                    physicsClientId=self.client_id)
                joints=tuple(solution[:len(self.joints)])
                if not self.valid_joints(joints):
                    best=IKResult(False,None,"joint_limits")
                    continue
                for j,q in zip(self.joints,joints):
                    p.resetJointState(self.robot_id,j,q,physicsClientId=self.client_id)
                actual=self.current_pose()
                pos_error=math.dist(actual.position,pose.position)
                dot=min(1.,abs(float(np.dot(actual.orientation,orientation))))
                angle_error=2*math.acos(dot)
                success=pos_error<=self.position_tolerance and angle_error<=self.orientation_tolerance
                result=IKResult(success,joints if success else None,
                                "ok" if success else "ik_residual",pos_error,angle_error)
                if success:
                    return result
                if result.position_error<best.position_error:
                    best=result
        return best

    def current_collision_reason(self, obstacles, *, allowed_bodies=(), payload_id=None, penetration=0.001):
        """Arm, nonadjacent self, and carried-object penetration checks."""
        for body in obstacles:
            if body in allowed_bodies or body==self.robot_id:
                continue
            contacts=p.getClosestPoints(self.robot_id,body,distance=0,physicsClientId=self.client_id)
            for contact in contacts:
                if contact[3]>=0 and contact[8]<-penetration:
                    return f"arm_collision:body={body}:link={contact[3]}"
        for a in range(len(self.joints)):
            for b in range(a+2,len(self.joints)):
                contacts=p.getClosestPoints(self.robot_id,self.robot_id,0,
                    linkIndexA=a,linkIndexB=b,physicsClientId=self.client_id)
                if any(c[8]<-penetration for c in contacts):
                    return f"self_collision:{a}:{b}"
        if payload_id is not None:
            for body in obstacles:
                if body in (payload_id,self.robot_id):
                    continue
                contacts=p.getClosestPoints(payload_id,body,0,physicsClientId=self.client_id)
                if any(c[8]<-penetration for c in contacts):
                    return f"payload_collision:body={body}"
        return None

    def collision_reason(self, joints, obstacles, *, allowed_bodies=(), payload_id=None, penetration=0.001):
        """Inspect a candidate arm/payload pose; restore the entire live state.

        Constraints are not stepped during inspection. Carry the object's
        current relative flange transform explicitly for collision prediction.
        """
        relative=None
        if payload_id is not None:
            current=self.current_pose()
            body_pose=p.getBasePositionAndOrientation(payload_id,physicsClientId=self.client_id)
            relative=p.multiplyTransforms(*p.invertTransform(current.position,current.orientation),*body_pose)
        with self.inspection_configuration(joints):
            if relative is not None:
                current=self.current_pose()
                body_pose=p.multiplyTransforms(current.position,current.orientation,*relative)
                p.resetBasePositionAndOrientation(payload_id,*body_pose,physicsClientId=self.client_id)
                p.performCollisionDetection(physicsClientId=self.client_id)
            return self.current_collision_reason(obstacles,allowed_bodies=allowed_bodies,
                                                 payload_id=payload_id,penetration=penetration)
