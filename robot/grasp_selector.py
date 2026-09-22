"""Reachability, top-down clearance, and configurable candidate ranking.

Pure geometry/scoring is dependency-independent. A Kinematics instance supplies
actual URDF-derived radius, validated IK, and optional collision checks.
"""
from dataclasses import dataclass, field
import math
from core.config import CLASS_IDS
from core.contracts import CartesianPose, ObjectObservation


@dataclass(frozen=True)
class TableBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    surface_z: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()) or self.x_min>=self.x_max or self.y_min>=self.y_max:
            raise ValueError("Invalid tabletop bounds.")


@dataclass(frozen=True)
class Footprint:
    """Conservative world-XY AABB of one object, metres."""
    object_id: int
    x: float
    y: float
    half_x: float
    half_y: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x,self.y,self.half_x,self.half_y)) or min(self.half_x,self.half_y)<=0:
            raise ValueError("Footprint extents must be positive and finite.")


@dataclass(frozen=True)
class ClearanceConfig:
    """Initial margin for a vertical virtual tool; no physical jaw dimensions.

    tool_radius is supplied from the loaded flange collision AABB bounding
    sphere, not an invented gripper width. This is a conservative approximation.
    """
    tool_radius: float
    safety_margin: float = 0.008
    score_distance: float = 0.08

    def __post_init__(self):
        if any(not math.isfinite(v) or v<0 for v in (self.tool_radius,self.safety_margin)) or not math.isfinite(self.score_distance) or self.score_distance<=0:
            raise ValueError("Invalid clearance parameters.")


@dataclass(frozen=True)
class Feasibility:
    feasible: bool
    reason: str
    score: float = 0.0
    margin: float = 0.0


def evaluate_clearance(target: Footprint, neighbours, table: TableBounds,
                       config: ClearanceConfig, approach=(0.,0.,-1.)) -> Feasibility:
    if tuple(approach)!=(0.,0.,-1.):
        return Feasibility(False,"unsupported_approach_direction")
    # Inflate whichever is larger: target footprint or virtual tool cross-section.
    hx=max(target.half_x,config.tool_radius)
    hy=max(target.half_y,config.tool_radius)
    edge=min(target.x-hx-table.x_min,table.x_max-target.x-hx,
             target.y-hy-table.y_min,table.y_max-target.y-hy)
    if edge<config.safety_margin:
        return Feasibility(False,"table_edge_clearance",margin=edge)
    clearance=edge
    for other in neighbours:
        if other.object_id==target.object_id:
            continue
        dx=abs(other.x-target.x)-(hx+other.half_x)
        dy=abs(other.y-target.y)-(hy+other.half_y)
        distance=math.hypot(max(0.,dx),max(0.,dy))
        if distance<config.safety_margin or (dx<0 and dy<0):
            return Feasibility(False,f"neighbour_clearance:{other.object_id}",margin=distance)
        clearance=min(clearance,distance)
    return Feasibility(True,"ok",min(1.,clearance/config.score_distance),clearance)


def evaluate_reachability(kinematics, poses, table: TableBounds, *, obstacles=(), allowed_bodies=()):
    """Table envelope is measured geometry, not a claimed reachable workspace."""
    seed=kinematics.current_joints()
    for pose in poses:
        x,y,z=pose.position
        if not all(map(math.isfinite,(x,y,z))):
            return Feasibility(False,"invalid_target")
        if not table.x_min<=x<=table.x_max or not table.y_min<=y<=table.y_max or z<table.surface_z:
            return Feasibility(False,"outside_table_envelope")
        if math.dist(pose.position,kinematics.base_position)>kinematics.max_radius:
            return Feasibility(False,"outside_chain_radius")
        result=kinematics.solve(pose,seed=seed)
        if not result.feasible:
            return Feasibility(False,result.reason)
        if not kinematics.valid_joints(result.joints):
            return Feasibility(False,"joint_limits")
        reason=kinematics.collision_reason(result.joints,obstacles,allowed_bodies=allowed_bodies)
        if reason:
            return Feasibility(False,reason)
        seed=result.joints
    return Feasibility(True,"ok",1.)


@dataclass(frozen=True)
class ScoreWeights:
    """Initial configurable weights, not optimized or benchmarked."""
    detection: float = 0.35
    validation: float = 0.20
    clearance: float = 0.25
    reachability: float = 0.20
    baseline: float = 0.35
    collision_penalty: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v<0 for v in vars(self).values()):
            raise ValueError("Weights must be finite and nonnegative.")
        if self.clearance+self.reachability+self.baseline+self.detection+self.validation<=0:
            raise ValueError("At least one positive weight is required.")


@dataclass(frozen=True)
class GraspCandidate:
    object_id: int
    observation: ObjectObservation
    grasp_pose: CartesianPose
    pre_grasp_pose: CartesianPose
    footprint: Footprint
    reachability: Feasibility
    clearance: Feasibility
    collision_risk: float = 0.0
    score: float = 0.0
    rejection_reason: str | None = None


def graspability(candidate: GraspCandidate, weights: ScoreWeights | None = None) -> float | None:
    """Normalise over available factors; never substitute baseline for YOLO confidence."""
    cfg=weights or ScoreWeights()
    if candidate.rejection_reason or not candidate.reachability.feasible or not candidate.clearance.feasible:
        return None
    obs=candidate.observation
    if CLASS_IDS.get(obs.class_name)!=obs.class_id or obs.source not in ("opencv","yolo"):
        return None
    if obs.source=="opencv" and (obs.detection_confidence is not None or obs.baseline_score is None):
        return None
    if obs.source=="yolo" and obs.detection_confidence is None:
        return None
    values=[(candidate.clearance.score,cfg.clearance),(candidate.reachability.score,cfg.reachability)]
    if obs.source=="yolo":
        values.append((obs.detection_confidence,cfg.detection))
        if obs.validation_score is not None:
            values.append((obs.validation_score,cfg.validation))
    else:
        values.append((obs.baseline_score,cfg.baseline))
    if any(not math.isfinite(v) or not 0<=v<=1 for v,_ in values):
        return None
    if not math.isfinite(candidate.collision_risk) or not 0<=candidate.collision_risk<1:
        return None
    total=sum(w for _,w in values)
    if total<=0:
        return None
    return sum(v*w for v,w in values)/total-cfg.collision_penalty*candidate.collision_risk


def rank_candidates(candidates, weights: ScoreWeights | None = None):
    from dataclasses import replace
    ranked=[]
    for candidate in candidates:
        score=graspability(candidate,weights)
        if score is not None:
            ranked.append(replace(candidate,score=score))
    return sorted(ranked,key=lambda c:(-c.score,c.object_id))
