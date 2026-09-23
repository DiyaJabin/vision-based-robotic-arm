"""Single-scene core integration, not an experiment runner.

Run with: python -m simulation.run_pick_and_place --mode baseline --direct
All PyBullet-dependent behaviour requires runtime verification before demo.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from core.config import CLASS_IDS, ConfidenceThresholds
from core.contracts import BoundingBox, CartesianPose, ImagePoint, ObjectObservation, PickOutcome, RunOutcome
from perception.calibration import compute_homography, pixel_to_world
from perception.hybrid_validator import HybridValidator, ValidationConfig
from perception.pose_estimator import estimate_pose
from perception.yolo_detector import YoloDetector
from robot.destinations import DestinationPlanner, DestinationError
from robot.grasp_selector import (TableBounds, Footprint, ClearanceConfig, GraspCandidate,
    ScoreWeights, evaluate_clearance, evaluate_reachability, rank_candidates)
from robot.gripper import ConstraintGripper, GripperConfig
from robot.pick_and_place import LoopConfig, run_loop


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = "baseline"
    weights: str = "yolov8n.pt"
    custom_weights: str | None = None
    confidence: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    # The KUKA flange collision geometry extends below the link frame. Keep a
    # larger runtime clearance before creating the simulation-only constraint.
    gripper: GripperConfig = field(default_factory=lambda: GripperConfig(standoff=0.050))
    loop: LoopConfig = field(default_factory=LoopConfig)
    association_distance: float = 0.035
    lift_height: float = 0.16
    clearance_margin: float = 0.008
    baseline_min_area: float = 60.0
    realtime: bool = False

    def __post_init__(self):
        if self.mode not in ("baseline", "yolo", "hybrid"):
            raise ValueError("Mode must be baseline, yolo, or hybrid.")
        for value in (self.association_distance,self.lift_height,self.clearance_margin,self.baseline_min_area):
            if not math.isfinite(value) or value<=0:
                raise ValueError("Pipeline dimensions/thresholds must be positive.")


def object_dimensions(class_name):
    """Known upright geometry; no learned RGB height claim."""
    from simulation import scene
    if class_name=="cube":
        return (scene.OBJECT_SIZE,)*3
    if class_name=="cylinder":
        return (2*scene.CYLINDER_RADIUS,2*scene.CYLINDER_RADIUS,scene.CYLINDER_HEIGHT)
    if class_name=="box":
        return tuple(2*v for v in scene.BOX_HALF_EXTENTS)
    raise ValueError("Unsupported class geometry.")


class SimulationAdapter:
    """Scene-owned adapter. Targets map body IDs to known simulated class names.

    Vision supplies grasp XY. Simulator poses identify which body to constrain,
    and provide collision/clearance and post-placement truth. They do not replace
    image-derived grasp centres. This privileged association is simulation-only.
    The existing foundation uses the default PyBullet client; require client 0.
    """
    def __init__(self,scene_data,targets,config: PipelineConfig | None = None,*,client_id=0,detector=None):
        import pybullet as p
        from simulation import scene, camera
        from robot.kinematics import Kinematics
        from robot.controller import RobotController, MotionConfig
        if client_id!=0:
            raise ValueError("Current scene/camera/settling foundation requires default client 0.")
        self.p,self.client_id=p,client_id
        self.scene,self.camera=scene,camera
        self.config=config or PipelineConfig()
        if any(name not in CLASS_IDS for name in targets.values()):
            raise ValueError("Targets must use supported class names.")
        self.targets=dict(targets)
        self.completed=set()
        self.data=scene_data
        self.kin=Kinematics(scene_data["robot"],client_id)
        self.gripper=ConstraintGripper(scene_data["robot"],self.kin.end_effector_link,client_id,self.config.gripper)
        self.obstacles=tuple(scene_data["environment"].values())+tuple(scene_data["destinations"].values())+tuple(targets)
        self.controller=RobotController(self.kin,MotionConfig(realtime=self.config.realtime),self.obstacles)
        lower,upper=p.getAABB(scene_data["environment"]["table"],physicsClientId=client_id)
        self.table=TableBounds(lower[0],upper[0],lower[1],upper[1],upper[2])
        if abs(self.table.surface_z-scene.TABLETOP_Z)>0.005:
            raise ValueError("Configured table height disagrees with loaded collision geometry.")
        self.destinations=DestinationPlanner.from_scene(scene_data["destinations"],client_id)
        # Bounding sphere of flange AABB: orientation-independent conservative tool radius.
        lo,hi=p.getAABB(scene_data["robot"],self.kin.end_effector_link,physicsClientId=client_id)
        self.clearance=ClearanceConfig(math.dist(lo,hi)/2,self.config.clearance_margin)
        self.detector=detector
        if self.config.mode in ("yolo", "hybrid") and detector is None:
            self.detector=YoloDetector(self.config.weights,custom_weights=self.config.custom_weights)
        self.validator=HybridValidator(self.config.confidence,self.config.validation)
        self.homographies={}
        # Four references use the scene's existing sampling workspace, not robot limits.
        world_xy=((scene.WORKSPACE_X[0],scene.WORKSPACE_Y[0]),
                  (scene.WORKSPACE_X[1],scene.WORKSPACE_Y[0]),
                  (scene.WORKSPACE_X[1],scene.WORKSPACE_Y[1]),
                  (scene.WORKSPACE_X[0],scene.WORKSPACE_Y[1]))
        for name in CLASS_IDS:
            centre_z=self.table.surface_z+object_dimensions(name)[2]/2
            image=camera.project_world_points([(x,y,centre_z) for x,y in world_xy])
            self.homographies[name]=compute_homography(image,world_xy)

    def remaining_ids(self):
        return tuple(sorted(set(self.targets)-self.completed))

    def hold(self):
        self.controller.hold()

    def prepare_observation(self):
        if self.gripper.object_id is not None:
            raise RuntimeError("Cannot reobserve while holding an unplaced object.")
        # The normal bent-arm home pose occludes the right side of this fixed
        # overhead camera. Use the neutral observation pose so all tabletop
        # targets remain visible while retaining controller collision checks.
        self.controller.move_to_joint_positions((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        from data.scripts.generate_dataset import wait_for_objects_to_settle
        if self.targets:
            wait_for_objects_to_settle([{"object_id":body} for body in self.targets])

    def _footprints(self):
        footprints=[]
        for body in tuple(self.targets)+tuple(self.data["destinations"].values()):
            lo,hi=self.p.getAABB(body,physicsClientId=self.client_id)
            footprints.append(Footprint(body,(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(hi[0]-lo[0])/2,(hi[1]-lo[1])/2))
        return footprints

    def _associate(self,class_name,xy,used,image_point=None):
        matches=[]
        for body in self.remaining_ids():
            if self.targets[body]!=class_name or body in used:
                continue
            position,_=self.p.getBasePositionAndOrientation(body,physicsClientId=self.client_id)
            distance=math.dist(xy,position[:2])
            if distance<=self.config.association_distance:
                matches.append((distance,body))
        matches.sort()
        if not matches and image_point is not None:
            # The homography maps a parallel object-top plane. Perspective
            # shifts the contour centroid slightly toward the camera, so use
            # the camera projection only to associate the known simulated body
            # while retaining the image-derived XY for the grasp pose.
            projected=[]
            for body in self.remaining_ids():
                if self.targets[body]!=class_name or body in used:
                    continue
                position,_=self.p.getBasePositionAndOrientation(body,physicsClientId=self.client_id)
                top=position[2]+object_dimensions(class_name)[2]/2
                uv=self.camera.project_world_points([(position[0],position[1],top)])[0]
                projected.append((math.dist(image_point,(uv[0],uv[1])),body))
            projected.sort()
            if projected and projected[0][0] <= 30.0:
                return projected[0][1]
        if not matches:
            raise ValueError("no_simulated_body_near_visual_centre")
        if len(matches)>1 and matches[1][0]-matches[0][0]<0.005:
            raise ValueError("ambiguous_simulated_body_association")
        return matches[0][1]

    def observe_and_rank(self):
        from perception.baseline_hsv import detect_objects
        from robot.kinematics import downward_pose
        frame=self.camera.capture_bgr()
        observations=[]
        rejected=[]
        if self.config.mode=="baseline":
            for item in detect_objects(frame,min_area=self.config.baseline_min_area):
                b=item["bbox"]; c=item["center"]
                observations.append(ObjectObservation(item["class_id"],item["class_name"],
                    BoundingBox(b["x"],b["y"],b["width"],b["height"]),ImagePoint(c["x"],c["y"]),
                    "opencv",baseline_score=item["baseline_score"]))
        else:
            for detection in self.detector.detect(frame):
                if self.config.mode == "yolo":
                    observations.append(ObjectObservation(detection.class_id,detection.class_name,
                        detection.bbox,detection.centre,"yolo",detection_confidence=detection.confidence))
                    continue
                validation=self.validator.validate(frame,detection)
                if validation.accepted:
                    observations.append(ObjectObservation(detection.class_id,detection.class_name,detection.bbox,
                        detection.centre,"yolo",detection_confidence=detection.confidence,
                        validation_score=validation.validation_score))
                else:
                    rejected.append(f"{detection.class_name}:{validation.reason}")
        footprints=self._footprints()
        candidates=[]
        used=set()
        for observation in observations:
            name=observation.class_name
            try:
                pose=estimate_pose(frame,observation.bbox,name)
                matrix=self.homographies[name]
                xy=pixel_to_world(matrix,pose.centre.u,pose.centre.v)
                body=self._associate(name,xy,used,(pose.centre.u,pose.centre.v))
                # A raised/fallen/tilted object violates the known upright top-plane assumption.
                position,orientation=self.p.getBasePositionAndOrientation(body,physicsClientId=self.client_id)
                dimensions=object_dimensions(name)
                up=self.p.getMatrixFromQuaternion(orientation)[8]
                if abs(position[2]-(self.table.surface_z+dimensions[2]/2))>0.012 or up<0.97:
                    raise ValueError("object_not_upright_on_table_plane")
                yaw=0.
                if pose.orientation_deg is not None:
                    angle=math.radians(pose.orientation_deg)
                    end=pixel_to_world(matrix,pose.centre.u+10*math.cos(angle),pose.centre.v+10*math.sin(angle))
                    yaw=math.atan2(end[1]-xy[1],end[0]-xy[0])
                top=self.table.surface_z+dimensions[2]
                grasp=downward_pose(*xy,top+self.config.gripper.standoff,yaw)
                pre=downward_pose(*xy,grasp.position[2]+self.config.lift_height,yaw)
                placement=self.destinations.placement_for(name,dimensions)
                destination_top=placement.object_position[2]+dimensions[2]/2
                place=downward_pose(*placement.object_position[:2],destination_top+self.config.gripper.standoff)
                pre_place=downward_pose(*placement.object_position[:2],place.position[2]+self.config.lift_height)
                actual=next(f for f in footprints if f.object_id==body)
                target=Footprint(body,*xy,actual.half_x,actual.half_y)
                clearance=evaluate_clearance(target,footprints,self.table,self.clearance)
                reachable=evaluate_reachability(self.kin,(pre,grasp,pre_place,place),self.table,
                    obstacles=self.obstacles,allowed_bodies=(body,))
                candidate=GraspCandidate(body,observation,grasp,pre,target,reachable,clearance)
                candidates.append(candidate)
                used.add(body)
                if not reachable.feasible or not clearance.feasible:
                    rejected.append(f"{name}:{reachable.reason if not reachable.feasible else clearance.reason}")
            except (ValueError,DestinationError) as error:
                rejected.append(f"{name}:{error}")
        return rank_candidates(candidates,self.config.score_weights),rejected

    def execute(self,candidate):
        from robot.controller import MotionError
        from robot.gripper import GraspError
        from data.scripts.generate_dataset import wait_for_objects_to_settle
        body,name=candidate.object_id,candidate.observation.class_name
        placement=self.destinations.placement_for(name,object_dimensions(name))
        try:
            self.gripper.open_gripper()
            self.controller.move_end_effector(candidate.pre_grasp_pose,allowed_bodies=(body,),linear=False)
            self.controller.move_end_effector(candidate.grasp_pose,allowed_bodies=(body,))
            self.gripper.close_gripper(body)
            self.controller.payload_id=body
            self.controller.move_end_effector(candidate.pre_grasp_pose,allowed_bodies=(body,))
            place=self.gripper.flange_pose_for_object(placement.object_position,placement.object_orientation)
            flight_z=max(candidate.pre_grasp_pose.position[2],place.position[2]+self.config.lift_height)
            over_source=CartesianPose((candidate.pre_grasp_pose.position[0],candidate.pre_grasp_pose.position[1],flight_z),place.orientation)
            over_destination=CartesianPose((place.position[0],place.position[1],flight_z),place.orientation)
            self.controller.execute_waypoints((over_source,over_destination,place),allowed_bodies=(body,))
            self.gripper.release_object()
            self.controller.payload_id=None
            self.controller.move_end_effector(over_destination,allowed_bodies=(body,))
            wait_for_objects_to_settle([{"object_id":body}])
            lo,hi=self.p.getAABB(body,physicsClientId=self.client_id)
            surface=self.destinations.surfaces[placement.bin_name]
            contacts=self.p.getContactPoints(bodyA=body,bodyB=placement.body_id,physicsClientId=self.client_id)
            contained=all(lo[d]>=surface.lower[d]-0.002 and hi[d]<=surface.upper[d]+0.002 for d in (0,1))
            if not contacts or not contained or abs(lo[2]-surface.upper[2])>0.008:
                raise MotionError("placement_not_supported_in_destination")
            self.completed.add(body)
            self.destinations.mark_occupied(placement.bin_name)
            return PickOutcome(body,name,"placed","destination_contact_verified",placement.bin_name)
        except Exception as error:
            self.controller.hold()
            return PickOutcome(body,name,"failed",f"{type(error).__name__}:{error}",placement.bin_name)


def run_scene(scene_data,targets,config: PipelineConfig | None = None,*,client_id=0,detector=None) -> RunOutcome:
    """Run an existing scene and return outcomes; caller owns connection cleanup."""
    adapter=SimulationAdapter(scene_data,targets,config,client_id=client_id,detector=detector)
    return run_loop(adapter,adapter.config.loop)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("baseline","yolo","hybrid"),default="baseline")
    parser.add_argument("--weights",default="yolov8n.pt")
    parser.add_argument("--custom-weights")
    parser.add_argument("--direct",action="store_true")
    parser.add_argument("--high-confidence",type=float,default=0.8)
    parser.add_argument("--medium-confidence",type=float,default=0.5)
    parser.add_argument("--max-observations",type=int,default=12)
    parser.add_argument("--max-picks",type=int,default=6)
    args=parser.parse_args()
    config=PipelineConfig(mode=args.mode,weights=args.weights,custom_weights=args.custom_weights,
        confidence=ConfidenceThresholds(args.high_confidence,args.medium_confidence),
        loop=LoopConfig(args.max_observations,args.max_picks),realtime=not args.direct)
    import pybullet as p
    from simulation import scene
    if p.isConnected(0):
        raise RuntimeError("Run this entry point in a process without an existing PyBullet client.")
    client=scene.connect_simulation(use_gui=not args.direct)
    try:
        data=scene.load_complete_scene()
        targets={body:name for name,body in data["objects"].items()}
        result=run_scene(data,targets,config,client_id=client)
        print(f"Core run: {result.status}: {result.reason}")
        for outcome in result.picks:
            print(f"{outcome.class_name}: {outcome.status}: {outcome.reason}; bin={outcome.destination}")
        for reason in result.rejections:
            print(f"Rejected: {reason}")
        print(f"Remaining objects: {result.remaining_object_ids}")
        return 0 if result.status=="completed" else 1
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__=="__main__":
    raise SystemExit(main())
