"""Pure planning checks; fake IK results are not physics verification."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from core.contracts import BoundingBox, ImagePoint, ObjectObservation, CartesianPose
from robot.destinations import DestinationPlanner, DestinationSurface, DestinationError
from robot.grasp_selector import (TableBounds,Footprint,ClearanceConfig,Feasibility,
    GraspCandidate,ScoreWeights,evaluate_clearance,evaluate_reachability,graspability,rank_candidates)


TABLE=TableBounds(-1,1,-1,1,.625)
POSE=CartesianPose((.4,0,.8),(0,1,0,0))


class FakeKinematics:
    base_position=(0,0,.625)
    max_radius=1.0
    reason=None
    def current_joints(self): return (0,)*7
    def solve(self,pose,seed=None): return SimpleNamespace(feasible=self.reason is None,joints=(0,)*7,reason=self.reason or "ok")
    def valid_joints(self,joints): return len(joints)==7
    def collision_reason(self,*args,**kwargs): return None


class ReachabilityTests(unittest.TestCase):
    def test_coarse_rejections(self):
        kin=FakeKinematics()
        result=evaluate_reachability(kin,[replace(POSE,position=(2,0,.8))],TABLE)
        self.assertEqual(result.reason,"outside_table_envelope")
        result=evaluate_reachability(kin,[replace(POSE,position=(.9,.9,.8))],TABLE)
        self.assertEqual(result.reason,"outside_chain_radius")
        result=evaluate_reachability(kin,[replace(POSE,position=(.4,0,.5))],TABLE)
        self.assertFalse(result.feasible)

    def test_ik_and_collision_reasons(self):
        kin=FakeKinematics();kin.reason="ik_residual"
        self.assertEqual(evaluate_reachability(kin,[POSE],TABLE).reason,"ik_residual")
        kin.reason=None; kin.collision_reason=lambda *a,**k:"arm_collision"
        self.assertEqual(evaluate_reachability(kin,[POSE],TABLE).reason,"arm_collision")
        kin.collision_reason=lambda *a,**k:None
        self.assertTrue(evaluate_reachability(kin,[POSE],TABLE).feasible)
        kin.valid_joints=lambda q:False
        self.assertEqual(evaluate_reachability(kin,[POSE],TABLE).reason,"joint_limits")


class ClearanceTests(unittest.TestCase):
    def setUp(self):
        self.target=Footprint(1,0,0,.025,.025)
        self.config=ClearanceConfig(.04,.01)

    def test_isolated_and_self(self):
        self.assertTrue(evaluate_clearance(self.target,[self.target],TABLE,self.config).feasible)

    def test_overlap_nearby_and_edge(self):
        self.assertFalse(evaluate_clearance(self.target,[Footprint(2,.05,0,.02,.02)],TABLE,self.config).feasible)
        self.assertTrue(evaluate_clearance(self.target,[Footprint(2,.2,0,.02,.02)],TABLE,self.config).feasible)
        edge=replace(self.target,x=.97)
        self.assertEqual(evaluate_clearance(edge,[],TABLE,self.config).reason,"table_edge_clearance")

    def test_direction_and_zero_margin_overlap(self):
        self.assertFalse(evaluate_clearance(self.target,[],TABLE,self.config,(1,0,0)).feasible)
        self.assertFalse(evaluate_clearance(self.target,[Footprint(2,0,0,.02,.02)],TABLE,ClearanceConfig(.04,0)).feasible)


class RankingTests(unittest.TestCase):
    def candidate(self,body,source="yolo",confidence=.8,clearance=.8):
        obs=ObjectObservation(0,"cube",BoundingBox(0,0,10,10),ImagePoint(5,5),source,
            detection_confidence=confidence if source=="yolo" else None,
            baseline_score=.2 if source=="opencv" else None)
        return GraspCandidate(body,obs,POSE,POSE,Footprint(body,.4,0,.025,.025),
                              Feasibility(True,"ok",1),Feasibility(True,"ok",clearance))

    def test_ranking_and_tie_break(self):
        low=self.candidate(2,confidence=.5)
        high=self.candidate(3,confidence=.9)
        equal=replace(high,object_id=1)
        self.assertEqual([c.object_id for c in rank_candidates([low,high,equal])],[1,3,2])

    def test_infeasible_and_collision_rejected(self):
        candidate=self.candidate(1)
        self.assertEqual(rank_candidates([replace(candidate,reachability=Feasibility(False,"outside"))]),[])
        self.assertIsNone(graspability(replace(candidate,collision_risk=1)))
        self.assertLess(graspability(replace(candidate,collision_risk=.3)),graspability(candidate))

    def test_baseline_has_no_detector_confidence(self):
        baseline=self.candidate(1,source="opencv")
        self.assertIsNotNone(graspability(baseline))
        invalid=replace(baseline,observation=replace(baseline.observation,detection_confidence=.2))
        self.assertIsNone(graspability(invalid))

    def test_weight_validation(self):
        with self.assertRaises(ValueError): ScoreWeights(clearance=-1)
        with self.assertRaises(ValueError): ScoreWeights(detection=0,validation=0,baseline=0,clearance=0,reachability=0)


class DestinationTests(unittest.TestCase):
    def planner(self):
        return DestinationPlanner([DestinationSurface(name,index,(x-.055,.255,.625),(x+.055,.345,.631))
                                   for name,index,x in (("A",1,.35),("B",2,.5),("C",3,.65))])

    def test_mapping_and_height(self):
        planner=self.planner()
        for name,bin_name in (("cube","A"),("cylinder","B"),("box","C")):
            result=planner.placement_for(name,(.05,.05,.05))
            self.assertEqual(result.bin_name,bin_name)
            self.assertAlmostEqual(result.object_position[2],.631+.025+.003)

    def test_capacity_and_size(self):
        planner=self.planner();planner.mark_occupied("A")
        with self.assertRaises(DestinationError): planner.placement_for("cube",(.05,)*3)
        with self.assertRaises(DestinationError): planner.placement_for("box",(.2,.1,.05))
        with self.assertRaises(DestinationError): planner.placement_for("cup",(.05,)*3)


if __name__=="__main__": unittest.main()
