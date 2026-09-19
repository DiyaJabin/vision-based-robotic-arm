"""Dependency-independent state-machine checks, not a physics simulation."""
from types import SimpleNamespace
import unittest
from core.contracts import PickOutcome
from robot.gripper import ConstraintGripper, GraspError
from robot.pick_and_place import LoopConfig, run_loop


class FakeBullet:
    JOINT_FIXED=4
    def __init__(self):
        self.created=[];self.removed=[]
        self.ee=(.4,0,.725)
        self.speed=(0,0,0)
    def getDynamicsInfo(self,*a,**kw): return (0.1,)
    def getLinkState(self,*a,**kw):
        q=(0,1,0,0)
        return (self.ee,q,None,None,self.ee,q,self.speed,(0,0,0))
    def getAABB(self,*a,**kw): return ((.375,-.025,.65),(.425,.025,.7))
    def getMatrixFromQuaternion(self,*a): return (-1,0,0,0,1,0,0,0,-1)
    def getBaseVelocity(self,*a,**kw): return ((0,0,0),(0,0,0))
    def getBasePositionAndOrientation(self,*a,**kw): return ((.4,0,.675),(0,0,0,1))
    def invertTransform(self,pos,q): return (tuple(-x for x in pos),q)
    def multiplyTransforms(self,p1,q1,p2,q2): return (tuple(a+b for a,b in zip(p1,p2)),q2)
    def createConstraint(self,*args,**kw): self.created.append((args,kw));return 42
    def changeConstraint(self,*a,**kw): pass
    def removeConstraint(self,identifier,**kw): self.removed.append(identifier)


class GripperStateTests(unittest.TestCase):
    def test_attach_release_and_no_duplicate_attachment(self):
        backend=FakeBullet();gripper=ConstraintGripper(0,6,backend=backend)
        with self.assertRaises(GraspError): gripper.attach_object(1)
        gripper.close_gripper(1)
        self.assertEqual(gripper.object_id,1)
        self.assertEqual(len(backend.created),1)
        with self.assertRaises(GraspError): gripper.attach_object(1)
        self.assertEqual(gripper.release_object(),1)
        self.assertEqual(backend.removed,[42])
        self.assertIsNone(gripper.object_id)
        self.assertIsNone(gripper.release_object())
        self.assertFalse(gripper.closed)

    def test_far_or_fast_object_is_not_attached(self):
        for far,fast in ((True,False),(False,True)):
            backend=FakeBullet()
            if far: backend.ee=(2,0,.725)
            if fast: backend.speed=(1,0,0)
            gripper=ConstraintGripper(0,6,backend=backend)
            with self.assertRaises(GraspError): gripper.close_gripper(1)
            self.assertEqual(backend.created,[])
            self.assertIsNone(gripper.object_id)


class FakeAdapter:
    def __init__(self):
        self.remaining=[1,2]
        self.observations=0
        self.holds=0
        self.failure=False
    def remaining_ids(self): return self.remaining
    def prepare_observation(self): pass
    def observe_and_rank(self):
        self.observations+=1
        return ([SimpleNamespace(object_id=self.remaining[0])] if self.remaining else []),[]
    def execute(self,candidate):
        if self.failure: return PickOutcome(candidate.object_id,"cube","failed","motion_timeout")
        self.remaining.remove(candidate.object_id)
        return PickOutcome(candidate.object_id,"cube","placed","verified","A")
    def hold(self): self.holds+=1


class LoopTests(unittest.TestCase):
    def test_reobserves_after_each_pick_and_returns_structured_outcomes(self):
        adapter=FakeAdapter();result=run_loop(adapter)
        self.assertEqual(result.status,"completed")
        self.assertEqual(result.observation_count,3)
        self.assertEqual(len(result.picks),2)
        self.assertEqual(result.remaining_object_ids,())
        self.assertEqual(adapter.holds,1)

    def test_motion_failure_stops(self):
        adapter=FakeAdapter();adapter.failure=True
        result=run_loop(adapter)
        self.assertEqual(result.status,"failed")
        self.assertEqual(result.reason,"motion_timeout")
        self.assertEqual(len(result.picks),1)
        self.assertEqual(adapter.holds,1)

    def test_empty_candidates_are_not_false_success(self):
        adapter=FakeAdapter();adapter.observe_and_rank=lambda:([],["cube:clearance"])
        result=run_loop(adapter)
        self.assertEqual(result.reason,"no_valid_targets")
        self.assertEqual(result.rejections,("cube:clearance",))
        self.assertEqual(result.remaining_object_ids,(1,2))

    def test_limits_and_exception_cleanup(self):
        result=run_loop(FakeAdapter(),LoopConfig(max_pick_attempts=1))
        self.assertEqual(result.reason,"pick_attempt_limit")
        self.assertEqual(len(result.picks),1)
        result=run_loop(FakeAdapter(),LoopConfig(max_observations=1))
        self.assertEqual(result.reason,"observation_limit")
        adapter=FakeAdapter()
        def fail(): raise RuntimeError("camera unavailable")
        adapter.observe_and_rank=fail
        self.assertEqual(run_loop(adapter).status,"failed")
        self.assertEqual(adapter.holds,1)


if __name__=="__main__": unittest.main()
