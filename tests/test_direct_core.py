"""Opt-in focused DIRECT checks. NOT executed/verified during implementation.

In the prepared Windows environment set RUN_PYBULLET_TESTS=1, then run:
python -m unittest tests.test_direct_core -v
Default discovery skips these tests; absence of dependencies is not a pass.
"""
import contextlib
import io
import math
import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_PYBULLET_TESTS")=="1", "PyBullet runtime verification explicitly deferred")
class DirectCoreTests(unittest.TestCase):
    def setUp(self):
        import pybullet as p
        from simulation import scene
        self.p,self.scene=p,scene
        self.client=scene.connect_simulation(use_gui=False)
        self.assertEqual(self.client,0)

    def tearDown(self):
        if self.p.isConnected(self.client): self.p.disconnect(self.client)

    def test_collision_settling_metadata_camera(self):
        from data.scripts.generate_dataset import wait_for_objects_to_settle,update_object_poses
        from simulation import camera
        environment=self.scene.load_environment()
        objects=self.scene.create_default_objects()
        records=[{"object_id":body} for body in objects.values()]
        for body in objects.values():
            self.assertTrue(self.p.getCollisionShapeData(body,-1))
        wait_for_objects_to_settle(records)
        update_object_poses(records)
        for record in records:
            body=record["object_id"]
            position,_=self.p.getBasePositionAndOrientation(body)
            self.assertEqual(tuple(record["world_position"][key] for key in ("x","y","z")),position)
            linear,angular=self.p.getBaseVelocity(body)
            self.assertLess(math.hypot(*linear),.005)
            self.assertLess(math.hypot(*angular),.05)
            self.assertTrue(self.p.getContactPoints(bodyA=body,bodyB=environment["table"]))
        frame=camera.capture_bgr()
        self.assertEqual(frame.shape,(480,640,3))

    def test_one_displaced_cube_pick_and_place(self):
        from simulation.run_pick_and_place import PipelineConfig,run_scene
        with contextlib.redirect_stdout(io.StringIO()):
            data=self.scene.load_complete_scene()
        for name,body in tuple(data["objects"].items()):
            if name!="cube":
                self.p.removeBody(body)
                del data["objects"][name]
        body=data["objects"]["cube"]
        # Scene initialization only; execution must move through motor controls.
        self.p.resetBasePositionAndOrientation(body,(.46,.08,self.scene.TABLETOP_Z+.025),
                                               self.p.getQuaternionFromEuler((0,0,.4)))
        result=run_scene(data,{body:"cube"},PipelineConfig(),client_id=self.client)
        self.assertEqual(result.status,"completed",repr(result))
        self.assertEqual(len(result.picks),1)
        self.assertEqual(result.picks[0].destination,"A")
        self.assertTrue(self.p.getContactPoints(bodyA=body,bodyB=data["destinations"]["destination_1"]))


if __name__=="__main__": unittest.main()
