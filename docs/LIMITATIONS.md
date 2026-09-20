# System Limitations

This project is intentionally scoped as a simulation-first robotic perception system. The current repository does not yet include physical hardware validation, so the documented limitations below are based on the implemented code and the project architecture rather than measured real-world performance.

## Simulation-to-real gap

The system is designed around a PyBullet tabletop scene with virtual camera capture and synthetic object generation. This reduces setup cost and supports rapid iteration, but it does not guarantee equivalent behavior under real camera optics, lighting variation, object texture, and mechanical tolerances.

## Fixed overhead camera

The current scene and calibration pipeline assume an overhead RGB camera view with a fixed geometry. This is a simplification of real warehouse or lab environments and may not generalize to camera motion, different heights, or occluded views.

## RGB-only perception

The detection and validation pipeline relies on RGB image observations and color/contour heuristics. Depth information, segmentation masks, or multimodal sensing are not yet integrated into the documented perception flow.

## Limited object classes

The initial class set is restricted to `cube`, `cylinder`, and `box`. The repository explicitly notes that bottle and cup are future classes without assigned IDs or validation logic.

## Synthetic dataset dependence

The image-generation utilities and configuration files support synthetic object placement under controlled simulation conditions. They do not replace a real dataset with natural camera noise, background clutter, or non-ideal lighting.

## Experimental thresholds

The confidence thresholds and validation settings in `core/config.py` and `perception/hybrid_validator.py` are intentionally documented as configurable initial heuristics rather than scientifically validated operating parameters.

## Graspability assumptions

The graspability and clearance logic uses conservative, top-down, simple-grasp assumptions. The repository documentation notes that the gripper is simulation-only and that the configured weights are not experimentally optimized.

## Runtime verification status

PyBullet-dependent runtime verification, motion execution, collision checks, and full end-to-end pick-and-place execution remain pending in the dedicated Windows environment. The project includes dependency-independent tests, but not a full physics validation pass.

## No physical hardware validation yet

There is no evidence in this repository of a validated real robot, camera, or gripper setup. Any claims about deployment readiness would exceed the implemented scope.
