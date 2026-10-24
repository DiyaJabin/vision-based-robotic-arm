# Code Walkthrough

This guide explains how the project is assembled and how to study it. It starts with the idea behind each stage, then names the implementation details worth reading.

## 1. Start here

Read these files in order:

1. `simulation/run_pick_and_place.py` — the integrated runtime.
2. `simulation/scene.py` and `simulation/camera.py` — the simulated world and camera.
3. `perception/baseline_hsv.py` — the simplest detector.
4. `perception/yolo_detector.py` and `perception/hybrid_validator.py` — learned detection and gating.
5. `robot/pick_and_place.py`, `robot/kinematics.py`, and `robot/grasp_selector.py` — motion and safety.

Run the baseline first. It is the simplest complete path and gives every later perception method the same planning and control stack.

## 2. What the whole system does

The simulator creates a tabletop, a seven-joint KUKA iiwa, three coloured objects, and three destination bins. The camera produces a 640 x 480 BGR image. OpenCV or YOLO proposes objects. The system estimates an image pose, maps the centre to tabletop metres, checks reachability and clearance, ranks feasible candidates, and executes a guarded pick-and-place sequence.

The important design boundary is that perception proposes a target, while the robot layer decides whether that target is safe and reachable.

## 3. Repository structure

- `core/` contains shared class IDs, thresholds, and data contracts.
- `simulation/` creates the scene, camera, and integrated runner.
- `perception/` contains OpenCV, YOLO, hybrid validation, pose, and calibration.
- `robot/` contains IK, controller, clearance, grasp selection, gripper, and destinations.
- `data/scripts/` generates scenes, segmentation labels, datasets, and training output.
- `experiments/` records randomized trials and produces comparisons and plots.
- `tests/` contains focused unit and integration-contract tests.
- `docs/` contains this guide, architecture notes, limitations, and media.

## 4. How PyBullet creates the scene

Study `simulation/scene.py`, especially `connect_simulation`, `load_environment`, `load_robot`, `create_object`, `create_destinations`, and `load_complete_scene`.

`connect_simulation` opens GUI or Direct mode and sets gravity and timestep. `load_environment` creates the plane and table. Objects are dynamic bodies with known class names, colours, dimensions, and body IDs. The returned scene dictionary is the shared source of simulator handles used by camera, planning, and verification.

The physics engine settles objects before capture or observation. That matters because a box still moving would make both labels and grasp planning misleading.

## 5. How the KUKA URDF is loaded

`load_robot` loads the KUKA iiwa URDF. `robot/kinematics.py::Kinematics.__init__` discovers the revolute joints and the end-effector link from the loaded URDF instead of assuming arbitrary indices.

The project expects seven revolute joints. Joint limits, force limits, velocity limits, and the end-effector link are retained for later validation.

## 6. What seven DoF means

A degree of freedom is one independently controllable joint variable. The KUKA has seven revolute joint angles, so its configuration is a seven-value tuple in radians.

Seven joints give redundancy: more than one configuration can sometimes reach the same tool pose. The solver uses a current or configured seed to prefer a stable solution, then checks the returned solution against limits and forward-kinematics error.

## 7. How the fixed virtual camera works

Read `simulation/camera.py::camera_matrices`, `project_world_points`, `capture_bgr`, and `capture_bgr_and_segmentation`.

The camera has a fixed overhead view and a known view/projection matrix. RGB is returned as an 8-bit BGR image for OpenCV and YOLO. Depth and segmentation are available for simulator-side work. Runtime object XY comes from RGB plus homography; depth is not silently substituted for calibration.

Segmentation encodes visible body IDs. The dataset builder masks those IDs to create exact visible target pixels, while excluding robot, table, bins, and background.

## 8. RGB, depth, and segmentation

RGB answers “what does the camera see?” and is the runtime perception input. Depth could answer “how far is this pixel?” but is not the primary XY mapping method here. Segmentation answers “which simulator body produced this pixel?” and is privileged simulator information used for labels, association diagnostics, and debugging.

Keeping segmentation out of runtime RGB detection prevents the synthetic labels from pretending to be a real camera sensor.

## 9. OpenCV baseline pipeline

Read `perception/baseline_hsv.py::detect_objects`.

The baseline converts BGR to HSV, creates one colour mask per supported class, cleans it with morphology, finds external contours, filters by area, and calculates a centroid, bounding box, area, and minimum-area-rectangle orientation.

The returned `baseline_score` is an area-based heuristic in [0, 1]. It is not a probability and must never be compared directly with YOLO confidence.

## 10. HSV

HSV separates hue from brightness and saturation. Hue identifies the red, green, and blue target colours more reliably than raw BGR under this controlled simulation.

Red uses two hue ranges because OpenCV wraps red around the hue boundary. Green and blue use one range each. Real-world lighting would require more calibration.

## 11. Masks and morphology

`create_mask` combines colour ranges into a binary image. Morphological opening removes small isolated noise. Closing fills small gaps inside an object region.

The kernel and iteration counts are deliberate heuristics. They improve the synthetic scene but are not universal camera settings.

## 12. Contours

`cv2.findContours` turns connected mask regions into candidate shapes. Area limits reject tiny noise and oversized regions. `cv2.boundingRect` gives an axis-aligned pixel box; moments give the centroid.

A contour is only a colour/shape observation. It is not yet a safe robot target.

## 13. Centroids and bounding boxes

The centroid is computed from image moments in pixel coordinates ((u, v)), where (u) increases right and (v) increases downward. The bounding box is also in pixels and is used to crop the object for pose and hybrid validation.

## 14. Orientation estimation

`calculate_orientation` uses `minAreaRect` to estimate the longer image-plane axis. Rectangles have a meaningful projected orientation. A square has equivalent axes, and a top-view cylinder has no observable physical yaw, so the pose code leaves cylinder yaw unset.

This is an image angle, not a world-frame joint angle.

## 15. YOLOv8n detector

Read `perception/yolo_detector.py::YoloDetector`.

The adapter loads either pretrained or custom weights, calls Ultralytics, validates native outputs, and maps supported class names to project IDs. It deliberately does not assume COCO numeric IDs mean cube, cylinder, or box.

`detect_generic` keeps native model results separate from project contracts. `detect` returns only the three project classes.

## 16. YOLO inference

Runtime input is an 8-bit BGR NumPy image. The adapter passes that image to Ultralytics, which performs its standard resize, letterbox, colour conversion, normalization, and tensor preparation. The returned coordinates are converted back to the original 640 x 480 image pixel frame.

This avoids having separate hand-written preprocessing for training and runtime.

## 17. Confidence

YOLO confidence is a learned detector score, not a guaranteed probability. A high synthetic mAP does not guarantee high confidence on every runtime frame, especially when objects are small.

The final model detected the correct runtime classes, but measured confidences were approximately 0.2518 for the cylinder, 0.0828 for the cube, and 0.0368 for the box.

## 18. Hybrid validation

Read `perception/hybrid_validator.py::HybridValidator.validate`.

The gate is intentionally conservative:

- at least 0.80: accept the YOLO observation directly;
- 0.50 to below 0.80: inspect the corresponding colour contour and shape;
- 0.03 to below 0.50: require the same OpenCV contour checks as a bounded
  low-score agreement path;
- below 0.03: reject and reobserve.

The medium path calculates colour fraction, contour area, solidity, circularity or rectangularity, and aspect ratio. It returns a separate OpenCV validation score. It never replaces detector confidence with a heuristic score.

This does not lower the medium threshold. It adds a separate `0.03` minimum
detector floor and only permits a low score when the YOLO box contains a valid
contour for the same project class. In the controlled scene, all three YOLO
boxes matched valid OpenCV contours and the hybrid completed placement.

## 19. Pose estimation

Read `perception/pose_estimator.py::estimate_pose`.

The estimator takes the YOLO or OpenCV pixel box, finds the class-colour contour inside it, and returns a centre, visible extents, and optional image orientation. It retains pixel units until calibration.

## 20. Homography

Read `perception/calibration.py::compute_homography`.

A homography is a 3 x 3 projective transform between two planes. Here the source plane is image pixels and the destination plane is tabletop XY in metres. Four nondegenerate point correspondences are normalized, solved, and checked for reprojection error.

A homography does not infer arbitrary 3D depth. The runtime constructs class-specific parallel planes using known simulator geometry.

## 21. Pixel-to-world mapping

`pixel_to_world` applies the homography to ((u, v)) and divides by the homogeneous scale. The result is an ((x, y)) position in tabletop metres.

The integrated runner then associates that observation with a remaining simulator body. The image-derived point supplies the grasp location; simulator truth is used only for controlled association and verification.

## 22. Reachability

Read `robot/grasp_selector.py::evaluate_reachability`.

The candidate contains a pre-grasp, grasp, pre-place, and place pose. Each pose is checked against the table envelope, a measured chain-radius bound, IK feasibility, joint limits, and collision checks.

A target can be visible yet rejected because the arm cannot safely reach it.

## 23. Collision checking

Read `robot/kinematics.py::collision_reason` and `current_collision_reason`.

Candidate joint states are inspected in a saved temporary simulator state. The robot is checked against table, destination, and other-object bodies. Nonadjacent self-collision and payload collision are also checked.

The intended target body is passed in `allowed_bodies` during approach and grasp, because touching the target is required. This exception does not remove rejection for table, other objects, destinations, or unsafe self-contact.

## 24. Clearance

`evaluate_clearance` treats the tool as a conservative horizontal radius derived from the loaded flange geometry. It checks table edges and neighbouring object footprints in tabletop metres.

This is an approximation, not a complete physical gripper model.

## 25. Graspability

`graspability` combines available detection, validation, clearance, and reachability factors using configurable weights. OpenCV baseline scores and YOLO confidence remain separate inputs.

Candidates that fail a safety or feasibility gate receive no usable score and are removed before ranking.

## 26. Inverse kinematics

`Kinematics.solve` calls PyBullet IK with URDF-derived limits and a seed. It then resets joints temporarily, measures the actual forward-kinematics pose, and checks position and orientation residuals.

This second check prevents a solver return value from being treated as success without verifying the achieved pose.

## 27. Controller and waypoints

Read `robot/controller.py`.

The controller turns validated Cartesian poses into bounded joint motion. It checks IK and collision conditions before motor commands. The integrated runner uses named stages rather than an unbounded free-form trajectory.

Important units are metres for position, radians for angles, and seconds for timing.

## 28. Simulated fixed-constraint gripper

Read `robot/gripper.py::ConstraintGripper`.

The current KUKA URDF has no realistic fingers. The gripper therefore checks proximity, height, downward orientation, object dynamics, and motion speed before creating a fixed PyBullet constraint.

This models simulator attachment after a plausible grasp condition. It does not claim frictional contact or physical force closure.

## 29. Destination mapping

Read `robot/destinations.py::DestinationPlanner`.

Class names map to destination bins: cube to Bin A, cylinder to Bin B, and box to Bin C. The planner provides object placement geometry and tracks occupied destination capacity.

## 30. Placement verification

After release, the runner waits for the object to settle, checks contact with the intended bin, checks containment within the bin surface bounds, and checks final height. Only then is the pick marked `destination_contact_verified`.

## 31. Synthetic dataset generation

Read `data/scripts/build_yolo_dataset.py` and the reusable scene functions in `data/scripts/generate_dataset.py`.

Each scene randomizes supported object count, class combination, position, and yaw, then waits for physics settling. The camera captures BGR and segmentation. Only visible target body pixels become labels.

The final working dataset has 90 images: 62 train, 13 validation, and 15 test.

## 32. Segmentation-derived labels

`body_pixels` extracts the low body-ID bits from PyBullet's segmentation encoding. `yolo_box` computes visible pixel bounds, clips them to the 640 x 480 frame, and normalizes centre and size.

Robot links, table, bins, background, and invisible objects are never emitted as target labels.

## 33. YOLO label format

Each image has a matching text file. Every line is:

```text
class_id x_center y_center width height
```

All coordinates are normalized to [0, 1]. The class IDs are 0 cube, 1 cylinder, and 2 box.

## 34. Train, validation, and test split

The builder shuffles with a reproducible seed and targets 70/15/15. With 90 images, the actual resulting counts are 62/13/15. `validate_yolo_dataset.py` checks image/label pairs, class IDs, finite normalized coordinates, and positive boxes.

## 35. Model training

Read `data/scripts/train_yolo.py`.

YOLOv8n starts from `yolov8n.pt`, trains on `dataset.yaml`, uses CPU-compatible workers and batch size, and saves Ultralytics outputs under `runs/`. The verified final run completed 15 epochs on CPU.

## 36. Epoch

An epoch is one pass through the training images. More epochs are not automatically better: validation metrics and runtime generalization must improve without overfitting.

## 37. Precision

Precision is the fraction of predicted detections that are correct. Low precision means many false positives.

## 38. Recall

Recall is the fraction of ground-truth objects detected. Low recall means missed objects.

## 39. IoU

Intersection over Union compares a predicted and true box:

```text
IoU = intersection area / union area
```

A detection can have the right class but poor localization, so both classification and overlap matter.

## 40. mAP

mAP summarizes precision-recall behaviour across confidence thresholds and IoU requirements. The project reports mAP@0.5 and mAP@0.5:0.95 on the held-out synthetic test split.

The measured values are synthetic-data metrics only: precision 0.9886, recall 0.8586, mAP@0.5 0.9950, and mAP@0.5:0.95 0.8331.

## 41. Runtime evaluation

Read `experiments/run_trials.py` and `experiments/logger.py`.

A trial creates a fresh scene, randomizes object poses, runs the selected mode, and records grasp, placement, failure reason, retries, and execution time. Results are written as CSV and JSON.

## 42. Randomized smoke trials

The five-trial diagnostic set reported:

| Method | Grasp success | Placement success | Mean execution time |
| --- | ---: | ---: | ---: |
| OpenCV | 0/5 | 0/5 | 1.74 s |
| YOLO-only | 0/5 | 0/5 | 5.79 s |
| Hybrid | 0/5 | 0/5 | 5.76 s |

Each method attempted a grasp in 3/5 trials. The current logger records a
grasp as successful only when the corresponding pick outcome reaches verified
placement, so the fresh rows contain 0/5 completed grasps and 0/5 placements.
Each method had three `no_valid_targets` and two `MotionError:joint_limits`
failures.

This is a small diagnostic sample, not a statistically significant benchmark. Controlled fixed-scene demos and randomized robustness answer different questions.

## 43. Controlled demo versus randomized robustness

The controlled baseline, YOLO-only, and hybrid runs place all three objects. Randomized trials sometimes fail before placement because object poses change reachability, IK residuals, camera association, or collision geometry.

A successful controlled demo does not prove broad randomized robustness. Conversely, a randomized failure does not invalidate the deterministic pipeline; it identifies a scene or planning case to investigate.

## 44. Known limitations

The project is simulation-only. The dataset is synthetic. The gripper is a fixed constraint. The camera is fixed overhead. There is no physical robot validation.

The original hybrid bottleneck was confidence calibration/generalization for
small runtime targets, not a disabled validator. The observed targets were
approximately 15–31 pixels wide, and the model's raw scores stayed below the
configured medium threshold. Independent OpenCV agreement supplies a bounded
rescue path without accepting scores below 0.03.

## 45. Hybrid debugging and final outcome

The diagnostic tool `data/scripts/diagnose_runtime_confidence.py` captures a real integrated frame and prints class, score, pixel box, centre, and box area.

The evidence showed:

- correct class mapping;
- expected 640 x 480 BGR input;
- matching overhead camera geometry;
- correct but low-confidence detections;
- OpenCV independently agreed with the YOLO cube and cylinder boxes, while
  the blue class also produced a second destination-pad contour;
- no need to lower the 0.50/0.80 safety bands; the bounded agreement floor was
  sufficient for the controlled scene.

A 240-image experiment was started but stopped after poor early validation and slow CPU epochs. The verified 90-image model was preserved. The runtime-scale audit showed that the existing labels already covered 15–30 pixel objects, so no replacement dataset was justified for this sprint.

## 46. Full runtime from start to finish

1. `scene.load_complete_scene` creates the world and returns body IDs.
2. `SimulationAdapter.prepare_observation` moves the arm to a neutral pose and waits for settling.
3. The camera captures a BGR frame.
4. OpenCV or YOLO produces image candidates.
5. Hybrid mode applies confidence gating.
6. Pose estimation finds pixel centre and orientation.
7. Homography maps the centre to tabletop metres.
8. Simulator association identifies the remaining target body.
9. The planner constructs grasp and placement waypoints.
10. IK and collision checks reject unsafe candidates.
11. Clearance and graspability rank feasible candidates.
12. The controller approaches and grasps.
13. The robot lifts and transfers through guarded waypoints.
14. The gripper releases at the class-specific bin.
15. The simulator settles the object.
16. Contact, containment, and support height verify placement.
17. The loop reobserves until all objects are placed or bounds are reached.

## How I would rebuild this project from scratch

Each stage has a concept, a file to study here, and a small reconstruction challenge.

### Stage 1 — Create a PyBullet world

Learn: clients, gravity, timesteps, planes, and collision bodies.  
Study: `simulation/scene.py::connect_simulation`.  
Challenge: create a plane and drop one box in Direct mode.

### Stage 2 — Understand URDF and robot joints

Learn: links, joints, joint types, and joint limits.  
Study: `simulation/scene.py::load_robot`, `robot/kinematics.py`.  
Challenge: print every KUKA joint name and limit.

### Stage 3 — Add one coloured cube

Learn: visual shape versus collision shape and dynamic bodies.  
Study: `simulation/scene.py::create_object`.  
Challenge: spawn a cube at a different tabletop coordinate.

### Stage 4 — Add one fixed RGB camera

Learn: view matrices, projection matrices, and camera frames.  
Study: `simulation/camera.py::camera_matrices`.  
Challenge: move the camera target and observe the image change.

### Stage 5 — Render the camera image

Learn: RGBA buffers, BGR conversion, and image dimensions.  
Study: `simulation/camera.py::capture_bgr`.  
Challenge: save one rendered frame to disk.

### Stage 6 — Detect one object with OpenCV

Learn: HSV thresholding and binary masks.  
Study: `perception/baseline_hsv.py::create_mask`.  
Challenge: isolate only the red cube.

### Stage 7 — Find centroid and bounding box

Learn: contours, moments, and pixel coordinates.  
Study: `perception/baseline_hsv.py::detect_objects`.  
Challenge: draw the detected centre and box on the frame.

### Stage 8 — Calibrate pixel-to-table coordinates

Learn: planar correspondences and homography.  
Study: `perception/calibration.py`.  
Challenge: calibrate four known tabletop corners and map a test pixel.

### Stage 9 — Move the robot with IK

Learn: Cartesian poses, quaternions, and inverse kinematics.  
Study: `robot/kinematics.py::solve`.  
Challenge: move the flange above the cube without grasping it.

### Stage 10 — Add collision checks

Learn: temporary state inspection and closest-point collision queries.  
Study: `robot/kinematics.py::collision_reason`.  
Challenge: place an obstacle and reject a colliding waypoint.

### Stage 11 — Implement a simple simulated grasp

Learn: preconditions, constraints, and relative transforms.  
Study: `robot/gripper.py::ConstraintGripper.attach_object`.  
Challenge: attach only when XY and height tolerances pass.

### Stage 12 — Pick and place one object

Learn: guarded waypoint sequences and post-action verification.  
Study: `robot/pick_and_place.py`, `simulation/run_pick_and_place.py::execute`.  
Challenge: move one cube to one fixed destination.

### Stage 13 — Add multiple classes and destinations

Learn: class IDs, name-to-bin mappings, and capacity.  
Study: `core/config.py`, `robot/destinations.py`.  
Challenge: route a cylinder and box to different bins.

### Stage 14 — Generate synthetic images and segmentation labels

Learn: reproducibility, settling, segmentation IDs, and visible-pixel boxes.  
Study: `data/scripts/build_yolo_dataset.py`.  
Challenge: generate six images and inspect three preview labels.

### Stage 15 — Learn YOLO label format

Learn: normalized centre-width-height boxes and class indices.  
Study: `DATASET.md`, `data/scripts/validate_yolo_dataset.py`.  
Challenge: hand-check one label against its image.

### Stage 16 — Train YOLOv8n

Learn: epochs, batches, validation, checkpoints, and early stopping.  
Study: `data/scripts/train_yolo.py`.  
Challenge: train a short smoke model and compare validation output.

### Stage 17 — Integrate YOLO into runtime

Learn: adapter boundaries, native model IDs, and project contracts.  
Study: `perception/yolo_detector.py`.  
Challenge: print raw and project-mapped detections for one frame.

### Stage 18 — Add hybrid validation

Learn: confidence bands, independent evidence, and rejection paths.  
Study: `perception/hybrid_validator.py`.  
Challenge: create high, medium, and low synthetic detections and inspect decisions.

### Stage 19 — Add graspability and clearance

Learn: geometric safety margins and ranking functions.  
Study: `robot/grasp_selector.py`.  
Challenge: put two objects close together and observe clearance rejection.

### Stage 20 — Add evaluation and randomized testing

Learn: structured logs, failure categories, aggregation, and honest sample sizes.  
Study: `experiments/run_trials.py`, `experiments/logger.py`, `experiments/compare_methods.py`.  
Challenge: run three trials, inspect the CSV, and classify each failure manually.

After these stages, compare controlled demonstrations with randomized trials. Keep detector metrics, perception metrics, and manipulation metrics separate so each number answers a clear question.

