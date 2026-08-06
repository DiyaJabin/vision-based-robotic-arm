# System Architecture

This document details the theoretical system architecture for the **Vision-Based Robotic Arm for Intelligent Tabletop Pick-and-Place**.

---

## Architecture Overview Pipeline

The system is structured as a sequential perception-to-action pipeline operating in a PyBullet tabletop simulation:

```text
+-------------------------------------------------------+
|                PyBullet Tabletop Scene                |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|             Virtual Overhead RGB Camera               |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                 Image Preprocessing                   |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|        OpenCV Baseline OR YOLOv8n Detection          |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|    Confidence-Gated Colour and Contour Validation     |  <-- Proposed Contribution
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|       Object Centre & Orientation Estimation          |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|       Pixel-to-World Coordinate Conversion            |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|         Reachability & Graspability Evaluation        |  <-- Proposed Contribution
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                   Inverse Kinematics                  |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|               Pick-and-Place Execution                |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                 Evaluation & Logging                  |
+-------------------------------------------------------+
```

---

## Detailed Pipeline Component Description

### 1. PyBullet Tabletop Scene
- **Role**: Serves as the primary physics simulation environment.
- **Components**: Includes ground plane, tabletop, robotic arm (KUKA iiwa / Franka Panda), target objects (cube, cylinder, bottle, cup, box), and destination tray.
- **Physics**: Configured with real-world gravity (\(g = -9.81\,\text{m/s}^2\)) and realistic contact friction.

### 2. Virtual Overhead RGB Camera
- **Role**: Captures high-resolution workspace images.
- **Placement**: Fixed overhead eye-in-sky or wrist-mounted virtual camera generating synthetic RGB arrays, depth maps, and segmentation masks.

### 3. Image Preprocessing
- **Role**: Prepares raw camera frames for object detection.
- **Operations**: Includes resizing, normalization, noise reduction (Gaussian blurring), and color space conversions (RGB to HSV).

### 4. Detection Module Options
- **Classical OpenCV Baseline**: HSV color segmentation, thresholding, morphological opening/closing, and contour extraction.
- **AI-Based YOLOv8n Detector**: Lightweight deep-learning model providing bounding box coordinates, class labels, and confidence scores.

---

## Proposed Technical Contribution

> [!IMPORTANT]
> **Key Innovation**: **Confidence-Gated Hybrid Perception and Graspability Evaluation**

The core novelty lies in combining neural object detection with classical computer vision and spatial feasibility constraints to ensure robust pick-and-place execution.

### Confidence-Gated Perception Filtering

```text
                      [ YOLOv8n Detection ]
                                |
               +----------------+----------------+
               |                |                |
               v                v                v
      High Confidence    Medium Confidence   Low Confidence
       (Score >= 0.85)   (0.50 <= Score < 0.85) (Score < 0.50)
               |                |                |
               v                v                v
          Accept &        Apply Colour &     Reject / Re-observe
        Extract Pose     Contour Validation     (Skip Target)
                                |
                        +-------+-------+
                        |               |
                        v               v
                     Passed          Failed
                        |               |
                        v               v
                    Accept &         Reject /
                  Extract Pose     Re-observe
```

1. **High-Confidence Detections (\(\ge 0.85\))**:
   - Directly accepted into pose estimation.
2. **Medium-Confidence Detections (\(0.50 - 0.84\))**:
   - Subjected to HSV color verification and geometric contour consistency checks (e.g., aspect ratio, perimeter-to-area ratio). Detections passing validation are accepted; failing ones are discarded.
3. **Low-Confidence Detections (\(< 0.50\))**:
   - Automatically rejected or flagged for re-observation to prevent false positive picks.

### Reachability and Graspability Evaluation

Before generating motion trajectories, the candidate object location is evaluated for physical execution safety:
- **Kinematic Reachability**: Checks whether the converted 3D world coordinate lies strictly within the workspace envelope of the robotic arm.
- **Gripper Clearance**: Verifies that adjacent distractor objects or table boundaries leave sufficient clearance for gripper jaw placement without collision.
- **Graspability Score**: Ranks candidate objects based on confidence, clearance, and distance to optimal end-effector alignment.

---

## Execution & Control Modules (Planned Future Phase)

- **Pixel-to-World Coordinate Conversion**: Uses camera intrinsic and extrinsic matrix calibration to transform image pixel centroids \((u, v)\) to workspace 3D coordinates \((X_w, Y_w, Z_w)\).
- **Inverse Kinematics (IK)**: Computes target joint angles required to position end-effector above and at target object.
- **Pick-and-Place Execution**: Multi-stage trajectory execution (Approach \(\rightarrow\) Descend \(\rightarrow\) Close Gripper \(\rightarrow\) Lift \(\rightarrow\) Move to Tray \(\rightarrow\) Release \(\rightarrow\) Retract).
- **Evaluation & Logging**: Records success rates, execution time, detection accuracy, and collision events for comparative analysis.
