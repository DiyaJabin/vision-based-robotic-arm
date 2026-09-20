<div align="center">

# Vision-Based Robotic Arm for Intelligent Tabletop Pick-and-Place

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyBullet](https://img.shields.io/badge/Simulation-PyBullet-orange)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Object%20Detection-YOLOv8n-purple)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

### A Simulation-First Approach to Robotic Perception, Pose Estimation, and Automated Grasping

</div>

---

## Project Description

This project develops an intelligent vision-guided robotic arm system designed for tabletop pick-and-place operations. Operating in a physics-based PyBullet simulation environment, the system utilizes a virtual overhead RGB camera to capture workspace scenes, identify objects using computer vision and deep learning, compute 3D world coordinates, evaluate graspability, and execute pick-and-place tasks via inverse kinematics.

---

## Problem Statement

Automated robotic pick-and-place systems face significant challenges when operating under object visual variations, partial occlusions, variable lighting, and uncertain grasp conditions. Standard computer vision techniques often struggle with complex scenes, while deep-learning detectors alone may produce false positives or lack precision for physical grasp execution. This project addresses these challenges by developing a robust perception pipeline combined with spatial feasibility evaluation.

---

## Proposed Simulation-First Approach

A simulation-first strategy using PyBullet is adopted to enable rapid algorithm iteration, safe visual perception testing, and repeatable kinematics validation prior to potential physical hardware deployment.

### Perception Framework & Scope

- **Simulation Platform**: **PyBullet** serves as the primary physics simulation engine and workspace environment.
- **Classical Vision Baseline**: **OpenCV** provides classical HSV color segmentation and contour detection baselines.
- **AI-Based Detection**: **YOLOv8n** will later provide deep-learning object detection for tabletop items.
- **Proposed Core Innovation**: The proposed method will combine YOLO detection confidence scores with classical color and contour validation (**Confidence-Gated Hybrid Perception**) to ensure high detection accuracy and safe grasp selection.
- **Implemented Core Interfaces**: The repository now includes YOLO result adaptation, confidence-gated validation, image pose estimation, planar homography, measured KUKA IK/control interfaces, conservative clearance and graspability ranking, a simulation-only constraint gripper, destination mapping, and a bounded pick-and-place loop.
- **Runtime verification status**: Core perception and planning contracts have dependency-independent tests. PyBullet-dependent motion, constraint grasping, collision checks, camera rendering, and complete pick-and-place execution require verification in the dedicated Windows environment.
- **Development Scope Note**: Core interfaces and a bounded simulation pick-and-place loop are implemented. YOLO training, runtime demonstration, and experimental evaluation remain future work.

---

## Main Project Objectives

1. Develop a high-fidelity PyBullet simulation environment featuring a tabletop, robotic arm, target objects, and destination tray.
2. Establish a synthetic image dataset generation strategy and YOLO-format annotation specification.
3. Build a classical OpenCV vision baseline for object centroid and orientation estimation.
4. Integrate YOLOv8n object detection with a confidence-gated hybrid validation filter.
5. Perform pixel-to-world coordinate transformation using virtual camera calibration.
6. Evaluate reachability and gripper clearance before attempting grasp motion planning.
7. Execute inverse-kinematics-driven robotic pick-and-place trajectories in PyBullet.

---

## Planned System Workflow

```text
PyBullet Tabletop Scene
        |
        v
Virtual Overhead RGB Camera
        |
        v
Image Preprocessing
        |
        v
OpenCV Baseline OR YOLOv8n Detection
        |
        v
Confidence-Gated Colour & Contour Validation  (Proposed Contribution)
        |
        v
Object Centre & Orientation Estimation
        |
        v
Pixel-to-World Coordinate Conversion
        |
        v
Reachability & Graspability Evaluation        (Proposed Contribution)
        |
        v
Inverse Kinematics
        |
        v
Pick-and-Place Execution
        |
        v
Evaluation & Logging
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Primary development programming language |
| **PyBullet** | Physics simulation engine, tabletop scene, and robot arm topology |
| **OpenCV** | Classical image processing, HSV segmentation, and contour analysis |
| **YOLOv8n** | Lightweight deep-learning object detection model |
| **PyTorch** | Deep-learning framework for model inference |
| **NumPy & SciPy** | Numerical matrix transformations and spatial calculations |
| **Pandas & Matplotlib** | Experiment logging, result aggregation, and plotting |
| **PyYAML** | Dataset and model configuration management |

---

## Repository Structure

```text
vision-based-robotic-arm/
├── README.md
├── Architecture.md
├── DATASET.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── core/
│   ├── __init__.py
│   ├── config.py
│   └── contracts.py
│
├── data/
│   ├── sample/
│   │   └── .gitkeep
│   └── scripts/
│       └── load_dataset.py
│
├── docs/
│   └── images/
│       └── .gitkeep
│
├── simulation/
│   ├── __init__.py
│   ├── scene.py
│   ├── camera.py
│   └── run_pick_and_place.py
│
├── perception/
│   ├── __init__.py
│   ├── yolo_detector.py
│   ├── hybrid_validator.py
│   ├── pose_estimator.py
│   └── calibration.py
│
├── robot/
│   ├── __init__.py
│   ├── kinematics.py
│   ├── controller.py
│   ├── gripper.py
│   ├── grasp_selector.py
│   ├── destinations.py
│   └── pick_and_place.py
│
├── experiments/
│   └── __init__.py
│
└── tests/
    └── __init__.py
```

---

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/vision-based-robotic-arm.git
   cd vision-based-robotic-arm
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     source .venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the DA1 Starter Code

### 1. Dataset Loading Utility

Scan and validate sample dataset directory:

```bash
python data/scripts/load_dataset.py --data-dir data/sample
```

### 2. PyBullet Simulation Feasibility Scene

Launch the interactive PyBullet tabletop simulation starter scene:

```bash
python simulation/scene.py
```

---

## Running Experiments

Use the evaluation scripts to log randomized trial data, compare methods, and plot saved results when the PyBullet runtime and model dependencies are available.

```bash
python experiments/run_trials.py --trials 10
python experiments/compare_methods.py
python experiments/plot_results.py
```

Evaluation scripts depend on runtime availability. If the PyBullet scene, YOLO model, or required runtime dependencies are not available, methods are skipped instead of fabricated, and the script reports `SKIPPED / NOT AVAILABLE`.

---

## Future Extensions

- Real-world webcam integration for sim-to-real transfer evaluation
- 6-DoF pose estimation using depth cameras (RGB-D)
- MoveIt 2 and ROS 2 middleware integration
- Gazebo simulation comparison
- Dynamic obstacle avoidance during trajectory planning
