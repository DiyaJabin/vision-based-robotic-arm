<div align="center">

# Vision-Based Robotic Arm

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyBullet](https://img.shields.io/badge/Simulation-PyBullet-orange)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Object%20Detection-YOLOv8-purple)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red?logo=pytorch&logoColor=white)

### A Simulation-Based Intelligent Tabletop Pick-and-Place System

</div>

---

## About the Project

This project develops a vision-based robotic arm capable of detecting tabletop objects and performing automated pick-and-place operations.

The system is implemented in PyBullet using a simulated robotic arm and a virtual overhead RGB camera. Images captured from the simulated workspace are processed using OpenCV and YOLOv8 to detect and locate objects.

The detected image coordinates are transformed into robot workspace coordinates. The robotic arm then uses inverse kinematics to move towards the selected object, grasp it and place it inside a designated destination area.

The project follows a simulation-first approach, allowing the complete vision, planning and control pipeline to be tested under repeatable conditions before considering physical hardware implementation.

---

## Main Features

- Robotic-arm simulation using PyBullet
- Virtual overhead RGB camera
- Tabletop object detection using YOLOv8n
- Classical image processing using OpenCV
- Object-centre and orientation estimation
- Pixel-to-world coordinate transformation
- Object reachability checking
- Graspability-based object selection
- Inverse-kinematics-based arm control
- Collision-aware pick-and-place execution
- Performance and failure logging

---

## Proposed Approach

The project compares two perception approaches.

### Baseline Method

The baseline method uses classical computer vision:

- HSV colour segmentation
- Binary masking
- Morphological operations
- Contour detection
- Object-centroid calculation
- Minimum-area rectangle for orientation estimation

### Proposed Method

The proposed method combines deep learning and classical computer vision:

- YOLOv8n object detection
- Confidence-based detection filtering
- Colour and contour validation
- Object pose estimation
- Reachability and gripper-clearance checking
- Graspability scoring
- Inverse kinematics
- Pick-and-place execution

High-confidence YOLO detections are accepted directly. Medium-confidence detections are validated using colour, contour and shape information. Low-confidence detections are rejected or processed again.

---

## System Workflow

```text
Virtual RGB Camera
        |
        v
Image Acquisition
        |
        v
Image Pre-processing
        |
        v
YOLOv8n Object Detection
        |
        v
Colour and Contour Validation
        |
        v
Object Centre and Orientation Estimation
        |
        v
Pixel-to-World Coordinate Transformation
        |
        v
Graspability and Reachability Check
        |
        v
Inverse Kinematics
        |
        v
Robotic Arm Pick-and-Place Execution
        |
        v
Performance Evaluation
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyBullet | Robotic-arm, camera and physics simulation |
| OpenCV | Image processing, contours and coordinate calibration |
| YOLOv8n | Object detection |
| PyTorch | Deep-learning model training and inference |
| NumPy | Numerical operations and coordinate transformations |
| SciPy | Scientific and mathematical utilities |
| Pandas | Experiment logging and result analysis |
| Matplotlib | Performance graphs and visualizations |
| Git and GitHub | Version control and team collaboration |

---

## Repository Structure

```text
vision-based-robotic-arm/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── scripts/
│
├── simulation/
│   ├── assets/
│   ├── robot_models/
│   ├── scene.py
│   ├── camera.py
│   └── environment.py
│
├── perception/
│   ├── baseline_hsv.py
│   ├── yolo_detector.py
│   ├── hybrid_validator.py
│   ├── pose_estimator.py
│   └── calibration.py
│
├── robot/
│   ├── kinematics.py
│   ├── grasp_selector.py
│   ├── planner.py
│   ├── controller.py
│   └── gripper.py
│
├── experiments/
│   ├── run_trials.py
│   ├── compare_methods.py
│   └── results/
│
├── models/
├── tests/
└── docs/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/vision-based-robotic-arm.git
cd vision-based-robotic-arm
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dependencies

```text
pybullet
opencv-python
ultralytics
torch
torchvision
numpy
scipy
pandas
matplotlib
```

---

## Running the Simulation

Run the PyBullet environment:

```bash
python simulation/scene.py
```

Other execution commands will be added as the perception and robot-control modules are implemented.

---

## Future Extensions

- ROS 2 integration
- Gazebo simulation
- MoveIt 2 motion planning
- Physical robotic-arm implementation
- RGB-D camera integration
- 6-DoF grasp estimation
- Sim-to-real transfer
