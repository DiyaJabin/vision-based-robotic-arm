# Dataset Strategy & Specification

This document details the planned dataset generation strategy, directory layout, label specification, and quality control pipeline for training and evaluating object detection models in the **Vision-Based Robotic Arm** project.

---

## Dataset Generation Overview

To train YOLOv8n effectively for tabletop pick-and-place, synthetic image data will be generated directly within PyBullet. This approach enables rapid generation of perfectly annotated ground truth bounding boxes and segmentations without manual labeling effort.

### Object Classes

The initial dataset targets five primary tabletop object categories:

| Class ID | Class Name | Description | Primary Geometry |
|---|---|---|---|
| `0` | `cube` | Small manipulation block | Box |
| `1` | `cylinder` | Cylindrical peg or container | Cylinder |
| `2` | `bottle` | Beverage container model | Multi-part Cylinder/Cone |
| `3` | `cup` | Open container object | Hollow Cylinder |
| `4` | `box` | Packaging box model | Rectangular Box |

### Domain Randomization Factors

To promote robust object detection and prevent overfitting to static simulation conditions, dataset generation applies domain randomization across scenes:

- **Position Randomization**: Object positions \((X, Y)\) randomized across tabletop workspace coordinates.
- **Rotation Randomization**: Yaw rotation angles randomized between \(0^\circ\) and \(360^\circ\).
- **Lighting Variation**: Light source position, intensity, and ambient color varied per frame.
- **Partial Occlusion**: Overlapping placement of target and distractor objects.
- **Distractor Objects**: Unlabeled background objects (spheres, irregular shapes) introduced to test detection specificity.

---

## Dataset Split & Directory Structure

The dataset uses a standard **70% Train / 15% Validation / 15% Test** split.

### YOLO Directory Layout

```text
data/yolo/
├── dataset.yaml
├── images/
│   ├── train/       # 70% of generated synthetic frames
│   ├── val/         # 15% of generated synthetic frames
│   └── test/        # 15% held-out evaluation frames
└── labels/
    ├── train/       # Corresponding YOLO label txt files
    ├── val/         # Corresponding YOLO label txt files
    └── test/        # Corresponding YOLO label txt files
```

---

## YOLO Annotation Format

Each image is accompanied by a `.txt` label file sharing the same base filename. Each line defines a single bounding box normalized relative to image dimensions \([0.0, 1.0]\):

```text
<class_id> <x_center> <y_center> <width> <height>
```

### Example Annotation (`frame_00042.txt`)

```text
0 0.4521 0.6120 0.0840 0.1120
1 0.7210 0.3850 0.0620 0.1450
3 0.2310 0.7890 0.0910 0.0980
```

Where:
- `class_id`: Integer index corresponding to object class (`0` = cube, `1` = cylinder, etc.).
- `x_center`, `y_center`: Bounding box center coordinates normalized by image width and height.
- `width`, `height`: Bounding box dimensions normalized by image width and height.

### Dataset YAML Specification (`dataset.yaml`)

```yaml
path: ../data/yolo
train: images/train
val: images/val
test: images/test

names:
  0: cube
  1: cylinder
  2: bottle
  3: cup
  4: box
```

---

## Data Quality Checks

Before training, automated validation scripts enforce dataset integrity:

1. **Empty Label Verification**: Ensures images without objects either have valid empty label files or are accounted for as negative samples.
2. **Coordinate Bounding**: Confirms all normalized coordinates fall strictly within \([0.0, 1.0]\).
3. **Resolution Consistency**: Verifies uniform image resolution (e.g., \(640 \times 640\)).
4. **Class Balance Audit**: Monitors class distribution to avoid training skew toward any single object type.

---

## Sim-to-Real Future Extension

For future sim-to-real transfer validation, real-world RGB images captured from an external webcam will be collected under realistic lab lighting conditions. This small test set will evaluate domain adaptation techniques without requiring full re-training on physical data.
