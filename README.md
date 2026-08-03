<div align="center">



\# Vision-Based Robotic Arm



!\[Python](https://img.shields.io/badge/Python-3.11-blue?logo=python\&logoColor=white)

!\[PyBullet](https://img.shields.io/badge/Simulation-PyBullet-orange)

!\[OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green?logo=opencv\&logoColor=white)

!\[YOLOv8](https://img.shields.io/badge/Object%20Detection-YOLOv8-purple)

!\[PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red?logo=pytorch\&logoColor=white)



!\[Project Status](https://img.shields.io/badge/Status-DA1%20Completed-brightgreen)

!\[Academic Project](https://img.shields.io/badge/Type-Academic%20Mini%20Project-blueviolet)

!\[Team Size](https://img.shields.io/badge/Team-3%20Members-informational)

!\[License](https://img.shields.io/badge/License-Educational-lightgrey)



\### A Simulation-First Intelligent Tabletop Pick-and-Place System



Built using \*\*PyBullet, OpenCV, YOLOv8, PyTorch and Python\*\*



</div>



\---



\## Project Overview



This project develops a vision-based robotic arm capable of identifying tabletop objects using an RGB camera, estimating their position and orientation, and performing automated pick-and-place operations.



The complete system will initially be implemented in simulation using PyBullet. A virtual overhead camera will capture images of randomly placed objects. The perception module will detect and localize the objects, after which their image coordinates will be converted into robot workspace coordinates.



The robotic arm will then use inverse kinematics and collision-aware motion planning to pick the selected object and place it inside a designated destination bin.



The project follows a simulation-first approach so that the complete perception, planning and control pipeline can be tested repeatedly under controlled and randomized conditions.



This project is being developed as part of the BCSE306L Digital Assignment Mini-Project.



\---



\## Problem Statement



Given a `640 × 480` RGB image from a fixed overhead camera observing up to five randomly positioned tabletop objects, the system shall:



1\. Detect each target object.

2\. Estimate the image-plane centre of the object.

3\. Estimate the in-plane orientation of the object.

4\. Convert the object pose into robot workspace coordinates.

5\. Check whether the object is reachable and graspable.

6\. Generate an appropriate pick-and-place trajectory.

7\. Command a simulated robotic arm to pick the object.

8\. Place the object inside the designated destination bin.

9\. Record the success, latency and failure details.



The system will be evaluated under conditions such as:



\- Lighting variation

\- Object rotation

\- Partial occlusion

\- Similar-looking objects

\- Calibration error

\- Distractor objects

\- Limited labelled data

\- Execution-latency constraints



\---



\## Project Objectives



The main objectives of the project are:



\- Build a complete vision-to-action robotic manipulation pipeline.

\- Detect and localize tabletop objects using computer vision.

\- Compare classical image processing with a deep-learning-based detector.

\- Convert image coordinates into robot workspace coordinates.

\- Control a simulated robotic arm using inverse kinematics.

\- Perform automated pick-and-place operations.

\- Evaluate end-to-end robotic performance instead of only detection accuracy.

\- Study failure cases such as missed detections, unreachable objects and grasp failures.

\- Develop a system that can later be extended to ROS 2, Gazebo or physical hardware.



\---



\## Proposed Approach



The project compares a baseline approach with a proposed hybrid vision system.



\### Baseline Method



The baseline method uses classical computer vision techniques:



\- HSV colour segmentation

\- Binary masking

\- Morphological filtering

\- Contour detection

\- Object-centroid calculation

\- Minimum-area rectangle for orientation

\- Fixed top-down grasping

\- Basic pixel-to-world coordinate mapping



The baseline provides a simple and lightweight method for comparison.



\### Proposed Method



The proposed system uses:



\- YOLOv8n object detection

\- Confidence-based detection filtering

\- HSV colour validation

\- Contour and shape validation

\- Object-centre estimation

\- Object-orientation estimation

\- Pixel-to-world coordinate transformation

\- Graspability scoring

\- Reachability checking

\- Inverse kinematics

\- Collision-aware motion planning

\- Pick-and-place execution

\- Failure detection and retry logic



\---



\## Proposed Novelty



The main contribution of the project is a confidence-gated hybrid perception and grasp-selection method.



The system will not rely only on YOLO detections. Instead, it will combine deep-learning-based detection with classical image-processing techniques.



The decision logic is:



\- High-confidence YOLO detections are accepted directly.

\- Medium-confidence detections are validated using colour, contour and shape information.

\- Low-confidence detections are rejected or trigger another observation.

\- Reachability and gripper-clearance checks are performed before execution.

\- Unreachable or unsafe objects are excluded from selection.



A graspability score will be calculated using factors such as:



\- YOLO detection confidence

\- Contour consistency

\- Object clearance

\- Distance from neighbouring objects

\- Robot reachability

\- Gripper clearance

\- Collision risk



The object with the highest valid graspability score will be selected.



\---



\## System Architecture



```text

Virtual RGB Camera

&#x20;       |

&#x20;       v

Image Acquisition

&#x20;       |

&#x20;       v

Image Pre-processing

&#x20;       |

&#x20;       v

YOLOv8n Object Detection

&#x20;       |

&#x20;       v

Confidence-Gated Hybrid Validation

&#x20;       |

&#x20;       +---- HSV Colour Validation

&#x20;       |

&#x20;       +---- Contour and Shape Validation

&#x20;       |

&#x20;       v

Object Centre and Orientation Estimation

&#x20;       |

&#x20;       v

Pixel-to-World Coordinate Transformation

&#x20;       |

&#x20;       v

Graspability and Reachability Check

&#x20;       |

&#x20;       v

Inverse Kinematics

&#x20;       |

&#x20;       v

Collision-Aware Motion Planning

&#x20;       |

&#x20;       v

Robotic Arm Pick-and-Place Execution

&#x20;       |

&#x20;       v

Performance Evaluation and Failure Logging

