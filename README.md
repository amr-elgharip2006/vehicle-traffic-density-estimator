# Vehicle Traffic Density Estimator

A simple computer vision project that counts vehicles and estimates traffic density from a video using YOLOv8 and OpenCV.

This project is part of the **Computer Vision and Robotics** course/project and is designed to be easy to run and understand.

---

## 1. Project Overview

The goal of this project is to:

- Detect vehicles in a traffic video.
- Count how many vehicles pass through a specific line.
- Estimate the **traffic density** based on the number of vehicles visible in the last *N* frames.
- Visualize all results directly on the video (bounding boxes, labels, counters, and density).

The project uses a pre-trained **YOLOv8n** model, so there is no need to train a model from scratch.

---

## 2. Features

- Vehicle detection using **YOLOv8n**.
- Supports multiple vehicle classes: `car`, `motorbike`, `bus`, and `truck`.
- Simple tracking using YOLO built-in tracking IDs.
- Line-crossing vehicle counting.
- Traffic density estimation using a sliding window over recent frames.
- Real-time visualization with OpenCV.

---

## 3. Tech Stack

- **Programming Language:** Python 3.11 (recommended)
- **Computer Vision:** OpenCV
- **Detection & Tracking:** YOLOv8 (Ultralytics)
- **Numerical Operations:** NumPy

---

## 4. Repository Structure

Recommended structure (following the competition instructions):

```text
Vehicle-Traffic-Density-Estimator/
│
├── README.md
│
├── Documentation/
│   ├── Presentation/        # Slides for the project
│   ├── Book/                # Optional detailed report
│   └── Poster/              # Optional project poster
│
├── Media/
│   ├── project_image_1.png
│   ├── project_image_2.png
│   ├── result_video.mp4     # Demo video of the final result
│   └── team_picture_1.jpg
│
├── Programming/
│   ├── Python/
│   │   └── script_1.py      # Main Python script (vehicle counting + density)
│   └── Other-Languages/
│       └── ...              # Any extra code (if used)
│
└── Any-Other-Sector/
    └── ...                  # For any extra modules or tools
