# AI-316 Lab 03 — Computer Vision & AI Pipeline

## Course

**AI Project Design and Development (AI-316)**
Air University Islamabad
Faculty of Computing & AI

## Lab Objective

This lab focuses on practical computer vision techniques, image preprocessing, edge detection, color-space analysis, YOLO object detection, and real-time AI-based event detection.

## Tasks Completed

* Resized and normalized traffic images using OpenCV.
* Explored HSV and LAB color spaces for industrial defect detection.
* Applied thresholding and masking techniques.
* Compared Gaussian and median filtering.
* Applied Sobel and Canny edge detection.
* Compared YOLO nano and medium models.
* Performed object detection and measured inference latency.
* Counted objects based on class and confidence.
* Implemented region-of-interest intrusion detection.
* Implemented real-time webcam object detection.
* Added confidence-based event triggering.
* Saved alert snapshots and event logs.

## Computer Vision Pipeline

```text
Input Image / Webcam
        ↓
Image Preprocessing
        ↓
YOLO Object Detection
        ↓
Confidence Filtering
        ↓
Object / Event Analysis
        ↓
Visualization & Alerts
```

## Technologies

* Python
* OpenCV
* NumPy
* YOLO / Ultralytics
* Matplotlib
* Webcam
* Git

## Sample Inputs

```text
traffic.jpg
product.jpg
scan.jpg
street.jpg
shelf.jpg
people.jpg
```

## Key Features

### Image Processing

* Resizing
* Normalization
* Color-space conversion
* Thresholding
* Masking
* Blurring
* Edge detection

### Object Detection

* YOLO inference
* Bounding boxes
* Class identification
* Confidence scores
* Object counting

### Real-Time Detection

* Webcam-based inference
* FPS measurement
* Region-of-interest monitoring
* Intrusion alerts
* Timestamped snapshots
* Event logging

## Outcome

A practical computer vision pipeline was implemented covering both traditional image processing techniques and modern YOLO-based object detection.
