# 🚨 CCTV Tunnel Accident Detection Using Deep Learning
## 📌 Overview
Developed a deep learning model using Faster R-CNN and SORT to detect accidents in tunnel CCTV footage within 10 seconds. Trained on 3,000+ COCO-formatted frames labeled via Roboflow, improving emergency response by reducing false positives.

This project aims to detect **accidents in tunnel environments** using **CCTV footage**. It uses a fine-tuned **Faster R-CNN** model for object detection and supports real-time or batch analysis on images and video. The project is implemented in **Python** using **PyTorch** and includes full training, evaluation, and inference workflows.

---

## 🚀 Features

- ✅ Real-time accident detection in tunnels
- ✅ Support for image and video input
- ✅ Visualizations with bounding boxes and confidence scores
- ✅ COCO-style evaluation and training from scratch
- ✅ Highly configurable with GPU support

---

## 🧠 Model Details

- **Backbone**: Faster R-CNN with ResNet-50 + FPN
- **Framework**: PyTorch
- **Classes**: Accident, Fire, Smoke (label-mapped to IDs 1, 2, 5)
- **Input Format**: COCO-style `.json` annotations (exported from Roboflow)

---

## 🖥 System Requirements

| Component       | Minimum                            | Recommended                          |
|----------------|-------------------------------------|--------------------------------------|
| OS             | Windows 10 / Ubuntu 20.04           | Windows 11 / Ubuntu 22.04            |
| GPU            | NVIDIA GTX 1050 Ti (4GB VRAM)       | NVIDIA RTX 3060 / 3070 (8GB+ VRAM)   |
| CPU            | Intel i5 8th Gen / AMD Ryzen 5      | Intel i7 10th Gen+ / Ryzen 7+        |
| RAM            | 8 GB                                | 16 GB or more                        |
| Disk Space     | 5 GB (for dataset + model files)    | 10 GB+                               |
| Python         | 3.8 or later                        | 3.10 (recommended)                   |
| CUDA           | Optional                            | CUDA 11.x+ (for GPU acceleration)    |

> **Note**: For training or video inference, a GPU is strongly recommended for faster processing.
