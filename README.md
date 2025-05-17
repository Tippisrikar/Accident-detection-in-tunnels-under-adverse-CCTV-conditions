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

