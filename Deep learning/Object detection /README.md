# COCO Object Detection with PyTorch

Object detection project using PyTorch's Faster R-CNN model on the COCO (Common Objects in Context) dataset.

## Overview
- Implements object detection using Faster R-CNN architecture
- Trained on the COCO 2017 dataset
- Detects and localizes multiple objects in images
- Supports 80+ object categories from the COCO dataset

## Dataset
- COCO 2017 Dataset
- Contains images with multiple object instances
- Includes bounding box annotations
- 80+ object categories (people, animals, vehicles, household items, etc.)
- JSON format annotations with category mappings

## Technologies Used
- PyTorch
- torchvision (Faster R-CNN)
- NumPy, Pandas
- JSON for annotation parsing

## Features
- Multi-object detection in single images
- Bounding box predictions
- Category ID to name mapping
- Can filter detections by specific categories (e.g., "hot dog")
- Pre-trained model support from PyTorch model zoo
