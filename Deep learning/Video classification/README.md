# UCF101 Action Recognition with R(2+1)D CNN

Video classification project that recognizes human actions in videos using the UCF101 dataset and R(2+1)D spatiotemporal CNN architecture.

## Overview
- Classifies human actions in video sequences
- Uses the UCF101 action recognition dataset
- Implements R(2+1)D CNN architecture for spatiotemporal feature learning
- Recognizes 5 different action categories

## Dataset
- UCF101 Dataset: Action recognition dataset with realistic videos
- Contains videos of humans performing various actions
- Organized by action categories
- Videos vary in length, lighting, and background

## Model Architecture
- R(2+1)D CNN: Separates spatial and temporal convolutions
- Pretrained models available in torchvision
- Processes video frames as 3D tensors
- Captures both appearance and motion information

## Technologies Used
- PyTorch
- torchvision (with video models)
- NumPy, Pandas
- Matplotlib, Seaborn
- Custom Dataset and DataLoader for video processing

## Features
- Video data loading and preprocessing
- Frame sampling and augmentation
- Action classification across multiple frames
- Visualization of predictions and results
