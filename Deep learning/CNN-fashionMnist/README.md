## Notebook quick view - [click here](https://note-book-share.vercel.app/#/https://github.com/A-k-a-sh/ML-DL-Projects/blob/main/Deep%20learning/CNN-fashionMnist/CNN-fashionMnist.ipynb)


# CNN Fashion-MNIST Classification

A comprehensive computer vision project using Convolutional Neural Networks (CNN) to classify clothing items from the Fashion-MNIST dataset. This project demonstrates end-to-end deep learning workflow from data loading to model evaluation.

## Overview

This notebook implements a complete CNN pipeline for image classification on Fashion-MNIST, covering data preprocessing, model architecture, training, and evaluation with extensive visualizations and explanations.

### Key Features

**Dataset Handling:**
- Uses Fashion-MNIST: 60,000 training + 10,000 test images
- Images are 28x28 grayscale (single channel)
- 10 clothing categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Automatic download and caching using `torchvision.datasets`
- Demonstrates `ToTensor()` transform: converts PIL/numpy (H×W×C, [0,255]) to PyTorch tensor (C×H×W, [0.0,1.0])

**Target Transformation Options:**
- Shows how to use `target_transform` for label manipulation
- One-hot encoding example for multi-class classification
- Label mapping to human-readable class names
- Useful for different loss functions and interpretability

**Data Exploration:**
- Dataset length and structure analysis
- Shape inspection: `[60000, 28, 28]` for training images
- Target distribution across 10 classes using `torch.unique()`
- Visualization of sample images with their labels
- Understanding PyTorch Dataset objects vs NumPy arrays

**CNN Architecture:**
- Custom convolutional neural network design
- Multiple Conv2D layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- Batch normalization and dropout for regularization

**Training Pipeline:**
- DataLoader configuration for batch processing
- Loss function selection (Cross-Entropy Loss)
- Optimizer setup (Adam/SGD)
- Training loop with validation
- Learning rate scheduling
- Model checkpoint saving

**Evaluation & Visualization:**
- Accuracy metrics on test set
- Confusion matrix for class-wise performance
- Sample predictions with true/predicted labels
- Training/validation loss curves
- Misclassification analysis

## Dataset Details

**Fashion-MNIST Specifications:**
- **Size**: 28×28 pixels, grayscale
- **Format**: Single channel (1×28×28 tensors)
- **Train**: 60,000 images
- **Test**: 10,000 images
- **Classes**: 10 (balanced distribution)
- **Pixel Range**: [0.0, 1.0] after ToTensor transformation

**Class Distribution:**
- Each class has ~6,000 training images
- Balanced dataset (equal representation)
- Suitable for benchmarking classification models

## Technologies Used

- **PyTorch**: Deep learning framework for model building and training
- **torchvision**: 
  - `datasets.FashionMNIST` for data loading
  - `transforms.ToTensor` for image preprocessing
  - Pre-trained models (optional transfer learning)
- **NumPy**: Array operations and data manipulation
- **Pandas**: Tabular data analysis and metrics
- **Matplotlib**: Visualization of images, plots, and results
- **Seaborn**: Statistical data visualization
- **sklearn**: Additional metrics and validation tools

## Learning Outcomes

1. **Computer Vision Fundamentals**: Understanding image data in deep learning
2. **CNN Architecture**: Building convolutional layers, pooling, and fully connected layers
3. **Data Preprocessing**: Transforms, normalization, and augmentation
4. **PyTorch Ecosystem**: Working with Datasets, DataLoaders, and transforms
5. **Model Training**: End-to-end training pipeline implementation
6. **Performance Analysis**: Evaluation metrics and visualization techniques

## Reference

- [Learn PyTorch - Computer Vision Tutorial](https://www.learnpytorch.io/03_pytorch_computer_vision/)
- [PyTorch Vision Documentation](https://pytorch.org/vision/)
