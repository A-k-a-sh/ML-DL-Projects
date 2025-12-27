# DCGAN - CelebA Face Generation

Deep Convolutional Generative Adversarial Network (DCGAN) implementation for generating realistic face images using the CelebA dataset.

## Overview
- Implements DCGAN architecture for image generation
- Trains on the CelebA (Celebrity Faces) dataset
- Generates new synthetic face images that resemble real celebrities
- Uses adversarial training with Generator and Discriminator networks

## Dataset
- CelebA Dataset: Large-scale celebrity face attributes dataset
- Contains aligned and cropped face images
- Downloaded using Kaggle Hub

## Technologies Used
- PyTorch
- NumPy, Pandas
- Matplotlib
- KaggleHub (for dataset download)

## Model Architecture
- Generator: Creates fake images from random noise
- Discriminator: Distinguishes between real and fake images
- Trained using adversarial loss
