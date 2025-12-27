# DCGAN - CelebA Face Generation

Deep Convolutional Generative Adversarial Network (DCGAN) implementation for generating photorealistic face images using the CelebA (Celebrity Faces Attributes) dataset. This project demonstrates advanced GAN training techniques and image generation.

## Overview

This notebook implements DCGAN architecture to learn the distribution of celebrity faces and generate new, synthetic face images that are visually similar to real celebrity photos. The project showcases adversarial training dynamics and deep learning's generative capabilities.

### What is DCGAN?

DCGAN (Deep Convolutional GAN) is an extension of the original GAN architecture that uses convolutional and transpose-convolutional layers to stabilize training and produce higher quality images. It's particularly effective for image generation tasks.

## Key Features

**Dataset Integration:**
- **CelebA Dataset**: Large-scale celebrity face attributes dataset with 200,000+ images
- Images are aligned and cropped to focus on faces
- Consistent lighting and pose for better training stability
- Downloaded via KaggleHub API for seamless integration
- Uses `img_align_celeba` folder containing preprocessed face images

**GAN Architecture:**

*Generator Network:*
- Input: Random noise vector (latent space, typically 100 dimensions)
- Architecture: Transpose convolutional layers (upsampling)
- Output: 64×64 (or higher) RGB face images
- Uses batch normalization and ReLU activations
- Progressive upsampling from latent vector to full image

*Discriminator Network:*
- Input: Real or generated 64×64 RGB images
- Architecture: Convolutional layers (downsampling)
- Output: Single scalar (real vs fake probability)
- Uses LeakyReLU activations and dropout
- Binary classification task with BCE loss

**Training Process:**
- **Adversarial Training**: Generator and Discriminator trained alternately
- Generator learns to fool the Discriminator
- Discriminator learns to distinguish real from fake
- Nash equilibrium seeking through minimax game
- Careful learning rate balancing to prevent mode collapse

**Implementation Details:**
- Custom data loading and preprocessing pipeline
- Image normalization to [-1, 1] range (tanh output)
- Batch processing for efficient GPU utilization
- Weight initialization strategies (e.g., Xavier/He initialization)
- Gradient clipping to prevent training instability
- Learning rate scheduling for convergence

**Visualization:**
- Generated face samples at different training epochs
- Loss curves for both Generator and Discriminator
- Latent space interpolation (morphing between faces)
- Comparison of real vs generated images
- Training progression visualization

## Dataset Details

**CelebA Specifications:**
- **Images**: 202,599 face images of celebrities
- **Resolution**: Originally 178×218, typically resized to 64×64 or 128×128
- **Format**: RGB color images
- **Preprocessing**: Aligned faces, centered, and cropped
- **Coverage**: Diverse ages, ethnicities, and genders

## Technologies Used

- **PyTorch**: Core deep learning framework for GAN implementation
- **torchvision**: Image transformations and utilities
- **NumPy**: Numerical operations and array handling
- **Pandas**: Data organization and metrics tracking
- **Matplotlib**: Visualization of generated images and training metrics
- **KaggleHub**: Automated dataset downloading and management
- **PIL/Pillow**: Image loading and preprocessing

## Training Challenges & Solutions

1. **Mode Collapse**: When Generator produces limited variety
   - Solution: Feature matching, mini-batch discrimination
   
2. **Training Instability**: Oscillating or diverging losses
   - Solution: Proper learning rates, gradient penalties, spectral normalization

3. **Vanishing Gradients**: Discriminator becomes too strong
   - Solution: One-sided label smoothing, noisy labels

## Applications

- Face generation for creative projects
- Data augmentation for face recognition systems
- Understanding latent space representations
- Style transfer and face editing
- Anomaly detection in face images

## Learning Outcomes

1. Understanding GAN fundamentals and adversarial training
2. Implementing Generator and Discriminator networks
3. Managing training stability in GANs
4. Image generation and quality assessment
5. Working with large-scale image datasets
6. Debugging and improving GAN performance
