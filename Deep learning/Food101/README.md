# Food101 Image Classification

A comprehensive deep learning project for classifying food images across 101 different food categories using the Food-101 dataset. This project demonstrates transfer learning, data handling with JSON metadata, and multi-class classification at scale.

## Overview

This notebook implements a complete food image classification system capable of identifying 101 different types of dishes from around the world. The project covers data loading from structured JSON files, model training with transfer learning, and evaluation on a challenging real-world dataset.

### Challenge & Motivation

Food recognition is a complex computer vision task due to:
- High intra-class variation (same dish can look very different)
- Inter-class similarity (different dishes can look similar)
- Varying presentation styles, plating, and lighting
- Cultural and regional variations in food preparation

## Key Features

**Dataset Structure:**
- **Food-101 Dataset**: Large-scale food image dataset
- **101 Food Categories**: Diverse cuisines and dish types
- **Training Images**: 75,750 images (750 per class)
- **Test Images**: 25,250 images (250 per class)
- **Organization**: Structured directory with JSON metadata

**Metadata Management:**
- **train.json**: Dictionary mapping class names to training image paths
  - Example: `{"apple_pie": ["apple_pie/123.jpg", ...], ...}`
- **test.json**: Dictionary mapping class names to test image paths
- **labels.txt**: List of all 101 food category names
- **classes.txt**: Additional class information
- Efficient loading and parsing of JSON metadata

**Food Categories Include:**
- Appetizers: spring rolls, edamame, samosa
- Main Courses: pizza, hamburger, sushi, ramen, tacos
- Desserts: cheesecake, ice cream, donuts, macarons
- Beverages: coffee, smoothies
- Ethnic Cuisines: pad thai, pho, bibimbap, falafel
- And many more...

**Data Pipeline:**
- Custom dataset class for Food-101 JSON structure
- Image loading from organized directory structure
- Preprocessing and augmentation:
  - Resize to consistent dimensions (224×224 for common CNNs)
  - Random crops and flips for augmentation
  - Color jittering for robustness
  - Normalization using ImageNet statistics
- DataLoader with efficient batching and multiprocessing

**Model Architecture:**
- **Transfer Learning Approach**: Leverages pre-trained models
  - ResNet-50/101: Deep residual networks
  - EfficientNet: Scalable and efficient architecture
  - Vision Transformers (ViT): Attention-based models
- **Fine-tuning Strategy**:
  - Freeze early layers (generic features)
  - Train later layers and classification head
  - Gradual unfreezing for better performance
- **Custom Classification Head**: 101-way softmax output

**Training Pipeline:**
- **Loss Function**: Cross-Entropy Loss for 101 classes
- **Optimizer**: Adam or AdamW with weight decay
- **Learning Rate Schedule**: 
  - Warmup phase for stability
  - Cosine annealing or step decay
- **Regularization**:
  - Dropout in classification head
  - Label smoothing
  - Mixup or CutMix augmentation
- **Monitoring**: Train/validation loss and accuracy tracking

**Evaluation:**
- Top-1 accuracy (exact match)
- Top-5 accuracy (correct class in top 5 predictions)
- Per-class precision, recall, F1-scores
- Confusion matrix for 101 classes
- Error analysis on difficult categories
- Visualization of predictions with confidence scores

**Challenges Addressed:**
- **Large Number of Classes**: 101-way classification complexity
- **Class Imbalance**: Ensuring balanced training
- **Data Quality**: Handling noisy real-world images
- **Computational Cost**: Efficient training strategies

## Dataset Details

**Food-101 Specifications:**
- **Total Images**: 101,000 images
- **Resolution**: Variable (typically 512×512), resized for training
- **Format**: RGB color images (JPG)
- **Categories**: 101 food classes, alphabetically ordered
- **Split**: 750 training + 250 test images per class
- **Source**: Real-world food photos with varying quality

**Class Examples:**
```
apple_pie, baby_back_ribs, baklava, beef_carpaccio, beef_tartare,
beet_salad, beignets, bibimbap, bread_pudding, breakfast_burrito,
bruschetta, caesar_salad, cannoli, caprese_salad, carrot_cake,
ceviche, cheese_plate, cheesecake, chicken_curry, chicken_quesadilla,
chicken_wings, chocolate_cake, chocolate_mousse, churros, clam_chowder,
... (and 76 more)
```

## Technologies Used

- **PyTorch**: Deep learning framework
- **torchvision**: 
  - Pre-trained models for transfer learning
  - Transform pipelines
- **NumPy**: Numerical operations
- **Pandas**: Data organization and metrics
- **Matplotlib**: Image visualization and plots
- **JSON**: Metadata parsing and handling
- **PIL/Pillow**: Image loading and manipulation
- **tqdm**: Progress bars for training loops
- **scikit-learn**: Metrics and evaluation tools

## Implementation Highlights

**Custom Dataset Class:**
```python
class Food101Dataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        # Load JSON metadata
        # Map images to labels
        # Handle transforms
```

**Transfer Learning Setup:**
- Load pre-trained ImageNet weights
- Modify final layer for 101 classes
- Selective layer freezing/unfreezing
- Fine-tuning with lower learning rates

**Data Augmentation:**
- Training: Random crops, flips, rotations, color jitter
- Validation/Test: Center crop, resize only
- Normalization with ImageNet mean/std

## Performance Metrics

**Typical Results:**
- **Top-1 Accuracy**: 70-85% (depending on model)
- **Top-5 Accuracy**: 90-95%
- **Training Time**: Several hours on GPU
- **Inference Speed**: Real-time on modern GPUs

**Challenging Classes:**
- Similar-looking dishes (different pasta types)
- Minimally presented foods
- Cultural variations of same dish

## Applications

- **Restaurant Menu Recognition**: Identify dishes from photos
- **Dietary Tracking**: Calorie and nutrition estimation
- **Food Ordering Apps**: Visual search for dishes
- **Recipe Recommendation**: Suggest recipes based on dish photos
- **Food Blogging**: Automatic tagging and categorization
- **Allergen Detection**: Identify potential allergens in dishes

## Learning Outcomes

1. Working with large-scale multi-class classification
2. Parsing and utilizing JSON metadata
3. Implementing custom PyTorch Dataset classes
4. Transfer learning and fine-tuning strategies
5. Handling real-world image data
6. Data augmentation for food images
7. Evaluation of 100+ class models
8. Managing computational resources for large models

## Future Enhancements

- Ingredient detection and listing
- Multi-label classification (multiple dishes in one image)
- Regional cuisine specialization
- Portion size estimation
- Nutritional information prediction
- Cross-dataset generalization
- Mobile deployment for on-device inference
