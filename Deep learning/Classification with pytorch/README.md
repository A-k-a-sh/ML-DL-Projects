# Binary and Multiclass Classification with PyTorch

This project contains two comprehensive classification implementations using PyTorch, demonstrating both binary and multiclass classification approaches from scratch.

## Project Files

### 1. binary_classification.ipynb

A complete implementation of binary classification using PyTorch on the circles dataset.

**Key Features:**
- **Dataset**: Uses scikit-learn's `make_circles` to generate 1000 samples with controlled noise (0.03)
- **Problem**: Classifies data points into two concentric circular patterns
- **Model Architecture**: 
  - Custom neural network with one hidden layer (2 → 5 → 1 neurons)
  - Implements both `nn.Module` subclassing and `nn.Sequential` approaches
  - Explains tradeoffs between different model building methods
- **Data Pipeline**:
  - Converts NumPy arrays to PyTorch tensors with proper dtype handling (float32)
  - 80-20 train-test split with reproducible random state
  - GPU acceleration support with automatic device detection
- **Visualization**:
  - Scatter plots showing disease vs no-disease patterns
  - Age vs maximum heart rate (thalach) relationships
  - Decision boundary visualization
  - Model prediction comparisons

**Learning Objectives:**
- Understanding neural network architecture for binary classification
- Converting between NumPy and PyTorch data types
- Implementing custom models with `nn.Module`
- Using `nn.Sequential` for simpler architectures
- GPU/CPU device management in PyTorch

### 2. Multiclass_classification.ipynb

Extends binary classification concepts to handle multiple classes simultaneously.

**Key Features:**
- **Dataset**: Uses scikit-learn's `make_blobs` to generate synthetic multiclass data
  - 1000 samples with 2 features
  - 4 distinct classes (clusters)
  - Cluster standard deviation of 1.7 for realistic separation
- **Problem**: Classifies data points into 4 different categories
- **Model Architecture**: Custom neural network designed for multiclass output
- **Approach**: Demonstrates softmax activation and cross-entropy loss for multiclass problems

**Learning Objectives:**
- Transitioning from binary to multiclass classification
- Understanding softmax and one-hot encoding
- Handling multiple output classes in neural networks
- Visualizing multi-class decision boundaries

## Technologies Used
- **PyTorch**: Core deep learning framework
- **scikit-learn**: Dataset generation and preprocessing
- **NumPy**: Numerical operations and array handling
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Comprehensive data visualization

## Common Concepts Covered
- Neural network fundamentals
- Forward propagation
- Loss functions (Binary Cross-Entropy, Cross-Entropy)
- Model training and evaluation
- Train-test splitting
- GPU acceleration
- Model state management

## Reference
Based on [Learn PyTorch - Classification Tutorial](https://www.learnpytorch.io/02_pytorch_classification/)
