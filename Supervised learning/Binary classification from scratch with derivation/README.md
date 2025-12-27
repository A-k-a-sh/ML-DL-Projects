# Binary Classification from Scratch with Derivation

Comprehensive implementation of binary classification algorithms built from first principles with complete mathematical derivations. This project demonstrates logistic regression without using scikit-learn's built-in classifiers, providing deep understanding of the underlying mathematics.

## Overview

This notebook implements binary classification from scratch, covering all mathematical concepts including gradient descent, cost functions, and optimization. The project predicts heart disease presence using patient medical data, implementing custom algorithms with NumPy operations.

### Learning Philosophy

**Why From Scratch?**
- Understand the mathematics behind classification
- Implement gradient descent optimization manually
- Learn how loss functions guide learning
- Appreciate what libraries like scikit-learn do under the hood
- Debug models by understanding internals

## Key Features

**Mathematical Derivations:**

1. **Logistic Regression Fundamentals**:
   - Sigmoid function: σ(z) = 1/(1 + e^(-z))
   - Hypothesis: h_θ(x) = σ(θ^T x)
   - Decision boundary derivation
   
2. **Cost Function (Binary Cross-Entropy)**:
   - J(θ) = -(1/m)Σ[y*log(h_θ(x)) + (1-y)*log(1-h_θ(x))]
   - Convex optimization landscape
   - Why cross-entropy for classification
   
3. **Gradient Descent**:
   - Update rule: θ_j := θ_j - α * ∂J(θ)/∂θ_j
   - Gradient calculation for logistic regression
   - Learning rate selection
   - Convergence criteria

**Implementation Details:**
- Custom sigmoid function implementation
- Forward propagation with matrix operations
- Backward propagation for gradient computation
- Iterative optimization loop
- Vectorized operations for efficiency
- No sklearn classifiers used (only preprocessing utilities)

**Dataset Analysis:**
- **Heart Disease Dataset** (303 patients)
- **Features (13 attributes)**:
  - age, sex, cp (chest pain type), trestbps (blood pressure)
  - chol (cholesterol), fbs (fasting blood sugar)
  - restecg (resting ECG), thalach (max heart rate)
  - exang (exercise induced angina), oldpeak, slope
  - ca (number of vessels), thal (thalassemia)
- **Target**: Binary (1=disease, 0=no disease)

**Exploratory Data Analysis:**
- Correlation matrix heatmap
- Feature-target relationships
- Scatter plots: age vs thalach colored by disease status
- Shows clear separation patterns with visualization
- Distribution histograms for all features

**Model Training:**
- Initialize parameters (weights and bias)
- Set hyperparameters (learning rate, iterations)
- Training loop with forward/backward passes
- Track loss convergence
- Plot learning curves

**Evaluation:**
- Accuracy on training and test sets
- Confusion matrix
- Precision, recall, F1-score
- ROC curve and AUC
- Decision boundary visualization
- Compare with random guessing baseline

## Technologies Used

- **NumPy**: Core mathematical operations, matrix calculations
- **Pandas**: Data loading, manipulation, and analysis
- **Matplotlib**: Plotting learning curves, decision boundaries
- **Seaborn**: Statistical visualizations, correlation heatmaps

## Implementation Workflow

1. **Data Loading & Preprocessing**: Load heart disease dataset
2. **Sigmoid Function**: Implement activation function
3. **Cost Function**: Binary cross-entropy loss
4. **Gradient Descent**: Parameter optimization algorithm
5. **Training**: Iterative parameter updates
6. **Evaluation**: Model performance metrics

## Key Visualizations

1. **Age vs Heart Rate by Disease Status**:
   - Red: Patients with disease
   - Blue: Healthy patients
   - Diagonal baseline showing random guess
   - Clear clustering patterns visible
   
2. **Correlation Heatmap**:
   - Annotated matrix showing feature correlations
   - Color-coded: identify strong predictors
   - Helps detect multicollinearity
   
3. **Learning Curves**:
   - Cost vs iteration number
   - Shows convergence behavior
   - Validates learning rate selection

## Learning Outcomes

1. **Mathematical Foundations**: Derive logistic regression from probability theory
2. **Implementation Skills**: Vectorize operations, handle numerical stability
3. **ML Concepts**: Hypothesis functions, loss functions, optimization
4. **Debugging**: Understand and fix gradient descent issues
5. **Evaluation**: Interpret classification metrics properly

## Extensions & Improvements

- Regularization (L1/L2) to prevent overfitting
- Feature engineering (polynomial features, interactions)
- Advanced optimization (Momentum, Adam)
- Multi-class extension (one-vs-rest or softmax)
- Cross-validation for robust evaluation
- Feature scaling for faster convergence
