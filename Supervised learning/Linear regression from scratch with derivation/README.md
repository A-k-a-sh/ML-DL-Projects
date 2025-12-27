# Linear Regression from Scratch with Derivation

Complete implementation of simple linear regression from first principles with full mathematical derivations. This project builds the foundational regression algorithm without using scikit-learn's built-in models, providing deep understanding of how machine learning optimization works.

## Overview

This notebook implements linear regression entirely from scratch using only NumPy for computations. It demonstrates gradient descent optimization, cost function minimization, and the mathematics behind predicting continuous outcomes from input features.

### Why Linear Regression from Scratch?

**Learning Benefits:**
- Understand the mathematics of supervised learning
- Implement gradient descent optimization manually  
- Learn how loss functions guide parameter updates
- Appreciate what sklearn.LinearRegression does internally
- Foundation for understanding neural networks (which are extensions of this)
- Debug ML models by understanding their internals

**Real-world Application:**
Predicting sales based on advertising spend - a common business analytics problem that demonstrates regression fundamentals.

## Key Mathematical Concepts

### 1. Linear Hypothesis

**Model**: y = mx + b (or y = θ₁x + θ₀ in ML notation)
- **y**: Predicted sales (target variable)
- **x**: Radio advertising spend (feature)
- **m (θ₁)**: Slope/weight parameter
- **b (θ₀)**: Intercept/bias parameter

### 2. Cost Function (Mean Squared Error)

**Formula**: J(θ) = (1/2m) Σ(h_θ(x^(i)) - y^(i))²

Where:
- m = number of training examples
- h_θ(x) = θ₁x + θ₀ (hypothesis function)
- Goal: Minimize J(θ) to find best θ₁ and θ₀

**Why MSE?**
- Differentiable (enables gradient descent)
- Convex (single global minimum)
- Penalizes large errors more heavily (squared term)
- Standard for regression problems

### 3. Gradient Descent

**Update Rules**:
- θ₁ := θ₁ - α * ∂J/∂θ₁
- θ₀ := θ₀ - α * ∂J/∂θ₀

**Gradient Calculations**:
- ∂J/∂θ₁ = (1/m) Σ(h_θ(x^(i)) - y^(i)) * x^(i)
- ∂J/∂θ₀ = (1/m) Σ(h_θ(x^(i)) - y^(i))

**Hyperparameters**:
- α (alpha): Learning rate (step size)
  - Too small: Slow convergence
  - Too large: Overshooting, divergence
  - Typical values: 0.001 to 0.1

### 4. Convergence

**When to Stop**:
- Cost change below threshold (e.g., < 0.0001)
- Maximum iterations reached
- Gradient magnitude near zero

## Dataset Details

**Synthetic Advertising Data:**
- **Generated**: Custom dataset simulating tech company ad spending
- **Size**: 50 data points
- **Feature**: Radio advertising spend ($)
  - Range: Congested in $35-45 (realistic constraint)
  - Generated using `np.random.uniform(35.0, 45.0)`
- **Target**: Sales units
  - Calculated as: sales = 0.6 * radio + noise
  - Noise: `np.random.normal(3, 1.5)` for realistic variation
  - Creates intentional correlation with some randomness

**Companies Included**:
50 major tech companies: Amazon, Google, Facebook, Apple, Microsoft, Netflix, Tesla, Twitter, Uber, Airbnb, Spotify, Adobe, Intel, IBM, Oracle, Salesforce, Cisco, Dell, HP, Nvidia, AMD, Qualcomm, Samsung, Sony, LG, Panasonic, Docusign, Zoom, Slack, Atlassian, Shopify, Square, Stripe, Palantir, Snowflake, Databricks, MongoDB, Elastic, Twilio, Okta, Cloudflare, Fastly, PagerDuty, ServiceNow, Workday, Splunk, Zscaler, CrowdStrike, Fortinet, PaloAlto, VMware, RedHat, GitHub, GitLab, DigitalOcean

**Why Synthetic Data?**
- Controlled correlation strength
- Known ground truth (for validation)
- Reproducible results
- Focus on algorithm, not data collection

## Implementation Workflow

### 1. Data Generation
```python
np.random.seed(42)  # Reproducibility

data = []
for i in range(50):
    company = np.random.choice(companies)
    radio = round(np.random.uniform(35.0, 45.0), 1)
    sales = round(0.6 * radio + np.random.normal(3, 1.5), 1)
    data.append([company, radio, sales])

df = pd.DataFrame(data, columns=['Company', 'Radio ($)', 'Sales'])
```

### 2. Hypothesis Function
```python
def hypothesis(x, theta0, theta1):
    return theta0 + theta1 * x
```

### 3. Cost Function Implementation
```python
def compute_cost(x, y, theta0, theta1):
    m = len(y)
    predictions = hypothesis(x, theta0, theta1)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost
```

### 4. Gradient Computation
```python
def compute_gradients(x, y, theta0, theta1):
    m = len(y)
    predictions = hypothesis(x, theta0, theta1)
    
    grad_theta0 = (1/m) * np.sum(predictions - y)
    grad_theta1 = (1/m) * np.sum((predictions - y) * x)
    
    return grad_theta0, grad_theta1
```

### 5. Gradient Descent Loop
```python
def gradient_descent(x, y, theta0, theta1, alpha, iterations):
    cost_history = []
    
    for i in range(iterations):
        grad_theta0, grad_theta1 = compute_gradients(x, y, theta0, theta1)
        
        theta0 = theta0 - alpha * grad_theta0
        theta1 = theta1 - alpha * grad_theta1
        
        cost = compute_cost(x, y, theta0, theta1)
        cost_history.append(cost)
    
    return theta0, theta1, cost_history
```

## Key Visualizations

### 1. Scatter Plot with Regression Line
- **X-axis**: Radio advertising spend ($35-45)
- **Y-axis**: Sales units
- **Points**: Company data (scatter)
- **Line**: Fitted regression line (y = θ₁x + θ₀)
- **Shows**: How well model captures relationship

### 2. Learning Curve
- **X-axis**: Iteration number
- **Y-axis**: Cost (MSE)
- **Pattern**: Decreasing curve showing convergence
- **Interpretation**:
  - Steep initial drop: Large gradients early
  - Plateau: Convergence reached
  - Increasing: Learning rate too high (overshooting)

### 3. Residual Plot
- **X-axis**: Predicted values
- **Y-axis**: Residuals (actual - predicted)
- **Ideal**: Random scatter around zero
- **Patterns indicate**: Model assumptions violated

## Model Evaluation

**Metrics**:
1. **R² Score (Coefficient of Determination)**:
   - R² = 1 - (SS_res / SS_tot)
   - Range: 0 to 1 (1 = perfect fit)
   - Interpretation: Proportion of variance explained

2. **Mean Squared Error (MSE)**:
   - Average squared prediction error
   - Units: (sales units)²

3. **Root Mean Squared Error (RMSE)**:
   - √MSE
   - Same units as target (interpretable)

4. **Mean Absolute Error (MAE)**:
   - Average absolute prediction error
   - Less sensitive to outliers than MSE

## Technologies Used

- **NumPy**: 
  - Array operations: `np.sum()`, `np.mean()`
  - Random number generation: `np.random.uniform()`, `np.random.normal()`
  - Mathematical functions
- **Pandas**: 
  - DataFrame for data organization
  - Easy data inspection with `.head()`
- **Matplotlib**: 
  - Scatter plots for data visualization
  - Line plots for regression line
  - Learning curves

## Learning Outcomes

1. **Mathematical Foundations**:
   - Derive linear regression equations
   - Understand gradient descent mathematically
   - Compute partial derivatives
   - Appreciate convex optimization

2. **Implementation Skills**:
   - Translate equations to code
   - Vectorize operations with NumPy
   - Initialize parameters properly
   - Tune hyperparameters (learning rate, iterations)

3. **Machine Learning Concepts**:
   - Supervised learning workflow
   - Training vs prediction phases
   - Cost function minimization
   - Convergence criteria
   - Model evaluation metrics

4. **Debugging ML Models**:
   - Recognize divergence (increasing cost)
   - Diagnose slow convergence
   - Validate gradient computations
   - Check implementation correctness

## Common Issues & Solutions

### Issue 1: Cost Increasing
- **Cause**: Learning rate too high
- **Solution**: Reduce α (e.g., 0.1 → 0.01)

### Issue 2: Slow Convergence
- **Cause**: Learning rate too small
- **Solution**: Increase α or add feature scaling

### Issue 3: Gradient Becomes NaN
- **Cause**: Numerical overflow
- **Solution**: Normalize features, reduce learning rate

### Issue 4: Local Minimum (Not applicable here)
- Linear regression has convex cost function
- Only one global minimum
- Gradient descent guaranteed to converge

## Extensions & Improvements

### 1. Feature Scaling
```python
x_scaled = (x - x.mean()) / x.std()
```
- Speeds up convergence
- Especially important with multiple features

### 2. Polynomial Regression
```python
x_poly = np.column_stack([x, x**2, x**3])
```
- Capture non-linear relationships
- Still uses linear algebra

### 3. Regularization (Ridge/Lasso)
```python
cost = MSE + λ * ||θ||²  # L2 regularization
```
- Prevent overfitting
- Handle multicollinearity

### 4. Mini-batch Gradient Descent
- Update parameters on subsets
- Balance between stochastic and batch GD
- Faster for large datasets

### 5. Advanced Optimizers
- Momentum: Accelerate convergence
- Adam: Adaptive learning rates
- RMSprop: Per-parameter learning rates

## Comparison with sklearn

**From Scratch**:
```python
theta0, theta1, cost_hist = gradient_descent(x, y, 0, 0, 0.01, 1000)
```

**With sklearn**:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x.reshape(-1,1), y)
theta1, theta0 = model.coef_[0], model.intercept_
```

**Takeaway**: sklearn does the same thing internally (but optimized)!

## Applications

- **Marketing**: ROI of advertising channels
- **Finance**: Price prediction models
- **Economics**: Demand forecasting
- **Real Estate**: Housing price estimation (with more features)
- **Healthcare**: Dosage response relationships

## Next Steps

After mastering simple linear regression:
1. **Multiple Linear Regression**: Multiple features (see Multivariate notebook)
2. **Polynomial Regression**: Non-linear relationships
3. **Logistic Regression**: Classification (see Binary Classification notebook)
4. **Neural Networks**: Stack multiple layers
