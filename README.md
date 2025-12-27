# Machine Learning & Deep Learning Projects Portfolio

A comprehensive collection of machine learning and deep learning projects covering supervised learning fundamentals, computer vision, object detection, video classification, generative models, and more. Each project demonstrates end-to-end implementation with real-world datasets.

## 📁 Project Organization

This repository is organized into two main categories:

### 🧠 Deep Learning Projects
Advanced neural network implementations using PyTorch for various domains including computer vision, generative AI, and sequence modeling.

### 📊 Supervised Learning Projects
Classical machine learning implementations including algorithms built from scratch with mathematical derivations, regression, and classification projects.

---

## 🔥 Deep Learning Projects

### 1. Classification with PyTorch
Binary and multiclass classification using neural networks on synthetic datasets.
- **Files**: `binary_classification.ipynb`, `Multiclass_classification.ipynb`
- **Key Concepts**: Neural networks, PyTorch basics, classification fundamentals
- **Datasets**: Circles (binary), Blobs (multiclass)
- **Models**: Custom nn.Module, nn.Sequential architectures

### 2. CNN-FashionMNIST
Convolutional Neural Network for clothing image classification.
- **Dataset**: Fashion-MNIST (60k training, 10k test images)
- **Classes**: 10 clothing categories
- **Architecture**: Custom CNN with conv layers, pooling, batch norm
- **Performance**: High accuracy on standardized benchmark

### 3. DCGAN - CelebA Face Generation
Deep Convolutional Generative Adversarial Network for generating realistic celebrity faces.
- **Dataset**: CelebA (200k+ aligned face images)
- **Architecture**: Generator (noise→image), Discriminator (image→real/fake)
- **Key Concepts**: Adversarial training, GAN dynamics, latent space
- **Output**: Synthetic photorealistic face images

### 4. Emotion Detection
Real-time facial emotion recognition using CNNs.
- **Classes**: 7 emotions (happy, sad, angry, fear, surprise, disgust, neutral)
- **Application**: Real-time webcam emotion detection
- **Features**: Face detection integration, live inference
- **Use Cases**: HCI, customer analytics, mental health monitoring

### 5. Food101 Classification
Large-scale food image classification across 101 categories.
- **Dataset**: 101,000 images (750 train + 250 test per class)
- **Approach**: Transfer learning with pre-trained models
- **Challenges**: High inter-class similarity, intra-class variation
- **Applications**: Restaurant menu recognition, dietary tracking

### 6. Object Detection (COCO)
Multi-object detection using Faster R-CNN on COCO dataset.
- **Files**: Two approaches (custom + pre-trained)
- **Dataset**: COCO 2017 (80 object categories)
- **Architecture**: Faster R-CNN with ResNet-50 + FPN backbone
- **Features**: Bounding boxes, class labels, confidence scores
- **Metrics**: mAP, IoU-based evaluation

### 7. Video Classification (UCF101)
Action recognition in videos using R(2+1)D spatiotemporal CNNs.
- **Dataset**: UCF101 (13k videos, 101 action categories)
- **Architecture**: R(2+1)D (factorized 3D convolutions)
- **Key Concepts**: Temporal modeling, frame sampling, video preprocessing
- **Performance**: State-of-the-art action recognition

### 8. Whisper Subtitle Generation
Automatic speech-to-text transcription and subtitle generation.
- **Model**: OpenAI Whisper (transformer-based ASR)
- **Models**: tiny, base, small, medium, large, turbo
- **Languages**: 99+ languages supported
- **Platforms**: Google Colab, Kaggle
- **Output Formats**: SRT, VTT, TXT, JSON, TSV

---

## 📈 Supervised Learning Projects

### 1. Binary Classification from Scratch
Logistic regression implemented from first principles with complete mathematical derivations.
- **Dataset**: Heart disease (303 patients, 13 features)
- **Implementation**: Custom gradient descent, sigmoid, cost function
- **No Libraries**: Built without sklearn classifiers
- **Learning**: Deep understanding of classification mathematics

### 2. Bulldozer Price Prediction
Regression model predicting heavy equipment auction prices.
- **Dataset**: 412k+ auction records
- **Challenge**: Temporal features, missing data, high cardinality
- **Features**: Date extraction, equipment specs, usage history
- **Models**: Random Forest, Gradient Boosting
- **Metrics**: RMSLE (Kaggle competition metric)

### 3. Heart Disease Prediction
Comprehensive multi-notebook exploration of classification algorithms.
- **Files**: setup + notebooks 1-8 (systematic experiments)
- **Dataset**: Heart disease (13 clinical features)
- **Models**: Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting
- **Approach**: Comparative analysis, hyperparameter tuning
- **Best Performance**: 85-91% accuracy

### 4. Linear Regression from Scratch
Simple linear regression implemented from first principles.
- **Dataset**: Synthetic advertising data (radio spend → sales)
- **Implementation**: Custom gradient descent, no sklearn
- **Derivations**: Cost function, gradients, optimization
- **Visualization**: Scatter plots, regression lines, learning curves

### 5. Movie Recommendation System
Content-based recommendation engine using TMDB dataset.
- **Dataset**: TMDB 5000 movies + credits
- **Approach**: Content-based filtering
- **Features**: Genres, cast, crew, movie metadata
- **Output**: Similar movie recommendations

### 6. Multivariate Regression from Scratch
Multiple linear regression with full mathematical implementation.
- **Dataset**: Synthetic advertising (TV + Radio + News → Sales)
- **Features**: Multiple input variables (3D problem)
- **Implementation**: Matrix operations, vectorized gradient descent
- **Concepts**: Feature scaling, multi-dimensional optimization

---

## 🛠️ Technologies & Frameworks

### Deep Learning
- **PyTorch**: Primary deep learning framework
- **torchvision**: Pre-trained models, datasets, transforms
- **Pillow/OpenCV**: Image processing
- **FFmpeg**: Video/audio processing

### Machine Learning
- **scikit-learn**: ML algorithms, preprocessing, metrics
- **XGBoost/LightGBM**: Gradient boosting frameworks

### Data Science
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Visualization

### Development Tools
- **Jupyter**: Interactive notebooks
- **KaggleHub**: Dataset management
- **Google Colab**: Cloud GPU access

---

## 📊 Key Concepts Covered

### Deep Learning
- Convolutional Neural Networks (CNNs)
- Generative Adversarial Networks (GANs)
- Transfer Learning
- Video Classification
- Object Detection (Faster R-CNN)
- Automatic Speech Recognition (ASR)
- Sequence Modeling

### Machine Learning
- Linear and Logistic Regression
- Decision Trees and Random Forests
- Support Vector Machines
- K-Nearest Neighbors
- Gradient Boosting
- Feature Engineering
- Hyperparameter Tuning
- Cross-Validation

### Mathematics
- Gradient Descent Optimization
- Backpropagation
- Cost Functions (MSE, Cross-Entropy)
- Regularization (L1/L2)
- Probability and Statistics

### Data Processing
- Missing Value Imputation
- Feature Scaling and Normalization
- Categorical Encoding
- Time-Series Feature Extraction
- Image Augmentation
- Video Frame Sampling

---

## 🎯 Learning Objectives

1. **Fundamentals**: Build ML/DL algorithms from scratch
2. **Theory**: Understand mathematical foundations
3. **Practice**: Work with real-world datasets
4. **Engineering**: Handle messy data, large-scale processing
5. **Evaluation**: Proper model assessment and metrics
6. **Deployment**: Considerations for production systems

---

## 🚀 Getting Started

### Prerequisites
```bash
# Create virtual environment
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch torchvision
pip install jupyter notebook
```

### Running Notebooks
```bash
# Navigate to project folder
cd "Deep learning/Classification with pytorch"

# Start Jupyter
jupyter notebook

# Open and run desired notebook
```

---

## 📚 Datasets

### Publicly Available
- **Fashion-MNIST**: torchvision.datasets
- **COCO 2017**: Official COCO website
- **Heart Disease**: UCI ML Repository / Kaggle
- **UCF101**: Official UCF website
- **CelebA**: Kaggle Hub
- **Food-101**: Kaggle
- **TMDB Movies**: Kaggle

### Custom Generated
- Circles, Blobs (sklearn.datasets)
- Advertising data (synthetic)

---

## 🔬 Project Highlights

### Most Complex
- **Video Classification**: Temporal modeling, 3D convolutions
- **DCGAN**: Adversarial training dynamics
- **Object Detection**: Multi-object localization

### Best for Learning
- **Binary Classification from Scratch**: Mathematical foundations
- **Heart Disease Prediction**: Systematic model comparison
- **CNN-FashionMNIST**: Computer vision introduction

### Real-World Impact
- **Emotion Detection**: HCI, accessibility
- **Heart Disease Prediction**: Healthcare applications
- **Bulldozer Pricing**: Business analytics

---

## 📈 Future Enhancements

- Add model interpretability (SHAP, LIME)
- Implement attention mechanisms
- Explore transformer architectures
- Add deployment scripts (FastAPI, Docker)
- Create web interfaces for demos
- Add unit tests and CI/CD
- Benchmark on additional datasets

---

## 📝 Notes

- Each project folder contains its own detailed README
- Code includes extensive comments and markdown explanations
- Focus on understanding over just achieving metrics
- Many projects include "from scratch" implementations
- Balanced between theory and practical application

---

## 👨‍💻 Author

Machine Learning & Deep Learning Portfolio
*Demonstrating end-to-end ML/DL project implementation*

---

## 📄 License

Educational and portfolio purposes. Datasets have their own licenses - please check individual dataset sources for usage rights.

---

## 🙏 Acknowledgments

- PyTorch tutorials and documentation
- Kaggle community and datasets
- UCI Machine Learning Repository
- OpenAI (Whisper model)
- Various open-source contributors