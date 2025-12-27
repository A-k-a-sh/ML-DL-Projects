# Facial Emotion Detection (Real-time)

Deep learning project for detecting and classifying facial emotions in real-time using Convolutional Neural Networks. This system can identify human emotions from facial expressions, enabling applications in human-computer interaction, customer sentiment analysis, and mental health monitoring.

## Overview

This notebook implements an end-to-end CNN-based emotion recognition system trained on the Facial Emotion Recognition dataset. The model can classify facial expressions into multiple emotion categories and is designed for real-time inference applications.

### Emotion Categories

The model typically recognizes 7 basic emotions:
- **Happy**: Smiling, joyful expressions
- **Sad**: Downturned mouth, drooping eyes
- **Angry**: Furrowed brows, tense jaw
- **Fear**: Wide eyes, raised eyebrows
- **Surprise**: Open mouth, raised eyebrows
- **Disgust**: Wrinkled nose, raised upper lip
- **Neutral**: Relaxed, expressionless face

## Key Features

**Dataset Handling:**
- **Facial Emotion Recognition Dataset**: Curated collection of facial expression images
- Downloaded via KaggleHub for seamless integration
- Images organized by emotion categories using `ImageFolder`
- Automatic label extraction from directory structure
- Train/validation/test splits for proper evaluation

**Data Preprocessing:**
- Image resizing to consistent dimensions (typically 48×48 or 64×64)
- Grayscale or RGB format depending on model requirements
- Normalization to [0, 1] or [-1, 1] range
- Data augmentation for improved generalization:
  - Random horizontal flips
  - Random rotations
  - Random crops
  - Brightness/contrast adjustments

**Model Architecture:**
- **CNN Design**: Multiple convolutional layers for facial feature extraction
- **Feature Learning**: Learns hierarchical representations
  - Early layers: Edges and simple patterns
  - Middle layers: Facial components (eyes, nose, mouth)
  - Deep layers: Complex emotion-related features
- **Regularization**: Dropout and batch normalization
- **Classification Head**: Fully connected layers with softmax output

**Training Pipeline:**
- **Device Management**: Automatic GPU detection and utilization (CUDA support)
- **Data Loading**: 
  - `ImageFolder` for organized dataset structure
  - `random_split` for creating train/val splits
  - `DataLoader` with batch processing and shuffling
- **Loss Function**: Cross-Entropy Loss for multi-class classification
- **Optimizer**: Adam or SGD with momentum
- **Learning Rate Scheduling**: ReduceLROnPlateau or StepLR
- **Early Stopping**: Prevents overfitting by monitoring validation loss

**Evaluation Metrics:**
- Overall accuracy across all emotions
- Per-class precision, recall, and F1-scores
- Confusion matrix for detailed class-wise analysis
- ROC curves and AUC for multi-class evaluation
- Misclassification analysis

**Real-time Inference:**
- Webcam integration for live emotion detection
- Face detection using Haar Cascades or MTCNN
- Real-time preprocessing and prediction
- Confidence score display
- Bounding box and emotion label overlay

**Visualization:**
- Sample images with emotion labels
- Training/validation accuracy and loss curves
- Confusion matrix heatmap
- Misclassified examples for error analysis
- Grad-CAM for model interpretability (attention maps)

## Dataset Details

**Specifications:**
- **Format**: Grayscale or RGB images of faces
- **Resolution**: Varies (typically 48×48 or 224×224 after preprocessing)
- **Classes**: 7 emotion categories
- **Size**: Thousands of labeled face images
- **Source**: Kaggle Facial Emotion Recognition dataset
- **Splits**: Pre-divided or custom train/val/test splits

**Data Organization:**
```
facial_emotion_dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    └── [same structure]
```

## Technologies Used

- **PyTorch**: Deep learning framework for model development
- **torchvision**: 
  - `ImageFolder` for dataset loading
  - Transforms for preprocessing
  - Pre-trained models (optional transfer learning)
- **torch.utils.data**: DataLoader for efficient batching
- **NumPy**: Numerical computations
- **Pandas**: Data analysis and metrics
- **Matplotlib & Seaborn**: Visualization and plotting
- **OpenCV (cv2)**: Real-time video capture and face detection
- **KaggleHub**: Dataset downloading
- **PIL/Pillow**: Image loading and manipulation

## Real-time Application Setup

**Requirements for Live Detection:**
1. Trained emotion classification model
2. Face detection model (Haar Cascade/MTCNN/RetinaFace)
3. Webcam or video input
4. Preprocessing pipeline matching training

**Inference Pipeline:**
1. Capture frame from webcam
2. Detect face(s) in the frame
3. Crop and preprocess face region
4. Pass through emotion CNN
5. Display prediction with confidence
6. Repeat for next frame

## Applications

- **Customer Service**: Analyze customer satisfaction in real-time
- **Mental Health**: Monitor emotional states for therapy
- **Education**: Assess student engagement and understanding
- **Human-Computer Interaction**: Adaptive interfaces based on user emotions
- **Security**: Detect suspicious behavior or distress
- **Entertainment**: Emotion-based content recommendations
- **Market Research**: Gauge reactions to products or advertisements

## Performance Considerations

- **Inference Speed**: Optimized for real-time processing (30+ FPS)
- **Model Size**: Balanced between accuracy and deployment constraints
- **Robustness**: Handles varying lighting, angles, and occlusions
- **Generalization**: Performs well across different demographics

## Learning Outcomes

1. CNN architecture design for facial recognition
2. Handling image classification with multiple classes
3. Data augmentation techniques for facial images
4. Transfer learning from pre-trained models
5. Real-time inference and video processing
6. Model deployment considerations
7. Evaluation of emotion recognition systems
8. Handling imbalanced datasets (if applicable)

## Future Enhancements

- Multi-face detection and tracking
- Temporal modeling with RNNs/LSTMs for video sequences
- Fine-grained emotion recognition (micro-expressions)
- Cross-dataset generalization
- Mobile deployment (TFLite, ONNX)
- Attention mechanisms for focus on facial regions
