## Notebook quick view - [click here](https://note-book-share.vercel.app/#/https://github.com/A-k-a-sh/ML-DL-Projects/blob/main/Deep%20learning/Object%20detection%20/coco-object-detection-with-pytorch-fastercnn.ipynb) | [click here](https://note-book-share.vercel.app/#/https://github.com/A-k-a-sh/ML-DL-Projects/blob/main/Deep%20learning/Object%20detection%20/coco-object-detection%20(1).ipynb)

# COCO Object Detection with PyTorch

Comprehensive object detection project using PyTorch's Faster R-CNN and other detection models on the COCO (Common Objects in Context) 2017 dataset. This project demonstrates modern object detection techniques, bounding box predictions, and multi-object recognition.

## Project Files

### 1. coco-object-detection (1).ipynb

Initial exploration and custom implementation of COCO object detection with detailed data analysis.

**Key Features:**
- **COCO 2017 Dataset Integration**:
  - Training set: 118,287 images
  - Validation set: 5,000 images
  - Test set: 40,670 images (annotations not public)
  - 80 object categories covering everyday objects
  
- **Annotation Structure**:
  - `instances_train2017.json`: Training annotations
  - `instances_val2017.json`: Validation annotations
  - JSON format with comprehensive metadata
  
- **COCO ID Mapping**:
  - Original COCO category IDs (non-contiguous: 1-90 with gaps)
  - Custom mapping to contiguous indices (0-79) for model training
  - Category dictionary: `{id: name}` mapping
  - Example: Category 1 (person), 18 (dog), 52 (banana), etc.
  
- **Category Filtering**:
  - Filter annotations by specific category (e.g., "hot dog")
  - Extract all images containing a target object
  - Useful for fine-grained detection or specific use cases
  
- **Data Exploration**:
  - Annotation file structure analysis
  - Category distribution statistics
  - Image-annotation relationships
  - Bounding box format: [x, y, width, height]
  
- **Custom Detection Pipeline**:
  - Build dataset from JSON annotations
  - Create custom DataLoader for COCO format
  - Handle variable number of objects per image

### 2. coco-object-detection-with-pytorch-fastercnn.ipynb

Production-ready implementation using PyTorch's pre-trained Faster R-CNN model.

**Key Features:**
- **Faster R-CNN Architecture**:
  - Two-stage detector (Region Proposal + Classification)
  - ResNet-50 or ResNet-101 backbone with FPN (Feature Pyramid Network)
  - RoI (Region of Interest) pooling for multi-scale detection
  - Pre-trained on COCO dataset for transfer learning
  
- **Model Loading**:
  - Load pre-trained weights from torchvision model zoo
  - Fine-tuning options for custom datasets
  - GPU acceleration with CUDA support
  
- **Inference Pipeline**:
  - Single image and batch inference
  - Confidence threshold filtering (e.g., 0.5)
  - Non-Maximum Suppression (NMS) for duplicate removal
  - Multi-object detection in single pass
  
- **Output Format**:
  - Bounding boxes: [xmin, ymin, xmax, ymax]
  - Class labels: Integer IDs mapped to category names
  - Confidence scores: Probabilities for each detection
  
- **Visualization**:
  - Draw bounding boxes on images
  - Annotate with class names and confidence scores
  - Color-coded boxes for different categories
  - Multiple detections per image
  
- **Performance Optimization**:
  - Batch processing for multiple images
  - Efficient tensor operations
  - Model in evaluation mode for faster inference

## COCO Dataset Details

**80 Object Categories:**

*People & Animals:*
- person, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

*Vehicles:*
- bicycle, car, motorcycle, airplane, bus, train, truck, boat

*Outdoor Objects:*
- traffic light, fire hydrant, stop sign, parking meter, bench

*Sports:*
- baseball bat, baseball glove, skateboard, surfboard, tennis racket, skis, snowboard, sports ball, kite, frisbee

*Kitchen & Dining:*
- bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

*Furniture:*
- chair, couch, potted plant, bed, dining table, toilet

*Electronics:*
- tv, laptop, mouse, remote, keyboard, cell phone

*Appliances:*
- microwave, oven, toaster, sink, refrigerator

*Indoor Objects:*
- book, clock, vase, scissors, teddy bear, hair dryer, toothbrush

**Annotation Format:**
```json
{
  "images": [{"id": 123, "file_name": "000000123.jpg", ...}],
  "annotations": [
    {
      "image_id": 123,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 5000,
      "segmentation": [...],
      "iscrowd": 0
    }
  ],
  "categories": [{"id": 1, "name": "person", "supercategory": "person"}]
}
```

## Technologies Used

- **PyTorch**: Deep learning framework
- **torchvision**: 
  - `models.detection.fasterrcnn_resnet50_fpn` (pre-trained Faster R-CNN)
  - Detection utilities and transforms
- **NumPy**: Array operations
- **Pandas**: Data analysis and metrics
- **Matplotlib**: Visualization of detections
- **PIL/Pillow**: Image loading and manipulation
- **JSON**: Parsing COCO annotation files
- **OpenCV (optional)**: Advanced image processing
- **COCO API (optional)**: Official evaluation metrics

## Key Concepts

**Faster R-CNN Architecture:**
1. **Backbone Network**: ResNet-50/101 with FPN for feature extraction
2. **Region Proposal Network (RPN)**: Generates candidate object regions
3. **RoI Pooling**: Extracts fixed-size features from proposals
4. **Classification & Regression Heads**: 
   - Object class prediction
   - Bounding box refinement

**Detection Pipeline:**
1. Input image preprocessing (resize, normalize)
2. Feature extraction through backbone
3. Region proposals generation
4. RoI feature extraction
5. Class prediction and box refinement
6. Post-processing (NMS, thresholding)
7. Output: boxes, labels, scores

**Performance Metrics:**
- **mAP** (mean Average Precision): Primary COCO metric
- **AP@IoU=0.5**: Detection at 50% IoU threshold
- **AP@IoU=0.75**: Detection at 75% IoU threshold  
- **AR** (Average Recall): Completeness of detections
- Per-class AP for category-wise performance

## Applications

- **Autonomous Vehicles**: Detect pedestrians, vehicles, traffic signs
- **Surveillance**: Monitor people and objects in security footage
- **Retail Analytics**: Count customers, track products
- **Sports Analysis**: Track players, balls, and equipment
- **Robotics**: Object recognition for manipulation
- **Content Moderation**: Identify inappropriate objects
- **Inventory Management**: Automated product counting
- **Accessibility**: Assist visually impaired with scene understanding

## Learning Outcomes

1. Understanding modern object detection architectures
2. Working with COCO dataset format and annotations
3. Implementing Faster R-CNN with PyTorch
4. Handling variable-size inputs and outputs
5. Post-processing techniques (NMS, thresholding)
6. Evaluation metrics for object detection
7. Visualizing detection results
8. Transfer learning for detection tasks
9. Handling multiple objects in single images
10. Category ID mapping and management

## Evaluation & Results

**Typical Faster R-CNN Performance:**
- **mAP**: ~37-42% on COCO val2017
- **Inference Speed**: 5-15 FPS on GPU (depending on image size)
- **Strong Categories**: People, vehicles, large animals
- **Challenging Categories**: Small objects, occluded instances

## Future Enhancements

- YOLO or RetinaNet for faster inference
- Instance segmentation with Mask R-CNN
- Keypoint detection for human pose estimation
- 3D object detection
- Video object tracking
- Custom object categories training
- Real-time webcam detection
- Mobile deployment optimization
