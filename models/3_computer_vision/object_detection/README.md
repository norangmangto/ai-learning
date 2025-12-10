# Object Detection

Localize and classify objects in images using bounding boxes.

## 📋 Overview

**Task:** Image → Bounding boxes + class labels
**Output:** Boxes (x, y, w, h) + confidence + class
**Baseline:** YOLOv8, Faster R-CNN
**Speed:** Real-time (30+ fps) with modern GPUs

## 🎯 Problem Formulation

```
Input: Image

Output:
[box1] → Person (0.95 confidence)
[box2] → Car (0.92 confidence)
[box3] → Dog (0.87 confidence)
...

Each detection includes:
- Bounding box (x_min, y_min, x_max, y_max)
- Confidence score (0-1)
- Class label
```

## 🏗️ Architecture Types

### 1. Two-Stage Detectors (Accurate but Slower)

#### Faster R-CNN
```
Image
  ↓
Backbone (ResNet, VGG) → Feature maps
  ↓
Region Proposal Network (RPN) → ~2000 proposals
  ↓
Select top proposals (NMS)
  ↓
RoI pooling → Fixed size features
  ↓
Classification + Bounding box regression
  ↓
Detections
```

**Pros:** High accuracy (90%+ mAP)
**Cons:** Slow (5-15 fps), complex

### 2. Single-Stage Detectors (Fast)

#### YOLO (You Only Look Once)
```
Image
  ↓
Single forward pass
  ↓
Divide image into grid (7×7, 13×13, etc.)
  ↓
Each cell predicts:
  - Class probabilities
  - Bounding boxes
  - Confidence scores
  ↓
NMS to remove duplicates
  ↓
Detections
```

**Pros:** Fast (30-100+ fps), simple
**Cons:** Lower accuracy for small objects

#### SSD (Single Shot Detector)
```
Similar to YOLO but:
- Multi-scale feature maps
- Better for small objects
```

### 3. Anchor-Free Detectors (Modern)

#### CenterNet, FCOS
```
Detect objects by their center
No need for predefined anchor boxes
More flexible, better for crowded scenes
```

## 📊 Popular Models

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| YOLOv8n | 100 fps | 70% mAP | 6MB |
| YOLOv8s | 60 fps | 75% mAP | 22MB |
| YOLOv8m | 30 fps | 80% mAP | 50MB |
| Faster R-CNN | 5 fps | 90% mAP | 200MB |
| EfficientDet-D3 | 30 fps | 85% mAP | 100MB |

## 🚀 Quick Start: YOLOv8

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8m.pt')  # Small, medium, large available

# Detect on image
results = model.predict(source='image.jpg', conf=0.5)

# Parse results
for result in results:
    boxes = result.boxes  # Bounding boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Coordinates
        conf = box.conf[0]  # Confidence
        cls = box.cls[0]  # Class

        print(f"Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"Confidence: {conf:.2f}")
        print(f"Class: {int(cls)}")

# Video inference
results = model.predict(source='video.mp4', conf=0.5)

# Real-time webcam
results = model.predict(source=0, conf=0.5)  # source=0 = webcam
```

## 🚀 Quick Start: Faster R-CNN

```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load pretrained
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Prepare image
transform = transforms.ToTensor()
image = Image.open('image.jpg')
image_tensor = transform(image).unsqueeze(0)

# Detect
with torch.no_grad():
    predictions = model(image_tensor)

# Parse results
for i in range(len(predictions[0]['boxes'])):
    box = predictions[0]['boxes'][i].numpy()
    score = predictions[0]['scores'][i].item()
    label = predictions[0]['labels'][i].item()

    print(f"Box: {box}, Score: {score:.3f}, Label: {label}")
```

## 📈 Evaluation Metrics

### Intersection over Union (IoU)
```
IoU = Intersection Area / Union Area

Perfect match: IoU = 1.0
No overlap: IoU = 0.0
Good match: IoU > 0.5

Used for:
- Determining if detection is "correct"
- Threshold typically 0.5 or 0.75 for evaluation
```

### Average Precision (AP)
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

AP = Area under precision-recall curve

mAP (mean AP) = Average across all classes
COCO mAP = Standard benchmark
```

## 💡 Challenges in Object Detection

### 1. Small Objects
```
Problem: Model misses small objects
Causes: Limited resolution, low context

Solutions:
- Use high-resolution input
- Multi-scale feature pyramids
- Larger model (YOLOv8l instead of YOLOv8s)
```

### 2. Crowded Scenes
```
Problem: Objects close together, overlapping
Causes: Confusion between nearby objects

Solutions:
- Anchor-free models (FCOS, CenterNet)
- Better NMS (soft-NMS, class-aware NMS)
- Data augmentation (mosaic, mixup)
```

### 3. Class Imbalance
```
Problem: Some classes rarer than others
Example: 90% cars, 5% pedestrians, 5% cyclists

Solutions:
- Weighted loss functions
- Oversample rare classes
- Hard negative mining
```

### 4. Speed vs Accuracy
```
Trade-off:
Faster model → Lower accuracy
Accurate model → Slower inference

Solution: Choose based on application
- Real-time: YOLOv8n or YOLOv8s
- Higher accuracy: YOLOv8l or Faster R-CNN
```

## 🎯 NMS (Non-Maximum Suppression)

```
Problem: Multiple detections for same object

Example:
Box 1: [100, 100, 200, 200], conf=0.95
Box 2: [102, 101, 198, 199], conf=0.90  (overlapping!)

Solution: Keep high-confidence, remove overlapping

Algorithm:
1. Sort by confidence: [Box1 (0.95), Box2 (0.90)]
2. Keep Box1
3. Calculate IoU(Box1, Box2) = 0.95
4. If IoU > 0.5 threshold: Remove Box2
5. Result: Keep only Box1

Output: Single detection per object
```

## 📊 Performance Comparison

```
YOLOv8s (recommended for most cases):
- 60 FPS on GPU
- 75% mAP on COCO
- 22MB model size
- Fast training
- Easy to use

Faster R-CNN (for highest accuracy):
- 5 FPS on GPU
- 90% mAP on COCO
- 200MB+ model size
- Slower training
- More complex

EfficientDet (balanced):
- 30 FPS on GPU
- 85% mAP on COCO
- 100MB model size
- Scalable (D0 to D7)
```

## 🎯 Model Selection Guide

```
Real-time (> 30 fps) needed?
├─ Yes, edge device → YOLOv8n or YOLOv8s
├─ Yes, GPU → YOLOv8m
└─ No, accuracy critical → Faster R-CNN

Limited compute?
├─ Mobile/Edge → YOLOv8n, MobileNetv2
├─ CPU only → YOLOv8s (quantized)
└─ High-end GPU → Faster R-CNN

Type of objects?
├─ Small objects → High-res input, large model
├─ Crowded scene → Anchor-free (YOLO, FCOS)
└─ Mixed sizes → Multi-scale detector
```

## 📈 Applications

| Domain | Use Case |
|--------|----------|
| **Security** | Surveillance, intruder detection |
| **Autonomous driving** | Vehicle, pedestrian, traffic sign detection |
| **Retail** | Shelf item detection, checkout |
| **Healthcare** | Medical image analysis |
| **Sports** | Player tracking, ball detection |
| **Manufacturing** | Quality control, defect detection |
| **Wildlife** | Species identification and counting |

## 🎓 Learning Outcomes

- [x] Two-stage vs single-stage detectors
- [x] YOLO architecture and variants
- [x] IoU and mAP metrics
- [x] NMS post-processing
- [x] Model selection strategies

## 📚 Key Papers

- **Faster R-CNN**: "Faster R-CNN: Towards Real-Time Object Detection" (Ren et al., 2015)
- **YOLO**: "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2016)
- **YOLOv8**: Latest version (2023)

## 💡 Production Tips

```
1. Start with YOLOv8
   - Easy to use
   - Good balance of speed/accuracy

2. Use confidence threshold
   ```python
   model.predict(..., conf=0.5)
   # Lower threshold → more detections (more false positives)
   # Higher threshold → fewer detections (more false negatives)
   ```

3. NMS threshold control
   ```python
   model.predict(..., iou=0.5)
   # Higher threshold → Keep more overlapping boxes
   # Lower threshold → Remove more overlapping boxes
   ```

4. Benchmark on your data
   - Detection may work differently on your images
   - Test accuracy/speed trade-off

5. Consider fine-tuning
   - If domain-specific
   - Limited training data needed
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete
