# Anchor-Free Object Detection (CenterNet)

This directory contains implementations of anchor-free object detection using CenterNet.

## Overview

Anchor-free detection treats objects as keypoints (center points) without using predefined anchor boxes, simplifying the detection pipeline.

## Anchor-Based vs. Anchor-Free

### Traditional Anchor-Based Detection (Faster R-CNN, YOLO, RetinaNet)
```
1. Generate thousands of anchor boxes
2. Classify each anchor (object or background)
3. Regress anchor to actual box
4. Apply NMS to remove duplicates

Problems:
- Many hyperparameters (anchor sizes, ratios, scales)
- Imbalanced training (many negatives)
- Sensitive to anchor design
- Requires NMS post-processing
```

### Anchor-Free Detection (CenterNet)
```
1. Detect object centers as keypoints
2. Regress size and offset for each center

Benefits:
- No anchor boxes needed
- Fewer hyperparameters
- NMS-free (one object per center point)
- Simpler pipeline
- Faster inference
```

## CenterNet Architecture

### Overall Pipeline
```
Input Image (512×512×3)
   ↓
Backbone (ResNet, DLA, HourglassNet)
   ↓
Feature Maps (128×128×C)
   ↓
Three Prediction Heads
├─ Heatmap (128×128×K)      ← Object centers (one per class)
├─ Size (128×128×2)         ← Width and height
└─ Offset (128×128×2)       ← Sub-pixel correction
   ↓
Decode Detections (no NMS needed!)
   ↓
[x, y, w, h, class, score]
```

### Key Components

#### 1. Heatmap Head
```python
# Predicts probability of object center at each location
heatmap: [B, K, H, W]  # K = number of classes

# Gaussian kernel around ground truth centers
# Peak at center, falls off with distance
```

#### 2. Size Head
```python
# Predicts object size at center point
size: [B, 2, H, W]  # (width, height)

# Directly regresses bounding box dimensions
```

#### 3. Offset Head
```python
# Corrects quantization error from downsampling
offset: [B, 2, H, W]  # (dx, dy)

# Ground truth: (x/R - floor(x/R), y/R - floor(y/R))
# where R = output stride (e.g., 4)
```

## Loss Functions

### 1. Heatmap Loss (Focal Loss variant)
```python
# Modified focal loss for keypoint detection
L_heatmap = -1/N Σ {
    (1 - p)^α × log(p)           if y = 1  (positive)
    (1 - y)^β × p^α × log(1-p)   otherwise  (negative)
}

where:
  y = ground truth heatmap (Gaussian)
  p = predicted probability
  α = 2 (focus on hard examples)
  β = 4 (reduce penalty near GT)
  N = number of objects
```

**Why this loss?**
- Handles extreme class imbalance (one center per object, thousands of negatives)
- Reduces penalty for predictions near ground truth
- Focuses on hard examples

### 2. Size Loss (L1)
```python
# L1 loss only at object centers
L_size = 1/N Σ |predicted_size - ground_truth_size|

# Only compute at GT center locations
```

### 3. Offset Loss (L1)
```python
# L1 loss for sub-pixel refinement
L_offset = 1/N Σ |predicted_offset - ground_truth_offset|

# Corrects discretization error
```

### Total Loss
```python
L_total = L_heatmap + λ_size × L_size + λ_offset × L_offset

# Common weights:
λ_size = 0.1
λ_offset = 1.0
```

## Keypoint Detection

### Creating Ground Truth Heatmaps
```python
# For each object with center (cx, cy):
1. Compute heatmap center: (cx/R, cy/R)  # R = output stride
2. Apply 2D Gaussian around center:

   G(x, y) = exp(-(x-cx)² + (y-cy)² / 2σ²)

   where σ = object_size / 6

3. Take maximum if multiple objects overlap
```

### Extracting Detections
```python
# 1. Find local maxima in heatmap
peaks = local_maxima(heatmap)  # 3×3 max pooling

# 2. For each peak above threshold:
center_x = x + offset_x[x, y]
center_y = y + offset_y[x, y]
w, h = size[x, y]

# 3. Convert to bounding box
bbox = [center_x - w/2, center_y - h/2, w, h]

# 4. No NMS needed! (one object per center)
```

## Usage

```bash
# Train CenterNet
python train_pytorch.py

# Use different backbone
python train_pytorch.py --backbone resnet18
python train_pytorch.py --backbone resnet50
python train_pytorch.py --backbone dla34
```

## Configuration

```python
CONFIG = {
    'backbone': 'resnet18',
    'num_classes': 80,         # COCO classes
    'input_size': 512,
    'output_stride': 4,        # Downsampling ratio
    'max_objects': 100,        # Max detections per image
    'batch_size': 16,
    'epochs': 140,
    'learning_rate': 1.25e-4,
}
```

## Backbone Options

### ResNet
- **ResNet-18**: Fast, good for real-time
- **ResNet-50**: Better accuracy, slower
- **ResNet-101**: Best accuracy, slowest

### DLA (Deep Layer Aggregation)
- **DLA-34**: Good balance
- Better feature fusion
- Higher accuracy than ResNet

### HourglassNet
- **Hourglass-104**: Best accuracy
- Symmetric encoder-decoder
- Slowest inference

## Advantages

### 1. Simplicity
- No anchor boxes to design
- No anchor matching logic
- Single-stage, end-to-end

### 2. No NMS
- One object per center point
- Faster post-processing
- Deterministic (no NMS threshold to tune)

### 3. Flexibility
- Easy to extend to other tasks:
  - 3D detection (add depth prediction)
  - Pose estimation (add keypoints)
  - Multi-task learning

### 4. Performance
- Competitive accuracy with anchor-based
- Faster inference (no NMS)
- Better on small objects

## Limitations

### 1. Overlapping Objects
- Objects with same center are hard to detect
- Rare in practice

### 2. Training Time
- Focal loss can be slower to converge
- Requires careful hyperparameter tuning

### 3. Memory
- Heatmap head adds overhead
- Larger than anchor-based heads

## Applications

- **Autonomous Driving**: Detect vehicles, pedestrians
- **Surveillance**: Track people and objects
- **Robotics**: Object manipulation
- **Retail**: Product detection
- **Medical**: Organ/lesion detection

## Comparison with Other Methods

| Method | Anchors | NMS | Speed | Accuracy | Complexity |
|--------|---------|-----|-------|----------|------------|
| Faster R-CNN | Yes | Yes | Slow | High | High |
| YOLO | Yes | Yes | Fast | Medium | Medium |
| RetinaNet | Yes | Yes | Medium | High | High |
| CenterNet | No | No | Fast | High | Low |
| FCOS | No | Yes | Fast | High | Medium |

## Variants

### CenterNet2
```python
# Two-stage refinement
# Better on small objects
# Cascade architecture
```

### TTFNet (Training Time-Friendly)
```python
# Faster training
# Better on small objects
# Modified loss functions
```

### FCOS (Fully Convolutional One-Stage)
```python
# Similar idea, per-pixel prediction
# Uses center-ness branch
# Still requires NMS
```

## Best Practices

1. **Use DLA backbone**: Better than ResNet for this task
2. **Multi-scale training**: Improves robustness
3. **Data augmentation**: Random crops, flips, color jitter
4. **Warm-up learning rate**: Start small, increase gradually
5. **Focal loss parameters**: α=2, β=4 work well
6. **Output stride**: 4 is standard, 2 for small objects

## Advanced Features

### Multi-Scale Testing
```python
# Test at multiple scales, combine results
scales = [0.5, 0.75, 1.0, 1.25, 1.5]
for scale in scales:
    detections.append(detect(resize(image, scale)))
```

### 3D Detection
```python
# Add depth and orientation prediction
heads = {
    'heatmap': K channels,
    'size': 2 channels (w, h),
    'offset': 2 channels (dx, dy),
    'depth': 1 channel,       # ← Add depth
    'rotation': 8 channels,   # ← Add rotation bins
}
```

### Multi-Task Learning
```python
# Combine with other tasks
tasks = {
    'detection': [...],
    'segmentation': [...],
    'depth': [...],
}
```

## Debugging Tips

### Heatmap Not Learning
- Check focal loss α, β parameters
- Verify Gaussian radius computation
- Ensure proper normalization

### Poor Size Prediction
- Increase λ_size weight
- Use better size encoding (log scale)
- Add IoU loss

### Missing Small Objects
- Reduce output stride (4 → 2)
- Use FPN for multi-scale features
- Increase resolution

## Requirements

```bash
pip install torch torchvision pycocotools matplotlib
```

## References

- CenterNet: [Zhou et al., 2019](https://arxiv.org/abs/1904.07850)
- CornerNet: [Law & Deng, 2018](https://arxiv.org/abs/1808.01244)
- FCOS: [Tian et al., 2019](https://arxiv.org/abs/1904.01355)
- CenterNet2: [Zhou et al., 2021](https://arxiv.org/abs/2103.07461)
