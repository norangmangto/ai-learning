# FCN (Fully Convolutional Networks)

This directory contains implementations of FCN for semantic segmentation.

## Overview

FCN was the first end-to-end, fully convolutional approach for semantic segmentation, replacing fully connected layers with convolutional layers to enable dense predictions.

## Problem Definition

**Input**: RGB image (H×W×3)
**Output**: Segmentation map (H×W) where each pixel has a class label

Example classes: road, car, person, sky, building, etc.

## Key Innovation

### Traditional CNN for Classification
```
Image → Conv → Pool → Conv → Pool → FC → FC → Class label
                                      ↑
                                 Fixed size input
                                 Single prediction
```

### FCN for Segmentation
```
Image → Conv → Pool → Conv → Pool → Conv → Upsample → Pixel-wise predictions
                                      ↑
                              No FC layers!
                              Spatial resolution preserved
```

**Benefits**:
- Accept any input size
- Output spatial map (not single class)
- End-to-end training
- Efficient (no sliding window)

## Architecture Variants

### FCN-32s (Coarse)
```
Input (224×224)
   ↓
VGG-16 backbone
   ↓
1×1 Conv (num_classes channels)
   ↓
32× Upsampling (transposed conv)
   ↓
Output (224×224)
```

**Characteristics**:
- Single upsampling step (32×)
- Coarse predictions
- Fast but less accurate

### FCN-16s (Medium)
```
Input
   ↓
VGG backbone
├─ Pool4 (1/16) ────┐
└─ Pool5 (1/32)     │
      ↓              │
   1×1 Conv          │
      ↓              │
   2× Upsample ──────┤ Fuse
      ↓              │
   16× Upsample ←────┘
      ↓
   Output
```

**Characteristics**:
- Adds skip connection from Pool4
- Better detail than FCN-32s
- 2× refinement

### FCN-8s (Fine)
```
Input
   ↓
VGG backbone
├─ Pool3 (1/8) ─────────┐
├─ Pool4 (1/16) ────┐   │
└─ Pool5 (1/32)     │   │
      ↓              │   │
   1×1 Conv          │   │
      ↓              │   │
   2× Upsample ──────┤   │
      ↓              │   │ Fuse
   2× Upsample ──────┤   │
      ↓              │   │
   8× Upsample ←─────┴───┘
      ↓
   Output
```

**Characteristics**:
- Two skip connections (Pool3, Pool4)
- Best detail and accuracy
- Most commonly used

## Key Components

### 1. Backbone (VGG-16)
```python
# Pre-trained classification network
# Remove FC layers, keep only conv layers
vgg16 = torchvision.models.vgg16(pretrained=True)
features = vgg16.features  # Only conv+pool layers
```

### 2. 1×1 Convolution
```python
# Convert feature maps to class scores
# No spatial downsampling
score = nn.Conv2d(512, num_classes, kernel_size=1)
```

### 3. Transposed Convolution (Upsampling)
```python
# Learnable upsampling (vs. fixed bilinear)
upsample = nn.ConvTranspose2d(
    in_channels, out_channels,
    kernel_size=4, stride=2, padding=1
)
```

### 4. Skip Connections
```python
# Fuse coarse deep features with fine shallow features
output = upsample(deep_features) + shallow_features
```

## Loss Function

### Cross-Entropy Loss
```python
# Standard loss for multi-class segmentation
loss = CrossEntropyLoss()(predictions, ground_truth)

# predictions: [B, C, H, W]
# ground_truth: [B, H, W] with class indices
```

### Weighted Loss (for class imbalance)
```python
# Weight classes inversely to frequency
class_weights = 1.0 / class_frequencies
loss = CrossEntropyLoss(weight=class_weights)
```

## Training Strategy

### Transfer Learning
```python
# 1. Load VGG-16 pre-trained on ImageNet
# 2. Freeze early layers (or use small LR)
# 3. Train only upsampling and scoring layers
# 4. (Optional) Fine-tune entire network
```

### Learning Rates
```python
# Different LRs for different parts
optimizer = optim.SGD([
    {'params': backbone.parameters(), 'lr': 1e-5},    # Small LR
    {'params': upsampling.parameters(), 'lr': 1e-3},  # Large LR
], momentum=0.9)
```

## Usage

```bash
# Train FCN-8s
python train_pytorch.py --model fcn8s

# Train FCN-16s
python train_pytorch.py --model fcn16s

# Train FCN-32s (faster but less accurate)
python train_pytorch.py --model fcn32s
```

## Configuration

```python
CONFIG = {
    'model': 'fcn8s',          # 'fcn8s', 'fcn16s', 'fcn32s'
    'backbone': 'vgg16',
    'num_classes': 21,         # PASCAL VOC
    'image_size': 512,
    'batch_size': 8,
    'epochs': 100,
    'learning_rate': 1e-4,
}
```

## Evaluation Metrics

### Pixel Accuracy
```python
accuracy = correct_pixels / total_pixels
```

### Mean IoU (mIoU)
```python
# Intersection over Union per class, then average
IoU_i = TP_i / (TP_i + FP_i + FN_i)
mIoU = mean(IoU_i for all classes)
```

### Per-Class IoU
```python
# Useful for understanding class-specific performance
class_ious = {
    'road': 0.95,
    'car': 0.87,
    'person': 0.72,
    ...
}
```

### Frequency Weighted IoU
```python
# Weight by class frequency
fw_IoU = Σ (freq_i × IoU_i)
```

## Datasets

### PASCAL VOC
- **Classes**: 20 object classes + background
- **Images**: ~10,000 training, ~1,000 val
- **Resolution**: Variable (typically 512×512)

### COCO-Stuff
- **Classes**: 80 thing + 91 stuff classes
- **Images**: 118,000 training
- **More challenging**

### Cityscapes
- **Classes**: 19 classes (urban scenes)
- **Images**: 5,000 high-res (1024×2048)
- **Autonomous driving**

### ADE20K
- **Classes**: 150 classes
- **Images**: 20,000 training
- **Diverse scenes**

## Applications

### Autonomous Driving
- Lane detection
- Drivable area segmentation
- Object/pedestrian segmentation

### Medical Imaging
- Organ segmentation
- Tumor detection
- Cell segmentation

### Agriculture
- Crop segmentation
- Disease detection
- Yield estimation

### Robotics
- Scene understanding
- Navigation
- Object manipulation

### Satellite Imagery
- Land use classification
- Building detection
- Crop monitoring

## Advantages

1. **End-to-End**: No post-processing pipelines
2. **Efficient**: Single forward pass (vs. sliding window)
3. **Flexible**: Any input size
4. **Transfer Learning**: Leverage ImageNet pre-training
5. **Skip Connections**: Preserve fine details

## Limitations

1. **Coarse Boundaries**: Upsampling can blur boundaries
2. **Fixed Receptive Field**: Limited context modeling
3. **No Multi-Scale**: Processes single resolution
4. **VGG Backbone**: Outdated, ResNet is better

## Improvements Over FCN

### U-Net (2015)
- More skip connections
- Symmetric architecture
- Better for small datasets

### DeepLab (2017)
- Atrous convolutions
- Multi-scale processing
- Better boundaries

### PSPNet (2017)
- Pyramid pooling
- Global context
- Better on diverse scenes

### HRNet (2019)
- Maintain high resolution
- Better detail preservation

## Best Practices

1. **Use Pre-trained Backbone**: ImageNet weights
2. **Data Augmentation**: Random crops, flips, color jitter
3. **Class Weighting**: Handle imbalanced classes
4. **Multi-Scale Training**: Improves robustness
5. **High Resolution**: Better detail (if memory allows)
6. **Batch Normalization**: Stabilizes training
7. **Learning Rate Schedule**: Cosine or polynomial decay

## Common Issues

### Checkerboard Artifacts
- **Cause**: Transposed convolution
- **Solution**: Use bilinear upsampling or larger kernel

### Poor Boundary Quality
- **Cause**: Coarse upsampling
- **Solution**: More skip connections, CRF post-processing

### Class Imbalance
- **Cause**: Some classes rare
- **Solution**: Weighted loss, focal loss

### Overfitting
- **Cause**: Small dataset
- **Solution**: Data augmentation, regularization

## Advanced Techniques

### Multi-Scale Testing
```python
# Test at multiple scales, combine results
scales = [0.5, 0.75, 1.0, 1.25, 1.5]
predictions = []
for scale in scales:
    pred = model(resize(image, scale))
    predictions.append(resize(pred, original_size))
final_pred = ensemble(predictions)
```

### CRF Post-Processing
```python
# Refine boundaries using Conditional Random Fields
refined = dense_crf(prediction, image)
```

### Online Hard Example Mining
```python
# Focus on hard pixels during training
hard_pixels = top_k_loss(predictions, targets, k=0.1)
loss = hard_pixels.mean()
```

## Modern Alternatives

For new projects, consider:
- **DeepLabV3+**: Better accuracy, atrous convolutions
- **HRNet**: Better detail preservation
- **Segformer**: Transformer-based, state-of-the-art
- **Mask2Former**: Universal segmentation

## Requirements

```bash
pip install torch torchvision Pillow numpy matplotlib
```

## References

- FCN: [Long et al., 2015](https://arxiv.org/abs/1411.4038)
- PASCAL VOC: [Everingham et al., 2010](http://host.robots.ox.ac.uk/pascal/VOC/)
- Evaluation Metrics: [Garcia-Garcia et al., 2017](https://arxiv.org/abs/1704.06857)
