# DeepLabV3 Semantic Segmentation

This directory contains implementations of DeepLabV3, a state-of-the-art semantic segmentation model using atrous (dilated) convolutions.

## Overview

DeepLabV3 improves semantic segmentation through atrous spatial pyramid pooling (ASPP), which captures multi-scale context efficiently without losing resolution.

## Key Innovations

### 1. Atrous (Dilated) Convolution

**Standard Convolution**:
```
Receptive field: 3×3
All adjacent pixels
```

**Atrous Convolution**:
```
Receptive field: 7×7 (rate=2)
Same parameters as 3×3
Sparse sampling pattern
```

**Formula**:
```python
y[i] = Σ x[i + r·k] × w[k]

where:
  r = dilation rate
  k = kernel position
```

**Benefits**:
- Larger receptive field without more parameters
- No resolution loss (vs. pooling then upsampling)
- Multi-scale context

### 2. ASPP (Atrous Spatial Pyramid Pooling)

Apply parallel atrous convolutions with different rates:

```
Input Features
      ↓
   ┌──┴──┬───────┬───────┬───────┬────────┐
   │     │       │       │       │        │
1×1   3×3     3×3     3×3    Global   Image
Conv  r=6     r=12    r=18   Pool     Pool
   │     │       │       │       │        │
   └──┬──┴───────┴───────┴───────┴────────┘
      ↓
   Concatenate
      ↓
   1×1 Conv (fuse)
      ↓
   Output
```

**Why ASPP?**
- Captures multi-scale context (small, medium, large objects)
- No need for image pyramids (more efficient)
- Global context via global pooling branch
- Learns to combine different scales

## Architecture

### Overall Structure
```
Input Image (512×512×3)
   ↓
ResNet-50/101 Backbone (output_stride=16)
├─ Entry flow
├─ Middle flow (atrous convolutions)
└─ Exit flow
   ↓
ASPP Module (multi-scale context)
├─ 1×1 conv
├─ 3×3 atrous conv (rate=6)
├─ 3×3 atrous conv (rate=12)
├─ 3×3 atrous conv (rate=18)
├─ Global average pooling + 1×1 conv
└─ Concatenate + 1×1 conv
   ↓
Upsampling (16×, bilinear or learned)
   ↓
Output Segmentation (512×512×num_classes)
```

### Output Stride

**Definition**: Ratio of input to feature map resolution

```python
output_stride = 16:  # 512×512 → 32×32 features
  - Less computation
  - Coarser features

output_stride = 8:   # 512×512 → 64×64 features
  - More computation
  - Finer features
  - Better accuracy
```

### Atrous Convolution in Backbone

Replace pooling with atrous convolutions:

```python
# Standard ResNet (output_stride=32)
Block3: stride=2  # Downsample
Block4: stride=2  # Downsample

# DeepLabV3 (output_stride=16)
Block3: stride=2  # Downsample
Block4: stride=1, rate=2  # Keep resolution, increase receptive field
```

## ASPP Module Details

### Five Parallel Branches

#### Branch 1: 1×1 Convolution
```python
# Capture local features
conv1x1 = nn.Conv2d(in_channels, 256, kernel_size=1)
```

#### Branch 2-4: 3×3 Atrous Convolutions
```python
# Multi-scale context
atrous_conv6 = nn.Conv2d(in_channels, 256, 3, padding=6, dilation=6)
atrous_conv12 = nn.Conv2d(in_channels, 256, 3, padding=12, dilation=12)
atrous_conv18 = nn.Conv2d(in_channels, 256, 3, padding=18, dilation=18)
```

#### Branch 5: Global Average Pooling
```python
# Global context
global_pool = nn.AdaptiveAvgPool2d(1)  # → [B, C, 1, 1]
conv1x1 = nn.Conv2d(in_channels, 256, 1)
upsample = F.interpolate(x, size=spatial_size)  # Back to feature size
```

### Fusion
```python
# Concatenate all branches
concat = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
# concat: [B, 256×5, H, W]

# Fuse with 1×1 conv
output = nn.Conv2d(256*5, 256, kernel_size=1)(concat)
# output: [B, 256, H, W]
```

## Usage

```bash
# Train DeepLabV3 with ResNet-50
python train_pytorch.py --backbone resnet50 --output_stride 16

# Better accuracy with output_stride=8
python train_pytorch.py --backbone resnet50 --output_stride 8

# Larger backbone
python train_pytorch.py --backbone resnet101 --output_stride 16
```

## Configuration

```python
CONFIG = {
    'backbone': 'resnet50',    # 'resnet50', 'resnet101'
    'output_stride': 16,       # 8 or 16
    'num_classes': 21,         # PASCAL VOC
    'aspp_rates': [6, 12, 18], # Dilation rates
    'image_size': 513,         # Odd size for better alignment
    'batch_size': 8,
    'epochs': 100,
    'learning_rate': 0.007,
}
```

## Loss Function

### Cross-Entropy Loss
```python
loss = nn.CrossEntropyLoss()(predictions, targets)
```

### With Class Weighting
```python
# Handle class imbalance
class_weights = compute_class_weights(dataset)
loss = nn.CrossEntropyLoss(weight=class_weights)
```

### Auxiliary Loss (Optional)
```python
# Add loss from intermediate features
main_loss = ce_loss(main_output, targets)
aux_loss = ce_loss(aux_output, targets)
total_loss = main_loss + 0.4 × aux_loss
```

## Training Strategy

### Learning Rate Schedule
```python
# Polynomial decay
lr = base_lr × (1 - iter/max_iter)^power

# Common: power=0.9
```

### Data Augmentation
```python
- Random scaling (0.5 to 2.0)
- Random cropping (crop_size=513)
- Random horizontal flip
- Color jittering (optional)
```

### Multi-Grid (Advanced)
```python
# Apply different dilation rates within blocks
multi_grid = [1, 2, 4]  # Within Block4

# Further increases receptive field
```

## Evaluation Metrics

### Mean IoU (Primary Metric)
```python
IoU_i = TP_i / (TP_i + FP_i + FN_i)
mIoU = mean(IoU_i for all classes)
```

### Pixel Accuracy
```python
accuracy = correct_pixels / total_pixels
```

### Per-Class IoU
```python
# Identify problematic classes
class_iou = {
    'road': 0.98,
    'car': 0.91,
    'person': 0.78,  # ← Harder class
    ...
}
```

## Datasets

### PASCAL VOC 2012
- **Classes**: 20 + background
- **mIoU**: ~85% (DeepLabV3+)

### Cityscapes
- **Classes**: 19 (urban scenes)
- **mIoU**: ~82% (DeepLabV3+)

### ADE20K
- **Classes**: 150
- **mIoU**: ~45% (DeepLabV3+)

### COCO-Stuff
- **Classes**: 171
- **Challenging**

## Applications

- **Autonomous Driving**: Scene understanding, lane detection
- **Medical Imaging**: Organ/tumor segmentation
- **Satellite Imagery**: Land cover classification
- **Augmented Reality**: Scene segmentation
- **Robotics**: Navigation, manipulation
- **Video Analysis**: Action recognition, object tracking

## Comparison with Other Methods

| Method | Backbone | mIoU (VOC) | Speed | Key Feature |
|--------|----------|------------|-------|-------------|
| FCN | VGG-16 | 62.7% | Fast | First end-to-end |
| U-Net | Custom | ~70% | Fast | Skip connections |
| DeepLabV2 | ResNet-101 | 79.7% | Medium | CRF, atrous conv |
| PSPNet | ResNet-101 | 82.6% | Medium | Pyramid pooling |
| DeepLabV3 | ResNet-101 | 85.7% | Medium | ASPP |
| DeepLabV3+ | Xception | 89.0% | Medium | Encoder-decoder |

## DeepLabV3 vs DeepLabV3+

### DeepLabV3
```
- Encoder only
- Simple upsampling
- Good for high-level features
```

### DeepLabV3+
```
- Encoder-decoder
- Skip connections
- Better boundaries
- Current state-of-the-art
```

## Advantages

1. **Multi-Scale Context**: ASPP captures different scales
2. **No Resolution Loss**: Atrous convolutions preserve spatial info
3. **Efficient**: No need for image pyramids
4. **Strong Backbone**: ResNet/Xception features
5. **Global Context**: Image-level pooling branch

## Limitations

1. **Computational Cost**: Atrous convolutions are expensive
2. **Memory**: High-resolution features need memory
3. **Boundary Quality**: Can be improved (→ DeepLabV3+)
4. **Small Objects**: May miss very small objects

## Best Practices

1. **Output Stride**: Use 8 for better accuracy if memory allows
2. **Pre-training**: Always use ImageNet weights
3. **Multi-Scale Training**: Improves robustness
4. **Synchronized Batch Norm**: For multi-GPU training
5. **Large Crop Size**: 513×513 recommended
6. **Polynomial LR Decay**: Works better than step decay
7. **Data Augmentation**: Critical for good performance

## Advanced Techniques

### Multi-Scale Inference
```python
# Test at multiple scales
scales = [0.5, 0.75, 1.0, 1.25, 1.5]
predictions = []
for scale in scales:
    pred = model(resize(image, scale))
    predictions.append(resize(pred, original_size))

# Average predictions
final = torch.mean(torch.stack(predictions), dim=0)
```

### Flip Augmentation
```python
# Also test with horizontal flip
pred_original = model(image)
pred_flipped = model(flip(image))
final = (pred_original + flip(pred_flipped)) / 2
```

### CRF Post-Processing
```python
# Refine boundaries (optional, expensive)
refined = dense_crf(prediction, image)
```

## Implementation Tips

### Synchronized Batch Norm
```python
# Important for multi-GPU training
# Regular BN computes stats per GPU (wrong for small batch per GPU)
from torch.nn import SyncBatchNorm
model = SyncBatchNorm.convert_sync_batchnorm(model)
```

### Mixed Precision Training
```python
# Faster training, less memory
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Modern Extensions

- **DeepLabV3+**: Add decoder for better boundaries
- **Auto-DeepLab**: Neural architecture search
- **Panoptic-DeepLab**: Instance + semantic segmentation
- **SegFormer**: Transformer-based alternative

## Requirements

```bash
pip install torch torchvision Pillow numpy matplotlib
```

## References

- DeepLabV3: [Chen et al., 2017](https://arxiv.org/abs/1706.05587)
- DeepLabV3+: [Chen et al., 2018](https://arxiv.org/abs/1802.02611)
- Atrous Convolution: [Yu & Koltun, 2016](https://arxiv.org/abs/1511.07122)
- PASCAL VOC: [Everingham et al., 2010](http://host.robots.ox.ac.uk/pascal/VOC/)
- Cityscapes: [Cordts et al., 2016](https://arxiv.org/abs/1604.01685)
