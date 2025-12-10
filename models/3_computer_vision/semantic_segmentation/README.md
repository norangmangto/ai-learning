# Semantic Segmentation

Classify each pixel in an image, assigning each to a class category.

## 📋 Overview

**Task:** Image → Pixel-level class predictions
**Output:** Segmentation mask with class labels per pixel
**Baseline:** U-Net, FCN, DeepLabV3
**Applications:** Autonomous driving, medical imaging, scene understanding

## 🎯 Problem Formulation

```
Input: Image (H × W × 3)
       [building][building][sky]
       [building][person]  [sky]
       [road]    [person]  [sky]

Output: Segmentation map (H × W)
        [0][0][1]
        [0][2][1]
        [3][2][1]

Classes:
0 = Building
1 = Sky
2 = Person
3 = Road

Each pixel assigned to a class!
```

## 🏗️ Architecture Types

### 1. Fully Convolutional Networks (FCN)

```
Input (H × W × 3)
  ↓
Encoder (downsampling)
  Conv → Conv → Pool
  Conv → Conv → Pool

Mid bottleneck
  Conv → Conv

Decoder (upsampling)
  Upsampling → Conv
  Upsampling → Conv

Output (H × W × num_classes)
```

**Pros:** Simple, direct
**Cons:** Loses fine details (upsampling isn't perfect)

### 2. U-Net (Encoder-Decoder with Skip Connections)

```
Encoder (contracting path):
Input (572×572) → Conv → Pool → Conv → Pool → ...
                ↓
           save feature map         save feature map

Bottleneck:
Conv (at smallest resolution)

Decoder (expanding path):
↑ Concatenate ← Saved encoder features
Conv → Upsample
↑ Concatenate ← Saved encoder features
Conv → Upsample
...
Output (388×388 or padded)
```

**Pros:**
- Skip connections preserve spatial detail
- Better segmentation quality
- Popular for medical imaging

**Cons:**
- More parameters than FCN
- Slower training

### 3. DeepLabV3/DeepLabV3+

```
Encoder (ResNet backbone + atrous convolutions)
  Spatial pyramid pooling (different dilation rates)

Decoder
  Bilinear upsampling
  Skip connections (if V3+)

Output segmentation map
```

**Pros:**
- State-of-the-art accuracy
- Handles multiple scales
- Good for complex scenes

**Cons:**
- More complex architecture
- Slower inference

## 📊 Model Comparison

| Model | Type | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| FCN-8 | Encoder-Decoder | ⚡⚡⚡ | 65% | Small |
| U-Net | Encoder-Decoder | ⚡⚡ | 82% | Medium |
| DeepLabV3 | Dilated Conv | ⚡ | 92% | Large |
| SegFormer | Transformer | ⚡ | 94% | Large |

## 🚀 Quick Start: PyTorch + Torchvision

```python
import torch
import torchvision.models.segmentation as segmentation
from torchvision import transforms
from PIL import Image
import numpy as np

# Load pretrained DeepLabV3
model = segmentation.deeplabv3_resnet50(
    pretrained=True,
    num_classes=21  # Pascal VOC classes
)
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open('image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Segment
with torch.no_grad():
    output = model(input_tensor)

# Get segmentation map
segmentation_map = output['out'].argmax(1)[0]  # (H, W)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(segmentation_map.numpy())
plt.colorbar(label='Class')
plt.show()
```

## 📈 Loss Functions

### Cross-Entropy Loss
```python
import torch.nn.functional as F

# Standard per-pixel classification
loss = F.cross_entropy(predictions, targets)
# predictions: (B, C, H, W)
# targets: (B, H, W) with class indices
```

### Dice Loss (Better for imbalanced data)
```
Dice = 2 * |X ∩ Y| / (|X| + |Y|)

Focuses on overlap
Better for medical imaging
Handles class imbalance
```

### Focal Loss (For hard pixels)
```
Reduces weight for easy pixels
Focuses learning on hard/misclassified pixels
Good when background dominates
```

## 📊 Evaluation Metrics

### Intersection over Union (IoU) per class
```
IoU = TP / (TP + FP + FN)

Example:
- Pixel is building in GT, predicted building → TP
- Pixel is sky in GT, predicted building → FP
- Pixel is building in GT, predicted sky → FN

IoU = #correct pixels / #relevant pixels
```

### Mean IoU (mIoU)
```
Average IoU across all classes

mIoU = (1/num_classes) * Σ IoU_class

Example:
Class 0 (road): IoU = 0.90
Class 1 (car): IoU = 0.75
Class 2 (person): IoU = 0.60
mIoU = (0.90 + 0.75 + 0.60) / 3 = 0.75
```

### Pixel Accuracy
```
Accuracy = #correct pixels / #total pixels

Simple but can be misleading if classes imbalanced
(95% accuracy if model always predicts dominant class)
```

## 💡 Key Concepts

### Class Imbalance
```
Example:
- Road: 60% of pixels
- Car: 30% of pixels
- Person: 10% of pixels

Problem: Model biased to predict road (dominant class)

Solutions:
- Weighted loss (higher weight for rare classes)
- Oversampling patches with rare classes
- Dice/Focal loss instead of cross-entropy
- Data augmentation (focus on rare classes)
```

### Boundary Pixels
```
Challenge: Accurate boundaries are hard

Example:
Car vs. background boundary

Solution:
- Boundary loss (penalize boundary misclassification)
- Post-processing (morphological operations)
- Careful data annotation
```

### Multi-scale Processing
```
Objects at different scales

Solution: Use multi-scale features
- Feature pyramid networks
- Atrous/dilated convolutions
- Pyramid pooling modules
```

## 🎯 Model Selection

```
Quick baseline? → FCN or U-Net
Best accuracy? → DeepLabV3+ or SegFormer
Real-time? → Efficient segmentation networks
Medical imaging? → U-Net (well-established)
Autonomous driving? → DeepLabV3+
```

## 📈 Applications

| Domain | Task | Challenge |
|--------|------|-----------|
| **Autonomous driving** | Road/lane/vehicle/pedestrian | Real-time, safety-critical |
| **Medical imaging** | Organ/tumor/lesion segmentation | High accuracy critical |
| **Satellite imagery** | Land use classification | Large-scale data |
| **Scene understanding** | Indoor/outdoor objects | Complex interactions |
| **Video analysis** | Temporal consistency | Motion estimation |

## ⚠️ Common Challenges

1. **Boundary accuracy**
   - Often misclassified
   - Solution: Boundary loss, post-processing

2. **Small objects**
   - Hard to detect
   - Solution: Multi-scale features, higher resolution

3. **Class imbalance**
   - Rare classes ignored
   - Solution: Weighted loss, focal loss

4. **Computational cost**
   - High-resolution inference is slow
   - Solution: Downsampling, efficient architectures

## 🎓 Learning Outcomes

- [x] Encoder-decoder architectures
- [x] Skip connections for detail preservation
- [x] Atrous/dilated convolutions
- [x] Per-pixel loss functions
- [x] IoU and mIoU metrics

## 📚 Key Papers

- **FCN**: "Fully Convolutional Networks" (Long et al., 2015)
- **U-Net**: "U-Net: Convolutional Networks for Medical Image Segmentation" (Ronneberger et al., 2015)
- **DeepLabV3**: "Rethinking Atrous Convolution" (Chen et al., 2017)

## 💡 Implementation Tips

```
1. Start with pretrained backbone
   - ResNet50 or ResNet101 backbone
   - Pretrained on ImageNet

2. Data augmentation is crucial
   - Random flips, rotations, crops
   - Color jittering
   - Mixup/CutMix

3. Use appropriate loss
   - Imbalanced classes → Weighted or Focal loss
   - Medical imaging → Dice loss
   - General → Cross-entropy

4. Validation strategy
   - Use IoU, not just accuracy
   - Evaluate per-class IoU
   - Check boundary pixels

5. Post-processing
   - CRF (Conditional Random Field)
   - Morphological operations
   - Connected component analysis
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete
