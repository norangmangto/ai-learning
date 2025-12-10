# Image Classification Models

## Overview

Image classification assigns images to predefined categories. This directory contains implementations using Convolutional Neural Networks (CNNs) and Vision Transformers across PyTorch and TensorFlow.

## What is Image Classification?

Image classification is assigning labels to images:
- **Binary**: Cat vs Dog, Healthy vs Diseased
- **Multi-class**: ImageNet (1000 classes), CIFAR-10 (10 classes)
- **Multi-label**: Multiple objects in one image

### Common Use Cases
- Medical imaging (X-ray, MRI classification)
- Autonomous vehicles (traffic sign recognition)
- Content moderation (inappropriate content detection)
- Wildlife monitoring (species identification)
- Quality control (defect detection)
- Security (face recognition, ID verification)

## Architectures Implemented

### 1. Convolutional Neural Networks (CNNs)

**File:** `train_pytorch.py`, `train_tensorflow.py`

**Theory:**
```
Input Image → Conv Layers → Pooling → Conv Layers → Flatten → Dense → Softmax
```

**CNN Building Blocks:**
1. **Convolutional Layer**: Extracts features using filters
   ```
   Output = Conv2D(input, kernel_size=3x3, filters=64)
   ```

2. **Activation**: Introduces non-linearity
   ```
   ReLU(x) = max(0, x)
   ```

3. **Pooling**: Reduces spatial dimensions
   ```
   MaxPool2D(pool_size=2x2) → reduces size by half
   ```

4. **Batch Normalization**: Stabilizes training
   ```
   Normalizes activations, speeds convergence
   ```

5. **Dropout**: Prevents overfitting
   ```
   Randomly drops neurons during training
   ```

**Popular Architectures:**

#### ResNet (Residual Network)
- **Parameters**: 25M (ResNet-50), 60M (ResNet-101)
- **Key Innovation**: Skip connections
- **Architecture**: 50-152 layers
- **When to Use**: General purpose, proven performance

```python
from torchvision.models import resnet50, resnet101
model = resnet50(pretrained=True)
```

**Advantages:**
- ✅ Solves vanishing gradient problem
- ✅ Can train very deep networks
- ✅ Excellent performance
- ✅ Widely supported

#### EfficientNet
- **Parameters**: 5M (B0) to 66M (B7)
- **Key Innovation**: Compound scaling (depth, width, resolution)
- **Architecture**: Optimized for efficiency
- **When to Use**: Need efficiency, mobile deployment

**Advantages:**
- ✅ Best accuracy/parameter ratio
- ✅ Scalable (B0-B7)
- ✅ Mobile-friendly
- ✅ Faster inference

#### VGG (Visual Geometry Group)
- **Parameters**: 138M (VGG-16)
- **Key Innovation**: Deep uniform architecture
- **Architecture**: 16-19 layers, 3x3 convolutions
- **When to Use**: Teaching, transfer learning

**Advantages:**
- ✅ Simple architecture
- ✅ Good for transfer learning
- ✅ Well understood

**Limitations:**
- ❌ Very large model size
- ❌ Slow inference
- ❌ High memory usage

#### Inception (GoogLeNet)
- **Parameters**: 24M (Inception-v3)
- **Key Innovation**: Multi-scale feature extraction
- **Architecture**: Inception modules
- **When to Use**: Different scale features important

**Advantages:**
- ✅ Efficient multi-scale processing
- ✅ Good accuracy
- ✅ Moderate size

#### MobileNet
- **Parameters**: 4M (MobileNet-V2)
- **Key Innovation**: Depthwise separable convolutions
- **Architecture**: Optimized for mobile
- **When to Use**: Mobile/edge deployment, real-time

**Advantages:**
- ✅ Very fast inference
- ✅ Small model size
- ✅ Low power consumption
- ✅ Mobile/IoT friendly

### 2. Vision Transformers (ViT)

**File:** `train_pytorch_v2.py`

**Theory:**
```
Image → Patches → Linear Projection → Transformer Encoder → Classification Head
```

**How ViT Works:**
1. Split image into patches (16x16 or 32x32)
2. Flatten patches and linearly project to embeddings
3. Add positional embeddings
4. Pass through transformer encoder (self-attention)
5. Use [CLS] token for classification

**When to Use:**
- Large datasets (ImageNet-21k, 300M+ images)
- High computational budget
- State-of-the-art performance needed
- Transfer learning from large models

**Advantages:**
- ✅ State-of-the-art accuracy (with large data)
- ✅ Global context modeling
- ✅ Scalable architecture
- ✅ Interpretable attention maps

**Limitations:**
- ❌ Requires huge datasets (or pretrained models)
- ❌ High computational cost
- ❌ Poor with small data
- ❌ Larger model sizes

**Variants:**
- **ViT-Base**: 86M parameters
- **ViT-Large**: 307M parameters
- **ViT-Huge**: 632M parameters
- **DeiT**: Data-efficient ViT (works with less data)
- **Swin Transformer**: Hierarchical ViT (better for detection/segmentation)

## Implementation Files

### PyTorch Implementation
**Files:** `train_pytorch.py`, `train_pytorch_v2.py`

**Features:**
- Pre-trained models from torchvision
- Custom CNN architectures
- Transfer learning
- Data augmentation
- Mixed precision training

**Supported Models:**
```python
from torchvision.models import (
    resnet50, resnet101,
    efficientnet_b0, efficientnet_b4,
    vgg16, vgg19,
    mobilenet_v2, mobilenet_v3_large,
    inception_v3,
    vit_b_16, vit_l_16
)
```

### TensorFlow/Keras Implementation
**File:** `train_tensorflow.py`

**Features:**
- Keras Applications models
- TensorFlow Hub integration
- TF Dataset pipeline
- Distributed training

**Supported Models:**
```python
from tensorflow.keras.applications import (
    ResNet50, ResNet101,
    EfficientNetB0, EfficientNetB7,
    VGG16, VGG19,
    MobileNetV2,
    InceptionV3
)
```

## Datasets

### Common Benchmarks

| Dataset | Classes | Training | Test | Resolution | Difficulty |
|---------|---------|----------|------|------------|------------|
| MNIST | 10 | 60k | 10k | 28x28 | Easy |
| Fashion-MNIST | 10 | 60k | 10k | 28x28 | Easy |
| CIFAR-10 | 10 | 50k | 10k | 32x32 | Medium |
| CIFAR-100 | 100 | 50k | 10k | 32x32 | Hard |
| ImageNet | 1000 | 1.2M | 50k | 224x224 | Hard |
| iNaturalist | 10k | 800k | 100k | Various | Very Hard |

## Quick Start

### 1. PyTorch with Pre-trained ResNet
```bash
python train_pytorch.py
```

### 2. PyTorch with Vision Transformer
```bash
python train_pytorch_v2.py
```

### 3. TensorFlow/Keras
```bash
python train_tensorflow.py
```

## Transfer Learning

### Why Transfer Learning?

- ✅ Faster training
- ✅ Better accuracy with less data
- ✅ Leverages pre-trained knowledge
- ✅ Requires less computational resources

### Fine-tuning Strategies

#### 1. Feature Extraction (Fastest)
```python
# Freeze all layers, train classifier only
model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

#### 2. Fine-tune Last Few Layers
```python
# Freeze early layers, fine-tune later layers
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False
```

#### 3. Full Fine-tuning
```python
# Unfreeze all, train with lower learning rate
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Use lr = 1e-4 instead of 1e-3
```

#### 4. Progressive Unfreezing
```python
# Gradually unfreeze layers from end to start
# Epoch 1-2: classifier only
# Epoch 3-4: + layer4
# Epoch 5+: all layers
```

## Data Augmentation

### Basic Augmentations
```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])
```

### Advanced Augmentations
```python
import albumentations as A

transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
    A.OneOf([
        A.GaussNoise(),
        A.GaussianBlur(),
        A.MotionBlur(),
    ]),
    A.OneOf([
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.ElasticTransform(),
    ]),
    A.Normalize(),
])
```

### AutoAugment / RandAugment
```python
from torchvision.transforms import AutoAugment

transforms.Compose([
    AutoAugment(),
    transforms.ToTensor(),
])
```

## Best Practices

### 1. Learning Rate Scheduling
```python
# Cosine Annealing
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Reduce on Plateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

# Step Decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### 2. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(inputs)
        loss = criterion(output, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Early Stopping
```python
best_val_acc = 0
patience = 5
patience_counter = 0

for epoch in range(epochs):
    train_loss = train()
    val_acc = validate()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### 4. Regularization Techniques
```python
# Weight Decay (L2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Dropout
nn.Dropout(p=0.5)

# Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixup / CutMix
# Mixes two images and labels
```

### 5. Optimal Hyperparameters

**Small Datasets (< 10k)**
```python
batch_size = 32
learning_rate = 1e-4  # lower for transfer learning
epochs = 50-100
weight_decay = 1e-3
dropout = 0.5
```

**Medium Datasets (10k-100k)**
```python
batch_size = 64
learning_rate = 1e-3
epochs = 50
weight_decay = 1e-4
dropout = 0.3
```

**Large Datasets (> 100k)**
```python
batch_size = 128-256
learning_rate = 1e-3 to 1e-2
epochs = 30-50
weight_decay = 1e-4
dropout = 0.2
```

## Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Top-1 Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Top-5 Accuracy (for ImageNet-like)
top5_acc = top_k_accuracy_score(y_true, y_pred_proba, k=5)

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
```

## Common Issues and Solutions

### Issue: Overfitting
**Symptoms:** Training acc high, validation acc low

**Solutions:**
- ✅ More data augmentation
- ✅ Increase dropout
- ✅ Add weight decay
- ✅ Reduce model complexity
- ✅ Early stopping
- ✅ Use pre-trained models

### Issue: Underfitting
**Symptoms:** Both training and validation acc low

**Solutions:**
- ✅ Increase model capacity
- ✅ Train longer
- ✅ Reduce regularization
- ✅ Better data quality
- ✅ Learning rate tuning

### Issue: Slow Training
**Solutions:**
- ✅ Use GPU (CUDA)
- ✅ Increase batch size
- ✅ Mixed precision training
- ✅ Smaller model (MobileNet, EfficientNet-B0)
- ✅ Fewer data augmentations

### Issue: Out of Memory (OOM)
**Solutions:**
- ✅ Reduce batch size
- ✅ Smaller model
- ✅ Gradient accumulation
- ✅ Mixed precision (FP16)
- ✅ Smaller image resolution

### Issue: Class Imbalance
**Solutions:**
- ✅ Weighted loss function
- ✅ Oversample minority classes
- ✅ Use focal loss
- ✅ Balanced batch sampling

## Model Comparison

| Model | Parameters | Top-1 Acc | Speed | Memory | Use Case |
|-------|-----------|-----------|-------|--------|----------|
| ResNet-50 | 25M | 76% | ⭐⭐⭐ | ⭐⭐⭐ | General |
| EfficientNet-B0 | 5M | 77% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Efficient |
| EfficientNet-B7 | 66M | 84% | ⭐⭐ | ⭐⭐ | Accuracy |
| MobileNet-V2 | 3.5M | 72% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Mobile |
| ViT-Base | 86M | 81% | ⭐⭐ | ⭐⭐ | Large data |
| ViT-Large | 307M | 85% | ⭐ | ⭐ | SOTA |

## Further Reading

- [ImageNet Classification Paper (AlexNet)](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [ResNet Paper: "Deep Residual Learning"](https://arxiv.org/abs/1512.03385)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [PyTorch Image Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## Next Steps

1. Try **Object Detection** (../../object_detection/) for localization
2. Explore **Semantic Segmentation** (../../semantic_segmentation/) for pixel-level classification
3. Learn **Transfer Learning** techniques in depth
4. Study **Image Augmentation** strategies
5. Deploy models with **TorchServe** or **TF Serving**
