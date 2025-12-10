# Image Classification Models

Comprehensive implementations of image classification using PyTorch and TensorFlow/Keras.

## 📁 Directory Structure

```
classification/
├── single_label/
│   ├── train_pytorch.py          # ResNet, EfficientNet, ViT with transfer learning
│   ├── train_pytorch_v2.py       # ConvNeXt, Swin, RegNet with advanced techniques
│   └── train_tensorflow.py       # Keras models with callbacks and fine-tuning
├── multi_label/
│   └── train_pytorch.py          # Multi-label classification with BCE loss
└── README.md                     # This file
```

## 🎯 What's Covered

### Single-Label Classification
- **Standard Architectures**: ResNet-50, EfficientNet-B0, ViT-B/16
- **Advanced Architectures**: ConvNeXt, Swin Transformer, RegNet
- **Frameworks**: PyTorch (standard + advanced), TensorFlow/Keras
- **Techniques**: Transfer learning, fine-tuning, data augmentation, mixed precision

### Multi-Label Classification
- **Task**: Predict multiple labels per image simultaneously
- **Loss Function**: Binary Cross-Entropy (BCE) instead of Cross-Entropy
- **Activation**: Sigmoid instead of Softmax
- **Metrics**: Hamming loss, subset accuracy, per-class F1
- **Advanced**: Weighted loss, focal loss, threshold optimization

## 🚀 Quick Start

### Single-Label (PyTorch)

```bash
cd single_label
python train_pytorch.py        # Basic ResNet-50 on CIFAR-10
```

**What you'll learn:**
- Transfer learning from ImageNet
- Training loop with validation
- EfficientNet and ViT alternatives
- Model evaluation and export

### Advanced Single-Label (PyTorch)

```bash
python train_pytorch_v2.py     # Modern architectures
```

**What you'll learn:**
- ConvNeXt (modern ConvNet)
- Swin Transformer (hierarchical vision transformer)
- Mixed precision training (faster + less memory)
- Advanced augmentation (RandAugment, AutoAugment)
- Learning rate schedules (Cosine, OneCycle)
- Model export (ONNX, TorchScript)

### Single-Label (TensorFlow/Keras)

```bash
python train_tensorflow.py     # Keras models
```

**What you'll learn:**
- Keras Sequential and Functional APIs
- Built-in callbacks (EarlyStopping, ModelCheckpoint)
- Fine-tuning strategy (freeze → unfreeze)
- Model export (SavedModel, TFLite, TF.js)

### Multi-Label (PyTorch)

```bash
cd multi_label
python train_pytorch.py        # Multi-label classification
```

**What you'll learn:**
- BCE loss for multiple labels
- Sigmoid activation
- Multi-label metrics
- Threshold optimization
- Class imbalance handling

## 📊 Architecture Comparison

### PyTorch Models

| Model | Parameters | ImageNet Top-1 | Speed | Best For |
|-------|-----------|----------------|-------|----------|
| ResNet-50 | 25.6M | 76.1% | Fast | General purpose, baseline |
| EfficientNet-B0 | 5.3M | 77.1% | Medium | Efficiency, mobile deployment |
| ViT-B/16 | 86.6M | 81.1% | Slow | Large datasets, high accuracy |
| ConvNeXt-Tiny | 28.6M | 82.1% | Medium | Modern ConvNet design |
| Swin-Tiny | 28.3M | 81.3% | Medium | Hierarchical features |
| RegNet-Y-400MF | 4.3M | 74.0% | Fast | Efficiency, speed |

### TensorFlow/Keras Models

| Model | Parameters | ImageNet Top-1 | Best For |
|-------|-----------|----------------|----------|
| ResNet50 | 25.6M | 74.9% | General purpose |
| EfficientNetB0 | 5.3M | 77.1% | Efficiency |
| MobileNetV3Large | 5.5M | 75.6% | Mobile devices |
| InceptionV3 | 23.9M | 77.9% | Multi-scale features |

## 🎓 Learning Path

### Beginner
1. **Start**: `single_label/train_pytorch.py`
   - Understand basic training loop
   - Learn transfer learning
   - Try ResNet-50 on CIFAR-10

### Intermediate
2. **Explore**: `single_label/train_pytorch_v2.py`
   - Modern architectures (ConvNeXt, Swin)
   - Mixed precision training
   - Advanced data augmentation

3. **Alternative**: `single_label/train_tensorflow.py`
   - Compare PyTorch vs TensorFlow/Keras
   - Use built-in callbacks
   - Explore different export formats

### Advanced
4. **Specialize**: `multi_label/train_pytorch.py`
   - Multi-label problems
   - BCE loss and sigmoid
   - Threshold optimization
   - Class imbalance strategies

## 💡 Key Concepts

### Transfer Learning
```python
# Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### Fine-Tuning Strategy
```python
# Phase 1: Train only new layers (5-10 epochs)
# ... train with frozen base ...

# Phase 2: Unfreeze and train all with lower LR
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# ... continue training ...
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    with autocast():  # Auto mixed precision
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Multi-Label vs Single-Label

| Aspect | Single-Label | Multi-Label |
|--------|--------------|-------------|
| **Task** | One class per image | Multiple classes per image |
| **Loss** | CrossEntropyLoss | BCEWithLogitsLoss |
| **Activation** | Softmax | Sigmoid |
| **Output** | Probabilities sum to 1 | Independent probabilities |
| **Example** | Cat OR Dog | Cat AND Dog |

## 🔧 Common Tasks

### Change Dataset
```python
# Replace CIFAR-10 with your dataset
from torchvision.datasets import ImageFolder

train_data = ImageFolder(
    root='path/to/train',
    transform=train_transform
)
```

### Adjust Learning Rate
```python
# Option 1: Fixed LR
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Option 2: Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)

# Option 3: OneCycle
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=len(train_loader),
    epochs=num_epochs
)
```

### Add Data Augmentation
```python
# PyTorch
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(),  # Advanced augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# TensorFlow/Keras
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])
```

### Export Models

**PyTorch → ONNX:**
```python
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

**PyTorch → TorchScript:**
```python
scripted = torch.jit.script(model)
scripted.save("model.pt")
```

**TensorFlow → SavedModel:**
```python
model.save('saved_model/')
```

**TensorFlow → TFLite:**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

## 📈 Performance Tips

### Speed Up Training
1. **Use mixed precision**: 40-50% faster on modern GPUs
2. **Increase batch size**: Better GPU utilization
3. **Use DataLoader workers**: `num_workers=4` for faster data loading
4. **Pin memory**: `pin_memory=True` for faster CPU→GPU transfer

### Improve Accuracy
1. **More data augmentation**: RandAugment, AutoAugment, Mixup
2. **Better optimizer**: AdamW with weight decay
3. **Learning rate schedule**: Cosine annealing or OneCycle
4. **Label smoothing**: Prevents overconfident predictions
5. **Fine-tuning**: Unfreeze base model after initial training

### Reduce Memory Usage
1. **Smaller batch size**: Trade speed for memory
2. **Gradient accumulation**: Simulate larger batches
3. **Mixed precision**: Uses 16-bit instead of 32-bit
4. **Smaller model**: EfficientNet or RegNet instead of ViT

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Or use gradient accumulation
accumulation_steps = 2
for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Poor Validation Accuracy
```python
# Check for data leakage
# Ensure train/val split is correct

# Add more augmentation
# Try different learning rates
# Train for more epochs

# Verify model is in eval mode
model.eval()
with torch.no_grad():
    # ... validation ...
```

### Model Not Learning
```python
# Check learning rate
# Verify loss is decreasing
# Ensure gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")

# Check data normalization
# Verify labels are correct format
```

## 📚 Next Steps

After mastering classification:
1. **Object Detection**: Detect multiple objects with bounding boxes
2. **Semantic Segmentation**: Classify every pixel in image
3. **Instance Segmentation**: Combine detection + segmentation
4. **Few-Shot Learning**: Learn from few examples
5. **Self-Supervised Learning**: Pre-training without labels

## 🔗 Resources

### Papers
- **ResNet**: "Deep Residual Learning for Image Recognition" (2015)
- **EfficientNet**: "EfficientNet: Rethinking Model Scaling" (2019)
- **ViT**: "An Image is Worth 16x16 Words" (2020)
- **ConvNeXt**: "A ConvNet for the 2020s" (2022)
- **Swin**: "Swin Transformer: Hierarchical Vision Transformer" (2021)

### Documentation
- **PyTorch Vision Models**: https://pytorch.org/vision/stable/models.html
- **TensorFlow Keras Applications**: https://keras.io/api/applications/
- **timm Library**: https://github.com/huggingface/pytorch-image-models

### Datasets
- **CIFAR-10/100**: Small-scale for quick experiments
- **ImageNet**: Large-scale benchmark (1000 classes, 1.2M images)
- **Food-101**: Food classification (101 classes)
- **Stanford Dogs**: Fine-grained (120 dog breeds)
- **iNaturalist**: Biodiversity (10K+ species)

## 🎯 Summary

This directory provides **complete, runnable implementations** of image classification:

✅ **Single-Label**: PyTorch (basic + advanced) and TensorFlow/Keras
✅ **Multi-Label**: BCE loss with sigmoid activation
✅ **6+ Architectures**: From ResNet to Swin Transformer
✅ **Transfer Learning**: Pretrained weights from ImageNet
✅ **Advanced Techniques**: Mixed precision, augmentation, scheduling
✅ **Model Export**: ONNX, TorchScript, TFLite, SavedModel
✅ **Production Ready**: Evaluation, metrics, deployment formats

Each file is **self-contained** and can be run independently. Start with `single_label/train_pytorch.py` for the basics, then explore advanced techniques and frameworks.
