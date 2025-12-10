# Quick Reference: Image Classification

## 🚀 Run the Examples

```bash
# Single-label classification (basic)
cd models/3_computer_vision/classification/single_label
python train_pytorch.py

# Single-label (advanced architectures)
python train_pytorch_v2.py

# Single-label (TensorFlow/Keras)
python train_tensorflow.py

# Multi-label classification
cd ../multi_label
python train_pytorch.py
```

## 📋 What's Implemented

### ✅ Single-Label Classification
- **train_pytorch.py** - ResNet-50, EfficientNet, ViT with transfer learning
- **train_pytorch_v2.py** - ConvNeXt, Swin, RegNet with advanced techniques
- **train_tensorflow.py** - ResNet50, EfficientNetB0, MobileNetV3 with Keras

### ✅ Multi-Label Classification
- **train_pytorch.py** - Multiple labels per image with BCE loss

## 🎯 Choose Your Path

| If You Want... | Use This File | Why |
|----------------|---------------|-----|
| **Quick start** | `train_pytorch.py` | Simple ResNet-50 example on CIFAR-10 |
| **Best accuracy** | `train_pytorch_v2.py` | Modern architectures (Swin, ConvNeXt) |
| **Production deployment** | `train_tensorflow.py` | Easy export to TFLite, TF.js |
| **Multiple labels** | `multi_label/train_pytorch.py` | Images with multiple tags |

## 🏗️ Architecture Quick Pick

**Need speed?** → RegNet, EfficientNet
**Need accuracy?** → Swin Transformer, ViT
**Need balance?** → ResNet-50, ConvNeXt
**Need mobile?** → MobileNetV3, EfficientNet-B0

## 💡 Key Code Snippets

### Load Pretrained Model
```python
import torchvision.models as models

# PyTorch
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# TensorFlow/Keras
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False)
```

### Training Loop
```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### Multi-Label Prediction
```python
# Use BCE loss and sigmoid
criterion = nn.BCEWithLogitsLoss()
outputs = model(inputs)  # Raw logits
probs = torch.sigmoid(outputs)  # Convert to probabilities
predictions = (probs > 0.5).float()  # Apply threshold
```

## 🎓 Learning Progression

1. **Start Here**: Run `train_pytorch.py` to understand basics
2. **Level Up**: Try `train_pytorch_v2.py` for modern architectures
3. **Compare**: Run `train_tensorflow.py` to see Keras approach
4. **Specialize**: Use `multi_label/train_pytorch.py` for multi-label tasks

## 📊 Expected Results

All examples use small training runs for demonstration:

- **CIFAR-10 (train_pytorch.py)**: ~70% accuracy in 5 epochs
- **Advanced models**: Better accuracy with proper training
- **Multi-label**: Metrics include hamming loss, subset accuracy, F1

## 🔧 Common Modifications

### Change Dataset
```python
# Replace with your data
train_data = ImageFolder('path/to/train', transform=transform)
```

### Adjust Training
```python
# Longer training
num_epochs = 50

# Better optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Add scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
```

### Export Model
```python
# PyTorch → ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# TensorFlow → TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

## 🐛 Troubleshooting

**Out of memory?**
- Reduce batch size: `batch_size = 16`
- Use mixed precision: `from torch.cuda.amp import autocast`

**Model not learning?**
- Check learning rate (try 0.001, 0.0001)
- Verify data normalization
- Ensure model is in train mode: `model.train()`

**Low accuracy?**
- Train for more epochs
- Add data augmentation
- Try fine-tuning (unfreeze base layers)

## 📚 Next Steps

After classification, explore:
- **Object Detection**: Detect and localize objects
- **Segmentation**: Pixel-level classification
- **Few-Shot Learning**: Learn from few examples

## ✨ Summary

This directory contains **4 complete implementations** covering:
- Single-label classification (3 approaches)
- Multi-label classification
- 10+ model architectures
- PyTorch and TensorFlow/Keras
- Transfer learning and fine-tuning
- Production-ready model export

All examples are **self-contained** and **ready to run**!
