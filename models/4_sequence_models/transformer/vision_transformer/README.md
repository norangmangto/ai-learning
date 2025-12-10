# Vision Transformer (ViT)

Transformer architecture adapted for image classification via patch embedding.

## 📋 Overview

**Architecture:** Image patching + Transformer encoder
**Input:** Images (any resolution)
**Output:** Class logits
**Best For:** Image classification, transfer learning

## 🏗️ Architecture

```
Image (H × W × C)
    ↓
Patch Embedding (16×16 patches)
    ↓
[CLS token] + [Patch embeddings] + [Position embeddings]
    ↓
Transformer Encoder (12-24 layers)
    ↓
[CLS token output] → MLP Head
    ↓
Class logits
```

## 🎯 Key Insight

Instead of convolutions, split image into patches and treat as sequence!

```
256×256 image with 16×16 patches = 256 patches
Like text: "This is a cat" → [This, is, a, cat]
Like image: [patch_1, patch_2, ..., patch_256]
```

## 📐 Patch Embedding

### Process

```python
# Input image: (3, 224, 224)
# Patch size: 16×16

patches = unfold(image, patch_size=16)
# Output: (49, 768)  # 14×14=196 patches (in typical ViT)
```

### Formula
```
P = (H / patch_size) × (W / patch_size)
D = C × patch_size²

For ViT-Base (224×224, 16×16 patches):
P = 14 × 14 = 196 patches
D = 3 × 16² = 768 dimensions
```

## 🧠 Architecture Components

### 1. Patch Embedding Layer
```
[Image] → Conv(kernel=16, stride=16) → [Embedded patches]
```
Each patch becomes a high-dimensional vector.

### 2. Position Embedding
```
[CLS] + [Patch1] + [Patch2] + ... + [Patch196]
  +        +          +              +
[Pos0] + [Pos1] + [Pos2] + ... + [Pos196]
```
Tells transformer which patches are adjacent.

### 3. Transformer Encoder
```
Input: [CLS, P1, P2, ..., P196]
  ↓
Multi-head self-attention (cross-patch attention)
  ↓
Each patch attends to all other patches!
  ↓
Output: [CLS, P1', P2', ..., P196']
```

### 4. Classification Head
```
[CLS] token (special token)
  ↓
Linear → Softmax
  ↓
Class probabilities
```

## 🔍 Self-Attention in Vision

```
Query each patch: "What other patches are relevant?"
  ↓
Attention weights show spatial relationships:
- Patches of the same object → High attention
- Distant patches → Lower attention

Example: Cat detection
- Ear patch attends to: eye, face, other ear patches
- Whisker patch attends to: mouth, face patches
- Tail patch might attend to: body patches
```

## 📊 Information Flow

```
Image: [background, cat_head, cat_body, cat_tail]
         ↓
Patches: [P1, P2, P3, ..., P196]
         ↓
Layer 1: Each patch learns simple features
         P1 sees: nearby edge patterns

Layer 6: Mid-level representations
         P1, P2 interact heavily (connected object)

Layer 12: High-level semantics
         "cat" emerges from collective patch understanding
```

## 🚀 Quick Start

```python
from train_pytorch import VisionTransformer
import torch

# Create model
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    d_model=768,
    num_heads=12,
    num_layers=12,
    d_ff=3072,
    dropout=0.1
)

# Forward pass
images = torch.randn(32, 3, 224, 224)  # Batch of 32 images
logits = model(images)  # (32, 1000)
predictions = logits.argmax(dim=1)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for images, labels in train_loader:
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 📈 Applications

| Task | Use Case |
|------|----------|
| **Classification** | ImageNet, CIFAR-10 |
| **Detection** | Object detection (+ decoder) |
| **Segmentation** | Semantic, instance segmentation |
| **Transfer Learning** | Pretrain on ImageNet, finetune |
| **Multimodal** | Vision-language (+ text encoder) |

## ✨ ViT Variants

| Variant | Size | Params | Speed | Accuracy |
|---------|------|--------|-------|----------|
| ViT-Tiny | 192 | 5M | ⚡⚡⚡ | 72% |
| ViT-Small | 384 | 22M | ⚡⚡ | 81% |
| ViT-Base | 768 | 86M | ⚡ | 84% |
| ViT-Large | 1024 | 307M | 🐌 | 87% |
| ViT-Huge | 1280 | 632M | 🐢 | 88% |

## 💡 Why ViT Works

```
CNN advantage: Local receptive field (inductive bias)
             → Good for small images

ViT advantage: Global self-attention
             → Learns what to attend to
             → Better with lots of data
             → More scalable

Trade-off:
- ViTs need more data than CNNs
- ViTs scale better (can be huge)
- ViTs are more interpretable (attention maps)
```

## 🎓 Position Embedding Visualization

```
Without position embedding:
[patch_1, patch_2, patch_3] = [patch_3, patch_1, patch_2]
(Model doesn't know spatial order!)

With position embedding:
[patch_1 + pos_1, patch_2 + pos_2, patch_3 + pos_3]
(Model knows spatial locations)
```

## 📊 Attention Visualization

```
Image: Cat on floor
       [Head] [Body] [Tail]
         P1    P2     P3

Layer 1 (low-level):
P1 → mostly attends to P1 (local features)
P2 → mostly attends to P2
P3 → mostly attends to P3

Layer 6 (mid-level):
P1 → also attends to P2 (connected object)
P2 ↔ P1, P3 (part of same animal)
P3 → attends to P1, P2 (connected)

Layer 12 (semantic):
All patches share information
Collective "cat" understanding
```

## ⚠️ Key Differences from CNNs

| Aspect | CNN | ViT |
|--------|-----|-----|
| **Receptive field** | Grows with layers | Global from start |
| **Inductive bias** | Locality, translation | None (learned) |
| **Data needed** | Medium (1M images) | High (14M+ images) |
| **Interpretability** | Learned filters | Attention maps |
| **Efficiency** | Fast (local ops) | Slower (global ops) |
| **Scalability** | Limited | Excellent |

## 🔄 Training Tips

1. **Large datasets preferred**
   - ViT-Base needs 14M+ images (ImageNet-21K)
   - Smaller datasets: use pretrained ViT

2. **Patch size matters**
   - Smaller patches: more tokens (slower, better detail)
   - Larger patches: fewer tokens (faster, less detail)
   - Default 16×16 is usually optimal

3. **Resolution handling**
   - ViTs can handle variable resolutions
   - Standard training: 224×224
   - Higher resolution fine-tuning: 384×384, 512×512

4. **Computational cost**
   - Attention: O(n²) where n = number of patches
   - 196 patches = manageable
   - 1024 patches = slow (unless efficient attention)

## 🎓 Learning Outcomes

- [x] Image-to-sequence transformation (patching)
- [x] Position embeddings for spatial information
- [x] Self-attention on image regions
- [x] Why ViTs outperform CNNs at scale
- [x] Attention visualization for interpretability

## 📚 Key Papers

- **ViT**: "An Image Is Worth 16×16 Words" (Dosovitskiy et al., 2020)
- **DeiT**: "Training Data-Efficient ViTs" (Touvron et al., 2021)
- **Swin**: "Shifted Windows ViT" (Liu et al., 2021)

## 📊 ViT vs CNN Performance

```
ImageNet Accuracy (100 epoch training):

ViT-B + 14M image pretrain:    84.6%
ResNet-50:                     76.1%
ResNet-101:                    79.8%

ViT advantage grows with:
- More training data
- Larger model scale
- Transfer learning tasks
```

## 💪 Advantages

✅ **Scalability** - Works with massive models (1B+ params)
✅ **Interpretability** - Attention maps show what it sees
✅ **Versatility** - Same architecture for many tasks
✅ **Transfer learning** - Excellent pretrained models

## 🚨 Disadvantages

❌ **Data hungry** - Needs lots of images
❌ **Slower inference** - Quadratic attention complexity
❌ **Less inductive bias** - Need more parameters for small data
❌ **Higher latency** - For real-time applications

---

**Last Updated:** December 2024
**Status:** ✅ Complete
