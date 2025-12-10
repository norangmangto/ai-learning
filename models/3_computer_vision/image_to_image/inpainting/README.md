# Image Inpainting

This directory contains implementations of image inpainting using partial convolutions and U-Net architecture.

## Overview

Image inpainting fills in missing or corrupted regions in images with plausible content that matches the surrounding context.

## Problem Definition

**Input**: Image with missing regions (holes/masks)
**Output**: Complete image with holes filled in naturally

```
[Image with hole] → [Inpainting Model] → [Complete image]
```

## Key Innovation: Partial Convolutions

Traditional convolutions treat masked and valid pixels equally. Partial convolutions solve this:

### Standard Convolution
```python
# Treats all pixels equally (including invalid ones)
output = conv(input * mask)  # ❌ Contaminated by masked regions
```

### Partial Convolution
```python
# Only uses valid pixels, updates mask
output = conv(input * mask) * scaling
new_mask = (mask_conv(mask) > 0)  # ✅ Propagates valid regions only
```

### Key Properties
1. **Mask-aware**: Only convolved with valid pixels
2. **Mask updating**: Automatically updates mask based on valid neighbors
3. **Re-normalization**: Scales output by ratio of valid pixels

## Architecture

```
Input Image + Mask
   ↓
Encoder (Partial Convolutions)
├─ Pool1 (1/2)  ─┐
├─ Pool2 (1/4)  ─┤
├─ Pool3 (1/8)  ─┤  Skip Connections
├─ Pool4 (1/16) ─┤
└─ Pool5 (1/32) ─┘
   ↓
Decoder (Partial Convolutions + Upsampling)
├─ Up1 + Skip5
├─ Up2 + Skip4
├─ Up3 + Skip3
├─ Up4 + Skip2
└─ Up5 + Skip1
   ↓
Output Image
```

## Loss Functions

### 1. Hole Loss (Primary)
```python
# Loss only on masked regions
loss_hole = L1(output * (1 - mask), target * (1 - mask))
weight = 6.0  # Weight hole region more heavily
```

### 2. Valid Loss
```python
# Loss on valid (non-masked) regions
loss_valid = L1(output * mask, target * mask)
weight = 1.0
```

### 3. Total Loss
```python
total_loss = 6 * loss_hole + loss_valid
```

### Optional: Perceptual Loss
```python
# Use VGG features for more realistic textures
loss_perceptual = L1(VGG(output), VGG(target))
```

## Mask Types

### Rectangular Holes
```python
# Simple rectangular regions
mask[y:y+h, x:x+w] = 0
```

### Irregular Holes
```python
# Random strokes and shapes (more realistic)
- Multiple rectangles
- Random brush strokes
- Arbitrary polygons
```

### Free-form Masks
```python
# User-drawn masks for interactive editing
mask = user_drawn_region()
```

## Usage

```bash
# Train inpainting model
python train_pytorch.py

# Different mask ratios
python train_pytorch.py --mask_ratio 0.2  # 20% masked
python train_pytorch.py --mask_ratio 0.5  # 50% masked
```

## Configuration

```python
CONFIG = {
    'image_size': 256,
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.0002,
    'mask_ratio': 0.3,     # Percentage to mask
}
```

## Applications

### Photo Restoration
- Remove scratches and damage from old photos
- Restore missing parts of historical images
- Repair corrupted digital photos

### Object Removal
- Remove unwanted objects from photos
- Clean up tourist photos (remove people)
- Content-aware fill in photo editors

### Image Editing
- Extend image boundaries
- Replace objects with context
- Creative manipulation

### Video Inpainting
- Remove watermarks
- Fix video artifacts
- Stabilization artifacts removal

### Medical Imaging
- Fill missing scan regions
- Artifact removal
- Image enhancement

## Evaluation Metrics

### Pixel-Level Metrics
- **L1 Loss**: Mean absolute error
- **L2 Loss**: Mean squared error
- **PSNR**: Peak Signal-to-Noise Ratio

### Perceptual Metrics
- **SSIM**: Structural similarity
- **Perceptual Loss**: VGG feature distance
- **FID**: Fréchet Inception Distance

### Visual Quality
- User studies
- Visual inspection
- Semantic correctness

## Challenges

### 1. Large Holes
- **Problem**: Hard to infer content from far-away context
- **Solution**: Coarse-to-fine refinement, stronger context modeling

### 2. Semantic Understanding
- **Problem**: Need to understand what should fill the hole
- **Solution**: Use powerful backbones, semantic guidance

### 3. Boundary Artifacts
- **Problem**: Visible seams at mask boundaries
- **Solution**: Smooth transitions, boundary loss

### 4. Color/Texture Matching
- **Problem**: Generated content may not match surroundings
- **Solution**: Perceptual loss, attention mechanisms

## Advanced Techniques

### GAN-based Inpainting
```python
# Use adversarial training for more realistic results
generator = InpaintingGenerator()
discriminator = Discriminator()
loss = adversarial_loss + reconstruction_loss
```

### Attention-based Models
```python
# Attend to relevant context regions
attention = compute_attention(query=hole_features, key=valid_features)
filled = aggregate(attention, valid_features)
```

### Coarse-to-Fine
```python
# First generate rough content, then refine
coarse_result = coarse_network(input)
refined_result = refinement_network(coarse_result)
```

## Comparison with Alternatives

### Traditional Methods
- **Diffusion**: Slow, good for small holes
- **Patch-based**: Fast but limited to repetitive textures
- **Exemplar-based**: Good for textures, poor for semantics

### Deep Learning Methods
- **Partial Convolutions**: Good balance, efficient
- **Gated Convolutions**: More flexible than partial conv
- **GANs**: More realistic but harder to train
- **Transformers**: State-of-the-art but computationally expensive

## Best Practices

1. **Diverse Masks**: Train with various mask shapes and sizes
2. **Progressive Training**: Start with easy examples, increase difficulty
3. **Perceptual Loss**: Improves visual quality
4. **Data Augmentation**: Increases robustness
5. **Multi-scale**: Process at multiple resolutions
6. **Sufficient Context**: Ensure enough valid pixels around holes

## Requirements

```bash
pip install torch torchvision Pillow numpy matplotlib
```

## References

- Partial Convolutions: [Liu et al., 2018](https://arxiv.org/abs/1804.07723)
- Gated Convolutions: [Yu et al., 2019](https://arxiv.org/abs/1806.03589)
- EdgeConnect: [Nazeri et al., 2019](https://arxiv.org/abs/1901.00212)
