# Neural Style Transfer

This directory contains implementations of neural style transfer for applying artistic styles to content images.

## Overview

Neural style transfer combines the content of one image with the style (textures, colors, patterns) of another image to create artistic renditions.

```
Content Image + Style Image → Neural Style Transfer → Stylized Output
(photo)         (painting)                            (photo in painting style)
```

## Core Concept

The key idea is to optimize an image to simultaneously:
1. Match the **content** of the content image
2. Match the **style** of the style image

### Loss Function
```python
total_loss = α × content_loss + β × style_loss + γ × tv_loss

where:
  α = content weight (typically 1)
  β = style weight (typically 1e6)
  γ = total variation weight (typically 1e-6)
```

## Content Representation

Uses high-level features from deep layers of VGG19:

```python
content_layers = ['conv4_2']  # Single deep layer

# Content loss: Euclidean distance between feature maps
content_loss = MSE(VGG_conv4_2(target), VGG_conv4_2(content))
```

**Why deep layers?**
- Capture high-level content structure
- Invariant to exact pixel values
- Preserve spatial layout

## Style Representation

Uses correlations between features (Gram matrices) from multiple layers:

```python
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Gram matrix: correlations between feature maps
G_ij = Σ_k (F_ik × F_jk)

# Style loss: Distance between Gram matrices
style_loss = Σ_layer MSE(Gram(target), Gram(style))
```

### Gram Matrix

The Gram matrix captures style by measuring correlations between feature channels:

```
Feature maps [C × H × W] → Reshape to [C × (H×W)]
                         ↓
Gram matrix = Features × Features^T
             ↓
           [C × C] matrix of correlations
```

**What it captures:**
- Textures (which patterns co-occur)
- Colors (which colors appear together)
- Local patterns (spatial relationships)
- **Not** specific spatial locations (style is location-independent)

## Algorithm

### Gatys et al. Method (Optimization-based)

```python
1. Initialize target image (typically from content image)
2. For each iteration:
   a. Extract features from target, content, style
   b. Compute content loss
   c. Compute style loss (from Gram matrices)
   d. Compute total variation loss (smoothness)
   e. Backpropagate and update target image
3. Return optimized target image
```

### Optimizer
```python
# L-BFGS works well for image optimization
optimizer = optim.LBFGS([target_image])

# Alternative: Adam
optimizer = optim.Adam([target_image], lr=0.01)
```

## Loss Components

### 1. Content Loss
```python
# Preserve high-level structure
loss = ||φ(target) - φ(content)||²

where φ = VGG features at layer conv4_2
```

### 2. Style Loss
```python
# Match texture statistics
loss = Σ_l ||G(φ_l(target)) - G(φ_l(style))||²

where G = Gram matrix
      l = layers conv1_1, conv2_1, ..., conv5_1
```

### 3. Total Variation Loss
```python
# Encourage spatial smoothness (reduce noise)
TV(x) = Σ |x_{i+1,j} - x_{i,j}| + |x_{i,j+1} - x_{i,j}|
```

## Usage

```bash
# Basic style transfer
python train_pytorch.py

# Custom images
python train_pytorch.py --content_path photo.jpg --style_path painting.jpg

# Adjust style strength
python train_pytorch.py --style_weight 1000000  # More style
python train_pytorch.py --style_weight 100000   # Less style
```

## Configuration

```python
CONFIG = {
    'image_size': 512,
    'style_weight': 1000000,   # β: Higher = more stylized
    'content_weight': 1,       # α: Higher = preserve content more
    'tv_weight': 1e-6,         # γ: Higher = smoother result
    'num_steps': 300,          # Optimization steps
    'learning_rate': 0.01,
}
```

## Tuning Weights

### High Style Weight (β >> α)
```python
style_weight = 10000000
# Result: Strong artistic style, less content preservation
# Use for: Artistic effects, abstract art
```

### Balanced Weights
```python
style_weight = 1000000
content_weight = 1
# Result: Good balance of style and content
# Use for: General purpose stylization
```

### High Content Weight (α >> β)
```python
content_weight = 10
style_weight = 100000
# Result: Strong content preservation, subtle style
# Use for: Subtle stylization, photo enhancement
```

## Applications

### Artistic Filters
- Apply famous painting styles to photos
- Social media filters
- Creative photography

### Video Stylization
- Apply style to video frames
- Maintain temporal consistency
- Real-time style transfer (with fast models)

### Photo Enhancement
- Artistic photo editing
- Instagram-style filters
- Creative effects

### Commercial Use
- Product visualization
- Advertising
- Design tools

## Variants

### Fast Neural Style Transfer
```python
# Train feed-forward network to apply style
# Pros: Real-time inference, single forward pass
# Cons: One network per style

model = StyleTransferNet()
stylized = model(content_image)  # Instant!
```

### Arbitrary Style Transfer (AdaIN)
```python
# Transfer any style without retraining
# Uses Adaptive Instance Normalization

output = decoder(AdaIN(content_features, style_features))
```

### Multi-Style Transfer
```python
# Learn multiple styles in single network
# Use style embeddings or conditional networks
```

### Photorealistic Style Transfer
```python
# Preserve photorealism while transferring style
# Add photorealism loss term
```

## Comparison of Methods

| Method | Speed | Quality | Flexibility | Training |
|--------|-------|---------|-------------|----------|
| Gatys (optimization) | Slow (minutes) | Excellent | Any style | No training |
| Fast NST | Fast (real-time) | Good | One style/model | Per-style training |
| AdaIN | Fast | Good | Any style | Single training |
| WCT | Medium | Excellent | Any style | Single training |

## Common Issues

### 1. Blurry Results
- **Cause**: Too much total variation loss
- **Solution**: Reduce tv_weight

### 2. Too Much Noise
- **Cause**: Too little total variation loss
- **Solution**: Increase tv_weight

### 3. Content Lost
- **Cause**: Style weight too high
- **Solution**: Reduce style_weight or increase content_weight

### 4. Style Not Strong Enough
- **Cause**: Content weight too high
- **Solution**: Increase style_weight

### 5. Slow Convergence
- **Cause**: Poor initialization, wrong optimizer
- **Solution**: Initialize from content image, use L-BFGS

## Best Practices

1. **Initialize from Content**: Start target image as content image
2. **Use L-BFGS**: Better convergence than Adam for this task
3. **Balance Weights**: Start with 1M style weight, adjust as needed
4. **Multiple Scales**: Process at multiple resolutions for better quality
5. **Pre-trained VGG**: Always use ImageNet pre-trained VGG19
6. **Normalize Input**: Use VGG normalization (mean, std)

## Advanced Techniques

### Color Preservation
```python
# Preserve original colors, transfer only texture
# Convert to luminance-chrominance space
```

### Semantic Style Transfer
```python
# Match style based on semantic regions
# Apply different styles to different objects
```

### 3D Style Transfer
```python
# Apply style to 3D models
# Maintain geometric consistency
```

## Requirements

```bash
pip install torch torchvision Pillow numpy matplotlib
```

## References

- Original Paper: [Gatys et al., 2015](https://arxiv.org/abs/1508.06576)
- Fast NST: [Johnson et al., 2016](https://arxiv.org/abs/1603.08155)
- AdaIN: [Huang & Belongie, 2017](https://arxiv.org/abs/1703.06868)
- Tutorial: [PyTorch Official](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
