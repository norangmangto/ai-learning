# U-Net for Image-to-Image Translation

This directory contains U-Net implementations for various image-to-image translation tasks.

## Overview

U-Net is a versatile encoder-decoder architecture designed for pixel-to-pixel tasks where input and output images have spatial correspondence.

```
Input Image → U-Net → Output Image
(same size)          (same size)
```

## Architecture

### Structure
```
Input (256×256×3)
   ↓
Encoder (Contracting Path)
├─ Conv1 → Pool1 (128×128) ─────────┐
├─ Conv2 → Pool2 (64×64)   ─────┐   │
├─ Conv3 → Pool3 (32×32)  ──┐   │   │
├─ Conv4 → Pool4 (16×16) ┐  │   │   │
│                         │  │   │   │
Bottleneck (8×8)          │  │   │   │
│                         │  │   │   │
Decoder (Expanding Path)  │  │   │   │
├─ Up1 + Skip4 ──────────┘  │   │   │
├─ Up2 + Skip3 ─────────────┘   │   │
├─ Up3 + Skip2 ─────────────────┘   │
└─ Up4 + Skip1 ─────────────────────┘
   ↓
Output (256×256×3)
```

### Key Features

#### 1. Encoder-Decoder Structure
```python
# Encoder: Capture context
- Progressively downsample
- Increase feature channels
- Extract hierarchical features

# Decoder: Precise localization
- Progressively upsample
- Decrease feature channels
- Reconstruct spatial details
```

#### 2. Skip Connections
```python
# Connect encoder to decoder at same resolution
decoder_features = concat([upsampled_features, encoder_features])

# Benefits:
- Preserve fine-grained details
- Better gradient flow
- Combine low-level and high-level features
```

#### 3. Symmetric Architecture
```python
# Mirror structure
- 4 down-sampling layers
- 4 up-sampling layers
- Matching feature dimensions
```

## Supported Tasks

### 1. Image Colorization
```python
Input: Grayscale image (L channel)
Output: Color image (RGB)
Task: Add color to black-and-white photos
```

### 2. Image Denoising
```python
Input: Noisy image
Output: Clean image
Task: Remove noise while preserving details
```

### 3. Image Enhancement
```python
Input: Low-quality image
Output: Enhanced image
Task: Improve overall quality, lighting, sharpness
```

### 4. Segmentation
```python
Input: Image
Output: Segmentation mask
Task: Pixel-wise classification
```

### 5. Sketch-to-Image
```python
Input: Line drawing
Output: Realistic image
Task: Generate full image from sketch
```

## Usage

```bash
# Train for colorization
python train_pytorch.py --task colorization

# Train for denoising
python train_pytorch.py --task denoising

# Train for enhancement
python train_pytorch.py --task enhancement
```

## Configuration

```python
CONFIG = {
    'task': 'colorization',  # 'colorization', 'denoising', 'enhancement'
    'image_size': 256,
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 0.0002,
    'in_channels': 1,       # 1 for gray, 3 for RGB
    'out_channels': 3,      # 3 for RGB
}
```

## Implementation Details

### Encoder Block
```python
def encoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
```

### Decoder Block
```python
def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
```

### Skip Connection
```python
# Concatenate encoder features with upsampled decoder features
up = upsample(decoder_features)
concat = torch.cat([up, encoder_features], dim=1)
decoder_features = decoder_block(concat)
```

## Task-Specific Details

### Colorization

**Input Processing**:
```python
# Convert RGB to LAB color space
lab = rgb_to_lab(image)
L = lab[:, :, 0]  # Luminance (grayscale)
ab = lab[:, :, 1:]  # Color channels

# Model input: L channel
# Model output: ab channels
# Final: Combine L + predicted_ab → RGB
```

**Why LAB?**
- Separates brightness (L) from color (ab)
- More natural color space for this task
- Easier to learn color without affecting brightness

### Denoising

**Noise Types**:
```python
# Gaussian noise
noisy = clean + gaussian_noise(mean=0, std=0.1)

# Salt-and-pepper noise
noisy = add_salt_pepper(clean, prob=0.05)

# Speckle noise (multiplicative)
noisy = clean + clean * gaussian_noise()
```

**Loss Function**:
```python
# L1 or L2 loss
loss = MSE(denoised, clean)  # L2
loss = L1(denoised, clean)   # L1 (less blurry)
```

### Enhancement

**Degradations**:
```python
# Simulate low-quality images
degraded = apply_random_degradation(image):
    - Lower brightness
    - Reduce contrast
    - Add blur
    - Add noise
    - Reduce saturation
```

**Multi-task Loss**:
```python
loss = pixel_loss + perceptual_loss + color_loss
```

## Loss Functions

### Pixel Loss
```python
# L1 or L2 between pixels
loss = L1(output, target)  # Preferred for image tasks
loss = MSE(output, target)  # Can be blurry
```

### Perceptual Loss
```python
# Compare VGG features
loss = L1(VGG(output), VGG(target))
# Better visual quality
```

### Adversarial Loss
```python
# Use discriminator for realistic outputs
loss_G = adversarial_loss + λ × reconstruction_loss
# Sharper results
```

## Evaluation Metrics

### Quantitative
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **FID**: Fréchet Inception Distance

### Qualitative
- Visual inspection
- User studies
- Comparison with ground truth

## Applications

### Photo Editing
- Colorize old black-and-white photos
- Remove noise from photos
- Enhance low-quality images
- Restore damaged photos

### Medical Imaging
- Denoise medical scans
- Enhance low-dose CT images
- Stain normalization
- Super-resolution for MRI

### Satellite Imagery
- Cloud removal
- Image enhancement
- Multi-spectral fusion

### Art and Design
- Sketch-to-photo generation
- Style transfer
- Texture synthesis

## Advantages of U-Net

1. **Skip Connections**: Preserve fine details
2. **Few Training Samples**: Works with small datasets
3. **Flexible**: Easily adapted to different tasks
4. **Fast**: Efficient inference
5. **End-to-End**: Single network for entire task

## Limitations

1. **Fixed Resolution**: Input/output must be same size
2. **Memory**: Skip connections increase memory usage
3. **No Multi-Scale**: Processes single resolution
4. **Blurry**: May produce smooth outputs (use GAN for sharper)

## Variants

### U-Net++
```python
# Nested skip connections
# Better feature fusion
# Improved performance
```

### Attention U-Net
```python
# Attention gates in skip connections
# Focus on relevant features
# Better for complex tasks
```

### Residual U-Net
```python
# Add residual connections
# Deeper networks
# Better gradient flow
```

### Dense U-Net
```python
# Dense connections like DenseNet
# Feature reuse
# Parameter efficient
```

## Best Practices

1. **Use L1 Loss**: Less blurry than L2
2. **Add Perceptual Loss**: Better visual quality
3. **Batch Normalization**: Stabilizes training
4. **Data Augmentation**: Flips, rotations, crops
5. **Progressive Training**: Start small, increase size
6. **Learning Rate Schedule**: Cosine or step decay
7. **Gradient Clipping**: Prevents exploding gradients

## Advanced Techniques

### Multi-Scale U-Net
```python
# Process multiple scales
# Better handling of details
```

### Conditional U-Net
```python
# Add conditioning information
# Control generation process
```

### 3D U-Net
```python
# Extend to volumetric data
# Medical imaging, video
```

## Requirements

```bash
pip install torch torchvision Pillow numpy scikit-image matplotlib
```

## References

- Original U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- U-Net++: [Zhou et al., 2018](https://arxiv.org/abs/1807.10165)
- Attention U-Net: [Oktay et al., 2018](https://arxiv.org/abs/1804.03999)
- Colorization: [Zhang et al., 2016](https://arxiv.org/abs/1603.08511)
