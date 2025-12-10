# Super Resolution

This directory contains implementations of single-image super resolution (SISR) using deep learning.

## Overview

Super resolution reconstructs high-resolution (HR) images from low-resolution (LR) inputs, increasing image quality and detail.

```
Low Resolution (64×64) → Super Resolution Model → High Resolution (256×256)
                                                  (4× upscaling)
```

## Problem Definition

**Input**: Low-resolution image (H×W)
**Output**: High-resolution image (scale×H × scale×W)

Common scales: 2×, 4×, 8×

## Architecture: SRResNet

### Overall Structure
```
LR Image
   ↓
Initial Conv (64 filters)
   ↓
Residual Blocks (16 blocks)
├─ Conv → BN → PReLU
├─ Conv → BN
└─ Skip connection
   ↓
Post-residual Conv
   ↓
Upsampling Blocks (2× for 4× SR)
├─ Conv
├─ Pixel Shuffle (2× upscale)
└─ PReLU
   ↓
Final Conv (3 channels)
   ↓
HR Image
```

### Key Components

#### 1. Residual Learning
```python
# Each residual block learns residual function
output = input + residual_function(input)

# Benefits:
- Easier optimization
- Deeper networks
- Better gradient flow
```

#### 2. Sub-Pixel Convolution (Pixel Shuffle)
```python
# Rearrange depth into space
# More efficient than bilinear/bicubic interpolation

Input: [B, C×r², H, W]
  ↓ Pixel Shuffle
Output: [B, C, r×H, r×W]

where r = upscaling factor
```

**Why pixel shuffle?**
- Learns upsampling (vs. fixed interpolation)
- Computationally efficient
- Better quality than deconvolution

#### 3. Batch Normalization + PReLU
```python
# Batch norm: Stabilizes training
# PReLU: Learnable activation with negative slope
```

## Loss Functions

### 1. Pixel Loss (MSE)
```python
# L2 loss between pixels
loss = MSE(SR_image, HR_image)

# Pros: Simple, fast convergence
# Cons: Can produce blurry results
```

### 2. Perceptual Loss (VGG)
```python
# Compare VGG features instead of pixels
loss = MSE(VGG(SR_image), VGG(HR_image))

# Pros: Better perceptual quality
# Cons: More computation
```

### 3. Adversarial Loss (SRGAN)
```python
# Use discriminator to encourage realistic textures
loss_G = adversarial_loss + λ × content_loss

# Pros: Sharp, realistic details
# Cons: May hallucinate details, harder to train
```

## Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)
```python
# Higher is better, measured in dB
PSNR = 20 × log10(MAX / √MSE)

# Common values:
- 20-25 dB: Low quality
- 25-30 dB: Acceptable
- 30-35 dB: Good
- 35+ dB: Excellent
```

### SSIM (Structural Similarity Index)
```python
# Measures structural similarity (0 to 1)
SSIM = (2μ_xμ_y + C1)(2σ_xy + C2) /
       (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

# Range: [0, 1], higher is better
# Better aligns with human perception than PSNR
```

### Perceptual Metrics
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **FID**: Fréchet Inception Distance
- **User Studies**: Subjective quality assessment

## Usage

```bash
# Train 4× super resolution model
python train_pytorch.py

# Different scaling factors
python train_pytorch.py --scale_factor 2  # 2× upscaling
python train_pytorch.py --scale_factor 8  # 8× upscaling
```

## Configuration

```python
CONFIG = {
    'scale_factor': 4,         # Upscaling factor
    'lr_size': 64,             # LR patch size
    'hr_size': 256,            # HR patch size (lr_size × scale)
    'num_residual_blocks': 16,
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 0.0001,
}
```

## Data Preparation

### Training Data
```python
# From HR images, create LR-HR pairs
hr_image = load_image()  # e.g., 256×256
lr_image = downsample(hr_image)  # e.g., 64×64

# Common downsampling methods:
- Bicubic interpolation
- Gaussian blur + downsampling
- Unknown degradation (real-world)
```

### Data Augmentation
```python
- Random crops
- Random horizontal/vertical flips
- Random rotations (90°, 180°, 270°)
- Color jittering (optional)
```

## Applications

### Photo Enhancement
- Upscale old low-resolution photos
- Improve scanned documents
- Enhance social media images

### Display/Printing
- Prepare images for large displays
- Print low-resolution images
- Zoom into image details

### Medical Imaging
- Enhance MRI/CT scan resolution
- Improve diagnostic image quality
- Reduce scan time (capture lower resolution)

### Satellite Imagery
- Enhance satellite images
- Improve surveillance quality
- Better map resolution

### Video Processing
- Upscale old videos to HD/4K
- Real-time video enhancement
- Streaming quality improvement

### Forensics
- Enhance security camera footage
- Improve license plate recognition
- Zoom into crime scene photos

## Model Variants

### SRCNN (First Deep Learning SR)
```python
# Simple CNN with 3 layers
# Fast but limited quality
```

### VDSR (Very Deep SR)
```python
# 20 layers with residual learning
# Better quality than SRCNN
```

### SRResNet
```python
# 16 residual blocks with sub-pixel convolution
# Good balance of speed and quality
```

### SRGAN
```python
# SRResNet + GAN training
# Best perceptual quality, may hallucinate
```

### EDSR (Enhanced Deep SR)
```python
# Remove batch norm from SRResNet
# Better performance, more memory
```

### ESRGAN
```python
# Enhanced SRGAN with better discriminator
# State-of-the-art perceptual quality
```

### Real-ESRGAN
```python
# Trained on real-world degradations
# Works on real photos (not just bicubic downsampling)
```

## Comparison

| Model | PSNR | Speed | Perceptual Quality | Model Size |
|-------|------|-------|-------------------|------------|
| Bicubic | Low | Very Fast | Poor | N/A |
| SRCNN | Medium | Fast | Medium | Small |
| VDSR | Good | Medium | Good | Medium |
| SRResNet | Good | Medium | Good | Medium |
| SRGAN | Medium | Medium | Excellent | Medium |
| EDSR | Excellent | Slow | Good | Large |
| ESRGAN | Good | Medium | Excellent | Medium |

## Challenges

### 1. Trade-off: PSNR vs. Perceptual Quality
- **High PSNR**: Smooth, less artifacts, may be blurry
- **High Perceptual**: Sharp textures, may hallucinate details

### 2. Computational Cost
- **Problem**: Large models, high memory usage
- **Solution**: Efficient architectures, quantization, pruning

### 3. Real-world Images
- **Problem**: Unknown degradation (not just bicubic)
- **Solution**: Train with diverse degradations, use blind SR

### 4. Artifacts
- **Problem**: Ringing, checkerboard patterns
- **Solution**: Better upsampling, perceptual loss

## Best Practices

1. **Use Sub-pixel Convolution**: Better than deconvolution
2. **Data Augmentation**: Increases robustness
3. **Perceptual Loss**: For better visual quality
4. **Progressive Training**: Start with small scale, then larger
5. **Pre-training**: Train on large datasets first
6. **Ensemble**: Combine multiple models for better results

## Advanced Techniques

### Multi-Scale Learning
```python
# Learn at multiple scales simultaneously
# Improves detail at different levels
```

### Attention Mechanisms
```python
# Focus on important regions
# Better handling of complex structures
```

### Reference-Based SR
```python
# Use similar HR images as reference
# Transfer textures from reference
```

### Zero-Shot SR
```python
# Learn from single image
# Internal image statistics
```

## Inference Optimization

### Tiling
```python
# Process large images in overlapping tiles
# Reduces memory usage
tile_size = 256
overlap = 32
```

### Half-Precision
```python
# Use FP16 for faster inference
model.half()
image = image.half()
```

### ONNX/TensorRT
```python
# Export to optimized runtimes
# 2-10× speedup
```

## Requirements

```bash
pip install torch torchvision Pillow numpy scikit-image matplotlib
```

## References

- SRCNN: [Dong et al., 2014](https://arxiv.org/abs/1501.00092)
- VDSR: [Kim et al., 2016](https://arxiv.org/abs/1511.04587)
- SRResNet/SRGAN: [Ledig et al., 2017](https://arxiv.org/abs/1609.04802)
- EDSR: [Lim et al., 2017](https://arxiv.org/abs/1707.02921)
- ESRGAN: [Wang et al., 2018](https://arxiv.org/abs/1809.00219)
- Real-ESRGAN: [Wang et al., 2021](https://arxiv.org/abs/2107.10833)
