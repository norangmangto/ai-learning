# Generative Adversarial Networks (GANs)

## Overview

GANs generate new data (images, audio, text) by training two neural networks in competition. This directory contains implementations of various GAN architectures for image generation.

## What are GANs?

**Theory:**
GANs consist of two networks playing a min-max game:
- **Generator (G)**: Creates fake data to fool the discriminator
- **Discriminator (D)**: Distinguishes real data from fake

```
Min-Max Game:
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

**Training Process:**
1. Generator creates fake images from random noise
2. Discriminator tries to classify real vs fake
3. Both networks improve iteratively
4. Eventually, generator creates realistic images

### Common Use Cases
- Image generation (faces, scenes, art)
- Image-to-image translation (style transfer)
- Data augmentation
- Super-resolution
- Image inpainting
- Video generation
- Drug discovery (molecular generation)

## GAN Architectures Implemented

### 1. DCGAN (Deep Convolutional GAN)

**Files:** `dcgan/train_pytorch.py`, `dcgan/train_tensorflow.py`

**Theory:**
Extension of GAN using convolutional layers instead of fully connected:
```
Generator: Noise (100D) → TransposedConv → BatchNorm → ReLU → ... → Image
Discriminator: Image → Conv → LeakyReLU → ... → Real/Fake (1D)
```

**Architecture Guidelines:**
- Replace pooling with strided convolutions
- Use batch normalization (except generator output, discriminator input)
- Remove fully connected layers
- Use ReLU in generator (Tanh for output)
- Use LeakyReLU in discriminator

**When to Use:**
- First GAN to try
- Image generation tasks
- Baseline for comparison
- Learning GAN fundamentals

**Advantages:**
- ✅ Stable training (compared to vanilla GAN)
- ✅ Good image quality
- ✅ Well-understood architecture
- ✅ Easy to implement

**Limitations:**
- ❌ Mode collapse risk
- ❌ Training instability
- ❌ Limited resolution (typically 64x64)
- ❌ No explicit control over generation

**Hyperparameters:**
```python
# Generator
latent_dim = 100              # Noise vector size
ngf = 64                      # Generator feature maps
# Architecture: 100 → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 3×64×64

# Discriminator
ndf = 64                      # Discriminator feature maps
# Architecture: 3×64×64 → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → 1

# Training
learning_rate_g = 0.0002      # Generator lr
learning_rate_d = 0.0002      # Discriminator lr
beta1 = 0.5                   # Adam optimizer beta1
batch_size = 128
epochs = 50-200
```

### 2. Conditional GAN (cGAN)

**Directory:** `conditional_gan/`

**Theory:**
GAN with conditional information (class labels, text, etc.):
```
G(z, y) → fake image of class y
D(x, y) → probability that x is real and belongs to class y
```

**When to Use:**
- Controlled generation (generate specific classes)
- Text-to-image synthesis
- Image-to-image translation
- Attribute-controlled generation

**Advantages:**
- ✅ Controllable generation
- ✅ Better mode coverage
- ✅ Task-specific generation
- ✅ Guided synthesis

**Limitations:**
- ❌ Requires labeled data
- ❌ More complex architecture
- ❌ Harder to train

**Applications:**
- Generate digit "7" (MNIST)
- Generate "cat" images (ImageNet)
- Text-to-image (description → image)
- Attribute control (smile, age, hair color)

### 3. StyleGAN

**Directory:** `stylegan/`

**Theory:**
Advanced architecture with style-based generator:
```
Mapping Network: z → w (learned intermediate latent space)
Synthesis Network: w → image (with style injection at each layer)
```

**Key Innovations:**
1. **Style injection**: Controls features at different scales
2. **Adaptive Instance Normalization (AdaIN)**: Applies styles
3. **Progressive growing**: Starts small, grows to high resolution
4. **Mixing regularization**: Improves diversity

**When to Use:**
- High-quality image generation
- Face generation
- Style transfer and manipulation
- Research projects

**Advantages:**
- ✅ State-of-the-art quality (1024x1024)
- ✅ Disentangled latent space
- ✅ Style mixing capabilities
- ✅ Controllable generation

**Limitations:**
- ❌ Very complex architecture
- ❌ Requires large datasets
- ❌ Computationally expensive
- ❌ Long training time

**Latent Space Properties:**
- **Coarse styles (4×4 - 8×8)**: Pose, face shape, glasses
- **Middle styles (16×16 - 32×32)**: Facial features, hairstyle
- **Fine styles (64×64 - 1024×1024)**: Color scheme, micro-structure

### 4. Progressive GAN

**Directory:** `progressive_gan/`

**Theory:**
Grows generator and discriminator progressively:
```
Start: 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128 → ... → 1024×1024
```

**How it Works:**
1. Start training at low resolution (4×4)
2. Gradually add layers for higher resolution
3. Fade in new layers smoothly
4. Results in stable high-resolution generation

**When to Use:**
- Need high-resolution images
- Stable training important
- Incremental quality improvement

**Advantages:**
- ✅ Stable training
- ✅ High-resolution output
- ✅ Faster convergence
- ✅ Progressive quality improvement

**Limitations:**
- ❌ Complex training procedure
- ❌ Longer overall training time
- ❌ More hyperparameters

## Implementation Files

### PyTorch Implementation
**Files:** `*/train_pytorch.py`

**Features:**
- Custom GAN architectures
- Weight initialization
- Gradient penalty (WGAN-GP)
- Spectral normalization
- TensorBoard logging

### TensorFlow/Keras Implementation
**Files:** `*/train_tensorflow.py`

**Features:**
- Keras Sequential/Functional API
- GAN training loop
- TensorBoard integration
- Checkpointing

## Training GANs

### Training Loop

```python
for epoch in range(epochs):
    for real_images in dataloader:
        # 1. Train Discriminator
        # 1a. Real images
        real_labels = torch.ones(batch_size, 1)
        d_loss_real = criterion(discriminator(real_images), real_labels)

        # 1b. Fake images
        noise = torch.randn(batch_size, latent_dim)
        fake_images = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)
        d_loss_fake = criterion(discriminator(fake_images.detach()), fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 2. Train Generator
        noise = torch.randn(batch_size, latent_dim)
        fake_images = generator(noise)
        g_loss = criterion(discriminator(fake_images), real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
```

### Best Practices

#### 1. Loss Functions

**Binary Cross-Entropy (BCE):**
```python
criterion = nn.BCELoss()  # Original GAN
```

**Wasserstein Loss (WGAN):**
```python
# Discriminator: maximize D(x) - D(G(z))
d_loss = -torch.mean(D_real) + torch.mean(D_fake)
# Generator: maximize D(G(z))
g_loss = -torch.mean(D_fake)
```

**Hinge Loss:**
```python
# Discriminator
d_loss = torch.mean(torch.relu(1 - D_real)) + torch.mean(torch.relu(1 + D_fake))
# Generator
g_loss = -torch.mean(D_fake)
```

#### 2. Stabilization Techniques

**Label Smoothing:**
```python
# Instead of 1.0 for real, use 0.9
real_labels = torch.ones(batch_size, 1) * 0.9
```

**Noisy Labels:**
```python
# Add noise to labels occasionally
if random.random() < 0.05:
    real_labels, fake_labels = fake_labels, real_labels
```

**Gradient Penalty (WGAN-GP):**
```python
def gradient_penalty(discriminator, real_images, fake_images):
    alpha = torch.rand(batch_size, 1, 1, 1)
    interpolates = alpha * real_images + (1 - alpha) * fake_images
    d_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

d_loss += lambda_gp * gradient_penalty(D, real_images, fake_images)
```

**Spectral Normalization:**
```python
from torch.nn.utils import spectral_norm

# Apply to discriminator layers
nn.Conv2d = spectral_norm(nn.Conv2d(...))
```

#### 3. Training Tips

**Discriminator Updates:**
```python
# Train discriminator more frequently
n_critic = 5  # Update D 5 times for each G update

for _ in range(n_critic):
    train_discriminator()

train_generator()
```

**Learning Rate:**
```python
# Often use different learning rates
lr_g = 0.0001  # Generator (slower)
lr_d = 0.0004  # Discriminator (faster)
```

**Batch Size:**
```python
# Larger batches help stability
batch_size = 64-256  # depends on GPU memory
```

#### 4. Monitoring Training

**Metrics to Track:**
- Discriminator loss (should be around 0.5-0.7)
- Generator loss (should decrease)
- D(real) - should be high (close to 1)
- D(fake) - should be low (close to 0) but increasing
- Inception Score (IS)
- Fréchet Inception Distance (FID)

**Visual Inspection:**
```python
# Generate samples every N epochs
if epoch % 10 == 0:
    with torch.no_grad():
        fixed_noise = torch.randn(64, latent_dim)
        fake_images = generator(fixed_noise)
        save_image(fake_images, f'samples/epoch_{epoch}.png')
```

## Common Issues and Solutions

### Issue: Mode Collapse
**Symptoms:** Generator produces same/similar images

**Solutions:**
- ✅ Use Wasserstein loss (WGAN-GP)
- ✅ Minibatch discrimination
- ✅ Feature matching
- ✅ Unrolled GAN
- ✅ Lower discriminator learning rate
- ✅ Add noise to discriminator inputs

### Issue: Training Instability
**Symptoms:** Loss oscillations, divergence

**Solutions:**
- ✅ Spectral normalization
- ✅ Gradient penalty
- ✅ Lower learning rates (1e-4)
- ✅ Label smoothing
- ✅ Batch normalization
- ✅ Update discriminator more frequently

### Issue: Poor Image Quality
**Solutions:**
- ✅ Train longer
- ✅ Increase model capacity
- ✅ Better architecture (Progressive GAN, StyleGAN)
- ✅ More training data
- ✅ Data augmentation
- ✅ Self-attention layers

### Issue: Discriminator Too Strong
**Symptoms:** Generator loss doesn't decrease

**Solutions:**
- ✅ Reduce discriminator capacity
- ✅ Add dropout to discriminator
- ✅ Lower discriminator learning rate
- ✅ Add noise to discriminator inputs
- ✅ Update generator more frequently

### Issue: Generator Too Strong
**Symptoms:** Discriminator loss near 0

**Solutions:**
- ✅ Increase discriminator capacity
- ✅ Train discriminator more frequently
- ✅ Higher discriminator learning rate
- ✅ Remove dropout from discriminator

## Evaluation Metrics

### 1. Inception Score (IS)
```python
from torchmetrics.image.inception import InceptionScore

is_metric = InceptionScore()
is_metric.update(generated_images)
score = is_metric.compute()
# Higher is better, typical range: 2-10
```

### 2. Fréchet Inception Distance (FID)
```python
from torchmetrics.image.fid import FrechetInceptionDistance

fid_metric = FrechetInceptionDistance()
fid_metric.update(real_images, real=True)
fid_metric.update(generated_images, real=False)
fid_score = fid_metric.compute()
# Lower is better, typical range: 10-100
```

### 3. Visual Inspection
- Diversity of generated samples
- Artifacts and distortions
- Realism and sharpness
- Mode coverage

## Advanced Techniques

### 1. Self-Attention GAN (SAGAN)
- Adds self-attention mechanisms
- Better long-range dependencies
- Improved image quality

### 2. BigGAN
- Very large models (up to 160M parameters)
- Class-conditional generation
- Truncation trick for quality/diversity tradeoff

### 3. Conditional GAN Extensions
- **Pix2Pix**: Image-to-image translation
- **CycleGAN**: Unpaired image translation
- **StarGAN**: Multi-domain translation

## Quick Start

### DCGAN (Recommended for beginners)
```bash
cd dcgan
python train_pytorch.py
```

### Conditional GAN
```bash
cd conditional_gan
python train_pytorch.py --num_classes 10
```

### StyleGAN (Advanced)
```bash
cd stylegan
python train_pytorch.py --dataset celeba-hq
```

## Further Reading

- [Original GAN Paper (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [StyleGAN Paper](https://arxiv.org/abs/1812.04948)
- [GAN Hacks: Tips and Tricks](https://github.com/soumith/ganhacks)
- [Progressive GAN Paper](https://arxiv.org/abs/1710.10196)

## Next Steps

1. Try **Autoencoders** (../autoencoder/) for reconstruction tasks
2. Explore **Diffusion Models** (../diffusion_models/) for state-of-the-art generation
3. Study **Image-to-Image Translation** (../../3_computer_vision/image_to_image/)
4. Learn **Variational Autoencoders** (VAE)
5. Implement **Text-to-Image** models (Stable Diffusion, DALL-E)
