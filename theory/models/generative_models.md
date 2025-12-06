# Generative AI Models

These models learn the data distribution to generate *new* samples similar to the training data.

## 1. Autoencoder

### Concept
Unsupervised network that learns to compress input into a lower-dimensional latent space and reconstruct it.
*   **Encoder**: $x \rightarrow z$ (latent)
*   **Decoder**: $z \rightarrow x'$ (reconstruction)
The "bottleneck" forces the model to learn efficient features.

### Use Cases
*   Denoising.
*   Dimensionality Reduction.
*   Anomaly Detection.

### Code
*   [Autoencoder](../../models/advanced/autoencoder/train_pytorch.py)

---

## 2. GAN (Generative Adversarial Network)

### Concept
A game between two networks:
*   **Generator**: Tries to create fake images to fool the Discriminator.
*   **Discriminator**: Tries to distinguish real images from fakes.
They train together in a zero-sum game until Nash Equilibrium.

### Pros & Cons
*   **Pros**: Generates very sharp, realistic images.
*   **Cons**: Unstable training (mode collapse). Hard to capture full distribution.

### Code
*   [GAN](../../models/advanced/gan/train_pytorch.py)

---

## 3. U-Net (Image-to-Image)

### Concept
An Encoder-Decoder with **Skip Connections**. The skip connections pass high-resolution details from the encoder directly to the decoder, making it perfect for tasks where the output aligns perfectly with the input spatially.

### Use Cases
*   Medical Segmentation.
*   Image Colorization.
*   Pix2Pix.

### Code
*   [U-Net Colorization](../../models/advanced/image_to_image/unet/train_pytorch.py)

---

## 4. Diffusion Models (Stable Diffusion)

### Concept
Generates data by reversing a noise-adding process.
1.  **Forward**: Gradually add Gaussian noise to an image until it's pure noise.
2.  **Reverse**: Train a neural network (usually U-Net or Transformer) to **predict the noise** added at each step, effectively "denoising" the image.
*   **Latent Diffusion**: Runs this process in a compressed latent space (via VAE) for efficiency.

### Pros & Cons
*   **Pros**: Incredible diversity and quality. No mode collapse.
*   **Cons**: Slow sampling (requires many steps).

### Code
*   [Dreambooth (SD1.5)](../../models/advanced/generative/dreambooth_lora/train.py)
*   [Dreambooth (SD3.5)](../../models/advanced/generative/dreambooth_lora_sd3/train.py)

---

## 5. Flow Matching (Flux.1)

### Concept
A generalization of Diffusion. Instead of a discrete noise schedule, it models a continuous Vector Field (flow) that transforms a simple distribution (Noise) into the data distribution (Image) via an ODE (Ordinary Differential Equation).
**Flux.1** uses **Rectified Flow**, which attempts to make the transport path as straight as possible, allowing for fewer sampling steps.

### Use Cases
*   State-of-the-Art Image Generation.

### Code
*   [Flux.1 Dreambooth](../../models/advanced/generative/dreambooth_lora_flux/train.py)
