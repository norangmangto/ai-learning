# Image-to-Video Generation with Text Prompts

## Overview
Image-to-video generation is a multimodal task that creates dynamic video sequences from a static input image, conditioned on text prompts. This technique combines computer vision and natural language processing to synthesize realistic motion and temporal dynamics.

## Key Concepts

### 1. **Image-to-Video (I2V) Generation**
- Takes a single image as the first frame
- Generates subsequent frames with coherent motion
- Preserves subject identity and scene context from the input image

### 2. **Text Conditioning**
- Uses text prompts to guide the type of motion/action
- Examples: "a person walking forward", "waves crashing on shore", "camera zooming out"
- Enables fine-grained control over generated video content

### 3. **Temporal Consistency**
- Ensures smooth transitions between frames
- Maintains object identity across time
- Prevents flickering and temporal artifacts

## Popular Approaches

### 1. **Stable Video Diffusion (SVD)**
- Based on Stability AI's latent diffusion models
- Uses a pre-trained image model extended to video
- Generates 14-25 frames at 6-8 fps
- Excellent for short video clips

### 2. **AnimateDiff**
- Motion module that can be added to existing image models
- Works with Stable Diffusion checkpoints
- Flexible and extensible architecture

### 3. **Gen-2 / Pika Labs (Commercial)**
- State-of-the-art commercial solutions
- Higher quality but API-based

### 4. **Latent Video Diffusion**
- Operates in latent space for efficiency
- Uses 3D convolutions for temporal modeling
- Decoder reconstructs pixel-space video

## Architecture Components

### Encoder
```
Input Image → Image Encoder (VAE/CLIP) → Latent Representation
Text Prompt → Text Encoder (CLIP/T5) → Text Embeddings
```

### Temporal U-Net
```
- 3D Convolutions (spatial + temporal)
- Self-Attention layers (frame-to-frame)
- Cross-Attention layers (text conditioning)
- Temporal positional encodings
```

### Decoder
```
Latent Video Frames → Video VAE Decoder → RGB Frames
```

## Technical Challenges

1. **Computational Cost**: Video generation requires significantly more compute than images
2. **Temporal Coherence**: Maintaining consistency across frames
3. **Motion Realism**: Generating physically plausible motion
4. **Long Videos**: Current models typically limited to 2-4 seconds
5. **Memory Requirements**: Processing multiple frames simultaneously

## Model Implementation

### Dependencies
```bash
pip install torch torchvision diffusers transformers accelerate safetensors imageio pillow
```

### Typical Pipeline
1. Load pre-trained image diffusion model
2. Extend spatial layers to temporal dimensions
3. Fine-tune or use motion modules
4. Encode input image and text prompt
5. Run diffusion sampling process
6. Decode latent frames to video

## Use Cases

- **Content Creation**: Social media, marketing videos
- **Animation**: Character animation from still images
- **Film Production**: Concept visualization, storyboarding
- **E-commerce**: Product demonstrations
- **Gaming**: Cutscene generation
- **Education**: Bringing historical photos to life

## Training Data

Models are typically trained on:
- Large-scale video datasets (WebVid-10M, YouTube clips)
- Image-text pairs extended to video
- Synthetic data with known motion patterns

## Evaluation Metrics

1. **Fréchet Video Distance (FVD)**: Video quality and diversity
2. **Temporal Consistency**: Frame-to-frame similarity metrics
3. **Text-Video Alignment**: CLIP score between prompt and video
4. **Inception Score (IS)**: Video quality
5. **User Studies**: Perceptual quality and motion realism

## Research Papers

- **Stable Video Diffusion** (Stability AI, 2023)
- **AnimateDiff** (2023)
- **Make-A-Video** (Meta, 2022)
- **Imagen Video** (Google, 2022)
- **Video Diffusion Models** (2022)

## Files in This Directory

- `train_svd.py`: Training script using Stable Video Diffusion architecture
- `generate_video.py`: Inference script for generating videos from images and prompts
- `model_architectures.py`: Custom model implementations
- `utils.py`: Helper functions for video processing

## Example Usage

```python
from generate_video import ImageToVideoGenerator

# Initialize generator
generator = ImageToVideoGenerator(
    model_name="stabilityai/stable-video-diffusion-img2vid",
    device="cuda"
)

# Generate video
video_frames = generator.generate(
    image_path="input_image.jpg",
    prompt="a person waving at the camera",
    num_frames=25,
    fps=8
)

# Save video
generator.save_video(video_frames, "output_video.mp4")
```

## Tips for Best Results

1. **Input Image Quality**: Use high-resolution, clear images
2. **Prompt Engineering**: Be specific about desired motion
3. **Frame Count**: Start with 14-16 frames for faster generation
4. **Guidance Scale**: 7-15 typically works well
5. **Denoising Strength**: Control how much the image changes (0.6-0.8)
6. **Seed Control**: Use fixed seeds for reproducibility

## Limitations

- Limited to short videos (2-4 seconds)
- May struggle with complex multi-object interactions
- Text conditioning effectiveness varies by model
- High computational requirements
- Potential for temporal artifacts
- Limited control over specific motion trajectories
