# Text-to-Image Generation

Generate high-quality images from text prompts using state-of-the-art AI models.

## 🚀 Available Models

### 1. **FLUX.1** (Recommended - Best Quality)
- **Model**: Black Forest Labs FLUX.1
- **Release**: August 2024
- **Best for**: Highest quality, exceptional text rendering, natural prompts
- **Versions**:
  - `FLUX.1-schnell`: Ultra-fast (4 steps, 2-5 seconds)
  - `FLUX.1-dev`: Best quality (20-50 steps)

**Run**: `python train_flux.py`

### 2. **Stable Diffusion 3.5**
- **Model**: Stability AI SD 3.5
- **Release**: October 2024  
- **Best for**: Great balance of quality and speed, good prompt following
- **Steps**: 28-50 recommended

**Run**: `python train_sd3.py`

## 📋 Requirements

### Install Dependencies
```bash
pip install torch torchvision diffusers transformers accelerate safetensors sentencepiece protobuf
```

### GPU Requirements
- **Minimum**: NVIDIA GPU with 12GB VRAM (RTX 3060, RTX 4070)
- **Recommended**: 16-24GB VRAM (RTX 4090, A5000, A100)
- **Models won't work well on CPU** - generation will be extremely slow

### Hugging Face Setup
1. Create account at [huggingface.co](https://huggingface.co)
2. Accept model licenses:
   - FLUX.1: https://huggingface.co/black-forest-labs/FLUX.1-schnell
   - SD 3.5: https://huggingface.co/stabilityai/stable-diffusion-3.5-large
3. Get access token: https://huggingface.co/settings/tokens
4. Login: `huggingface-cli login`

## 🎨 Quick Start

### Generate Your First Image

**Option 1: FLUX.1 (Fastest)**
```python
from train_flux import generate_images

generate_images(
    prompts="A serene mountain landscape at sunset, photorealistic",
    num_inference_steps=4,
    height=1024,
    width=1024
)
```

**Option 2: Stable Diffusion 3.5**
```python
from train_sd3 import generate_images

generate_images(
    prompts="A futuristic city with flying cars, cyberpunk style",
    num_inference_steps=28,
    guidance_scale=7.0,
    height=1024,
    width=1024
)
```

### Batch Generation
```python
prompts = [
    "A cute robot reading a book",
    "An astronaut riding a horse on Mars",
    "A magical forest with glowing mushrooms"
]

generate_images(prompts=prompts)
```

## 💡 Prompt Writing Tips

### Good Prompt Structure
```
[Subject] [doing action], [style/medium], [lighting], [quality keywords]
```

### Examples
✅ **Good**: "A wise old wizard reading a spell book, oil painting, warm candlelight, intricate details, fantasy art"

❌ **Bad**: "wizard book"

### Key Elements
- **Subject**: What you want to see
- **Action**: What they're doing
- **Style**: Photorealistic, anime, oil painting, digital art, 3D render
- **Lighting**: Sunset, studio lighting, neon lights, dramatic shadows
- **Quality**: 8k, highly detailed, cinematic, professional photography
- **Mood**: Cozy, epic, mysterious, cheerful

### Style Keywords
- **Photo**: photorealistic, DSLR, 85mm lens, bokeh, professional photography
- **Art**: oil painting, watercolor, pencil sketch, concept art, illustration
- **Digital**: digital art, 3D render, CGI, Unreal Engine
- **Anime**: anime, manga, Studio Ghibli, cel shaded
- **Other**: cyberpunk, steampunk, fantasy art, surrealism

## ⚙️ Configuration

### Image Quality vs Speed

| Model | Steps | Time (A100) | Quality |
|-------|-------|-------------|---------|
| FLUX.1-schnell | 4 | 2-5s | Very Good |
| FLUX.1-dev | 20 | 15-20s | Excellent |
| FLUX.1-dev | 50 | 40-50s | Best |
| SD 3.5 | 28 | 10-15s | Great |
| SD 3.5 | 50 | 20-30s | Excellent |

### Resolution Options
- **Fast**: 512x512 (lower VRAM, faster)
- **Standard**: 1024x1024 (recommended)
- **High**: 1536x1536 (requires more VRAM)
- **Portrait**: 832x1216
- **Landscape**: 1216x832

### Parameters Explained

**num_inference_steps**: Number of denoising iterations
- Lower (4-10): Faster, less detailed
- Medium (20-30): Good balance
- Higher (40-50): Best quality, slower

**guidance_scale**: How closely to follow the prompt
- 0: No guidance (FLUX.1-schnell)
- 3.5: Recommended for FLUX.1-dev
- 7-9: Recommended for SD models
- Higher: Stronger prompt following, may reduce diversity

**seed**: For reproducible results
- Set to a number (e.g., 42) to get the same image
- Leave as None for random variations

## 🎯 Best Practices

### For Photorealism
```python
"A professional photograph of [subject], DSLR, 85mm f/1.8, bokeh background, 
natural lighting, high resolution, sharp focus, professional photography"
```

### For Artistic Styles
```python
"An oil painting of [subject], impressionist style, vibrant colors, 
visible brushstrokes, museum quality, fine art"
```

### For Text in Images
```python
"A photograph of a sign that says 'YOUR TEXT HERE' in bold letters, 
clear readable text, high contrast, sharp focus"
```

**Note**: FLUX.1 is significantly better at text rendering than SD models.

## 🔧 Troubleshooting

### Out of Memory Error
1. Reduce image resolution: `height=768, width=768`
2. Use FLUX.1-schnell instead of dev
3. Enable VAE tiling (uncomment in code)
4. Close other GPU applications
5. Reduce batch size to 1 image at a time

### Slow Generation
- **On CPU**: Models require GPU, CPU is 100x+ slower
- **First run**: Model download takes time (~30GB for FLUX)
- **Subsequent runs**: Should be much faster

### Poor Quality
- Increase `num_inference_steps`
- Improve prompt detail and specificity
- Try different `guidance_scale` values
- Use higher resolution
- Try FLUX.1-dev instead of schnell

### Model Won't Load
1. Check Hugging Face login: `huggingface-cli whoami`
2. Verify model license accepted
3. Check disk space (models are large)
4. Update packages: `pip install -U diffusers transformers`

## 📊 Model Comparison

| Feature | FLUX.1-schnell | FLUX.1-dev | SD 3.5 |
|---------|----------------|------------|--------|
| Quality | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Text Rendering | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Prompt Following | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| VRAM Usage | 12GB+ | 16GB+ | 12GB+ |
| License | Apache 2.0 | Non-commercial* | Stability AI** |

*FLUX.1-dev requires commercial license from BFL  
**SD 3.5 allows commercial use with conditions

## 🔗 Resources

- [FLUX.1 Model Card](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [Stable Diffusion 3.5 Model Card](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)

## 📝 Output Structure

```
generated_images_[model]/
├── flux_schnell_20241209_143022_prompt1_img1.png
├── flux_schnell_20241209_143045_prompt2_img1.png
└── generation_metadata.json
```

Metadata includes:
- Prompt text
- Model and parameters used
- Image resolution
- Random seed (for reproduction)
- Timestamp

## 🚀 Next Steps

1. **Start Simple**: Run the example scripts as-is
2. **Experiment**: Try different prompts and parameters
3. **Compare**: Generate same prompt with different models
4. **Optimize**: Adjust settings for your GPU and use case
5. **Iterate**: Refine prompts based on results

Happy generating! 🎨✨
