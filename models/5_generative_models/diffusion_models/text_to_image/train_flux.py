"""
Text-to-Image Generation using FLUX.1
FLUX.1 is one of the most advanced open-source text-to-image models (August 2024)
Known for exceptional prompt following, detail quality, and text rendering.
"""

import torch
from diffusers import FluxPipeline
import os
from datetime import datetime
import json

def generate_images(
    prompts,
    model_id="black-forest-labs/FLUX.1-schnell",  # Fast version (1-4 steps)
    # model_id="black-forest-labs/FLUX.1-dev",   # High quality version (20-50 steps)
    output_dir="generated_images_flux",
    num_inference_steps=4,  # Use 4 for schnell, 20-50 for dev
    guidance_scale=0.0,     # FLUX.1-schnell doesn't use guidance
    num_images_per_prompt=1,
    height=1024,
    width=1024,
    seed=None
):
    """
    Generate images from text prompts using FLUX.1
    
    Args:
        prompts: List of text prompts or single prompt string
        model_id: Model identifier (schnell=fast, dev=high quality)
        output_dir: Directory to save generated images
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale (0 for schnell, 3.5 for dev)
        num_images_per_prompt: Number of images to generate per prompt
        height: Image height in pixels (512-2048)
        width: Image width in pixels (512-2048)
        seed: Random seed for reproducibility
    """
    
    print("=" * 60)
    print("Text-to-Image Generation with FLUX.1")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert single prompt to list
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if device == "cpu":
        print("⚠ Warning: FLUX.1 is extremely heavy and NOT recommended for CPU!")
        print("  This will likely crash or take hours per image.")
        print("  Please use a GPU with at least 12GB VRAM.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            return
    
    # Load the pipeline
    print(f"\nLoading model: {model_id}")
    print("⚠ Note: FLUX.1 models are very large (~30GB). First download may take time.")
    
    try:
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            # pipe.enable_vae_tiling()  # Uncomment for lower VRAM usage
            print("✓ Enabled memory optimizations")
        
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have Hugging Face access and have accepted the license")
        print("2. Login with: huggingface-cli login")
        print("3. Install: pip install diffusers transformers accelerate sentencepiece protobuf")
        print("4. GPU Memory: FLUX.1 needs 12GB+ VRAM")
        print(f"5. Try the lighter model: black-forest-labs/FLUX.1-schnell")
        return
    
    # Set random seed if provided
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        generator = None
    
    # Model-specific settings
    model_name = "schnell" if "schnell" in model_id.lower() else "dev"
    if model_name == "schnell" and guidance_scale != 0.0:
        print(f"Note: FLUX.1-schnell ignores guidance_scale, setting to 0")
        guidance_scale = 0.0
    
    # Generate images
    print(f"\nGenerating {len(prompts)} image(s)...")
    print(f"Model: FLUX.1-{model_name}")
    print(f"Parameters: steps={num_inference_steps}, guidance={guidance_scale}, size={width}x{height}")
    print("-" * 60)
    
    all_results = []
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Prompt: '{prompt}'")
        
        try:
            # Generate image
            if model_name == "schnell":
                # Schnell doesn't use guidance scale
                images = pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    height=height,
                    width=width,
                    generator=generator
                ).images
            else:
                # Dev version uses guidance scale
                images = pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    height=height,
                    width=width,
                    generator=generator
                ).images
            
            # Save images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for img_idx, image in enumerate(images):
                filename = f"flux_{model_name}_{timestamp}_prompt{idx}_img{img_idx+1}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                
                result = {
                    "prompt": prompt,
                    "filename": filename,
                    "model": f"FLUX.1-{model_name}",
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "size": f"{width}x{height}",
                    "seed": seed if seed else "random"
                }
                all_results.append(result)
                
                print(f"  ✓ Saved: {filename}")
        
        except Exception as e:
            print(f"  ✗ Error generating image: {e}")
            if "out of memory" in str(e).lower():
                print("  💡 Try reducing image size or using FLUX.1-schnell")
            continue
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "generation_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"✓ Generated {len(all_results)} image(s)")
    print(f"✓ Saved to: {output_dir}/")
    print(f"✓ Metadata saved to: {metadata_file}")
    
    # QA Validation
    print("\n" + "=" * 60)
    print("QA Validation")
    print("=" * 60)
    
    print("\n--- Sanity Checks ---")
    
    # Check 1: All images generated
    expected_images = len(prompts) * num_images_per_prompt
    actual_images = len(all_results)
    if actual_images == expected_images:
        print(f"✓ All {expected_images} images generated successfully")
    else:
        print(f"⚠ Generated {actual_images}/{expected_images} images")
    
    # Check 2: Files exist and have reasonable sizes
    missing_files = []
    small_files = []
    total_size_mb = 0
    
    for result in all_results:
        filepath = os.path.join(output_dir, result["filename"])
        if not os.path.exists(filepath):
            missing_files.append(result["filename"])
        else:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            total_size_mb += size_mb
            if size_mb < 0.01:  # Less than 10KB
                small_files.append((result["filename"], size_mb))
    
    if not missing_files:
        print("✓ All generated images saved to disk")
    else:
        print(f"✗ Missing files: {missing_files}")
    
    if not small_files:
        print(f"✓ All images have reasonable file sizes (Total: {total_size_mb:.2f} MB)")
    else:
        print(f"⚠ Suspiciously small files: {small_files}")
    
    # Check 3: Metadata saved
    if os.path.exists(metadata_file):
        print("✓ Metadata file saved successfully")
    else:
        print("✗ Metadata file missing")
    
    # Check 4: Image quality indicators
    print(f"\nGeneration Statistics:")
    print(f"  Model: FLUX.1-{model_name}")
    print(f"  Average file size: {total_size_mb/len(all_results):.2f} MB")
    print(f"  Resolution: {width}x{height}")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        actual_images == expected_images and
        not missing_files and
        not small_files and
        os.path.exists(metadata_file)
    )
    
    if validation_passed:
        print("✓ Validation PASSED - All images generated successfully")
    else:
        print("✗ Validation FAILED - Some issues detected")
    
    return all_results


def main():
    """Main function with example prompts"""
    
    # Example prompts showcasing FLUX.1's strengths
    example_prompts = [
        "A photograph of a sign that says 'FLUX.1' in bold letters, with a beautiful sunset background, photorealistic, 8k",
        "An oil painting of a wise old wizard reading a book titled 'Machine Learning', intricate details, fantasy art",
        "A hyperrealistic portrait of a cyberpunk hacker with glowing blue eyes and neon tattoos, cinematic lighting",
        "A cute cartoon cat wearing glasses and a bowtie, sitting at a desk with a laptop, digital art, warm colors"
    ]
    
    print("\n🎨 FLUX.1 - State-of-the-Art Text-to-Image Generation")
    print("\nExample prompts:")
    for i, prompt in enumerate(example_prompts, 1):
        print(f"{i}. {prompt}")
    
    print("\n" + "=" * 60)
    print("Starting generation with first prompt...")
    print("=" * 60)
    
    # Generate image with the first prompt
    # Using FLUX.1-schnell for speed (4 steps)
    # For best quality, switch to FLUX.1-dev with 20-50 steps
    
    generate_images(
        prompts=[example_prompts[0]],
        model_id="black-forest-labs/FLUX.1-schnell",  # Fast version
        num_inference_steps=4,   # 4 steps for schnell
        guidance_scale=0.0,      # Schnell doesn't use guidance
        height=1024,
        width=1024,
        seed=42
    )
    
    print("\n" + "=" * 60)
    print("FLUX.1 Model Comparison:")
    print("=" * 60)
    print("FLUX.1-schnell:")
    print("  • Ultra-fast: 4 steps (~2-5 seconds on A100)")
    print("  • Great quality for speed")
    print("  • No guidance scale needed")
    print("  • Best for: Quick iterations, batch generation")
    print("\nFLUX.1-dev:")
    print("  • Best quality: 20-50 steps")
    print("  • Superior prompt following")
    print("  • Supports guidance scale (3.5 recommended)")
    print("  • Best for: Final renders, complex scenes")
    print("\n" + "=" * 60)
    print("Tips for FLUX.1:")
    print("=" * 60)
    print("• FLUX excels at text rendering - try prompts with signs/text")
    print("• Natural language works well - be descriptive")
    print("• Supports wide range of styles: photo, art, anime, 3D")
    print("• Works at various resolutions: 512-2048px")
    print("• Guidance scale 3.5 recommended for FLUX.1-dev")
    print("• Lower steps than SD models due to advanced architecture")
    print("\nGPU Requirements:")
    print("  • Minimum: 12GB VRAM (schnell, lower resolutions)")
    print("  • Recommended: 24GB VRAM (dev, high resolutions)")
    print("  • Consider using quantized versions for lower VRAM")


if __name__ == "__main__":
    main()
