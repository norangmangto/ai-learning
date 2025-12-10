"""
Text-to-Image Generation using Stable Diffusion 3.5
This script generates high-quality images from text prompts using the latest Stable Diffusion 3.5 model.
"""

import torch
from diffusers import StableDiffusion3Pipeline
import os
from datetime import datetime
import json


def generate_images(
    prompts,
    model_id="stabilityai/stable-diffusion-3.5-large",
    output_dir="generated_images_sd3",
    num_inference_steps=28,
    guidance_scale=7.0,
    num_images_per_prompt=1,
    height=1024,
    width=1024,
    seed=None,
):
    """
    Generate images from text prompts using Stable Diffusion 3.5

    Args:
        prompts: List of text prompts or single prompt string
        model_id: Model identifier (default: SD 3.5 Large)
        output_dir: Directory to save generated images
        num_inference_steps: Number of denoising steps (higher = better quality, slower)
        guidance_scale: How closely to follow the prompt (7-9 recommended)
        num_images_per_prompt: Number of images to generate per prompt
        height: Image height in pixels
        width: Image width in pixels
        seed: Random seed for reproducibility (None for random)
    """

    print("=" * 60)
    print("Text-to-Image Generation with Stable Diffusion 3.5")
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
        print("⚠ Warning: Running on CPU. This will be very slow!")
        print("  Consider using a GPU or reducing image resolution.")

    # Load the pipeline
    print(f"\nLoading model: {model_id}")
    print("This may take a few minutes on first run...")

    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
        )
        pipe = pipe.to(device)

        # Enable memory optimizations for lower VRAM usage
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            print("✓ Enabled CPU offloading for memory optimization")

        print("✓ Model loaded successfully")

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have accepted the model license at:")
        print(f"   https://huggingface.co/{model_id}")
        print("2. Login with: huggingface-cli login")
        print(
            "3. Install required packages: pip install diffusers transformers accelerate"
        )
        return

    # Set random seed if provided
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        generator = None

    # Generate images
    print(f"\nGenerating {len(prompts)} image(s)...")
    print(
        f"Parameters: steps={num_inference_steps}, guidance={guidance_scale}, size={width}x{height}"
    )
    print("-" * 60)

    all_results = []

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Prompt: '{prompt}'")

        try:
            # Generate image
            images = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                generator=generator,
            ).images

            # Save images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for img_idx, image in enumerate(images):
                filename = f"sd3_{timestamp}_prompt{idx}_img{img_idx+1}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)

                result = {
                    "prompt": prompt,
                    "filename": filename,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "size": f"{width}x{height}",
                    "seed": seed if seed else "random",
                }
                all_results.append(result)

                print(f"  ✓ Saved: {filename}")

        except Exception as e:
            print(f"  ✗ Error generating image: {e}")
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

    # Check 1: All images generated successfully
    expected_images = len(prompts) * num_images_per_prompt
    actual_images = len(all_results)
    if actual_images == expected_images:
        print(f"✓ All {expected_images} images generated successfully")
    else:
        print(f"⚠ Generated {actual_images}/{expected_images} images")

    # Check 2: Files exist
    missing_files = []
    for result in all_results:
        filepath = os.path.join(output_dir, result["filename"])
        if not os.path.exists(filepath):
            missing_files.append(result["filename"])

    if not missing_files:
        print("✓ All generated images saved to disk")
    else:
        print(f"✗ Missing files: {missing_files}")

    # Check 3: Check file sizes (images should be substantial)
    small_files = []
    for result in all_results:
        filepath = os.path.join(output_dir, result["filename"])
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            if size_kb < 10:  # Less than 10KB is suspicious
                small_files.append((result["filename"], size_kb))

    if not small_files:
        print("✓ All images have reasonable file sizes")
    else:
        print(f"⚠ Suspiciously small files: {small_files}")

    # Check 4: Metadata saved
    if os.path.exists(metadata_file):
        print("✓ Metadata file saved successfully")
    else:
        print("✗ Metadata file missing")

    print("\n=== Overall Validation Result ===")
    validation_passed = (
        actual_images == expected_images
        and not missing_files
        and not small_files
        and os.path.exists(metadata_file)
    )

    if validation_passed:
        print("✓ Validation PASSED - All images generated successfully")
    else:
        print("✗ Validation FAILED - Some issues detected")

    return all_results


def main():
    """Main function with example prompts"""

    # Example prompts - customize these!
    example_prompts = [
        "A serene mountain landscape at sunset with a lake reflecting the orange sky, photorealistic, 8k, highly detailed",
        "A futuristic cyberpunk city with neon lights, flying cars, and holographic advertisements, digital art",
        "A cute robot reading a book in a cozy library, warm lighting, Studio Ghibli style",
        "An astronaut riding a horse on Mars, cinematic lighting, epic composition",
    ]

    print("\n🎨 Stable Diffusion 3.5 - Text-to-Image Generation")
    print("\nExample prompts:")
    for i, prompt in enumerate(example_prompts, 1):
        print(f"{i}. {prompt}")

    print("\n" + "=" * 60)
    print("Starting generation with first prompt...")
    print("=" * 60)

    # Generate image with the first prompt
    # For demonstration, we'll use just one prompt
    # Uncomment the line below to generate all example prompts
    # generate_images(example_prompts)

    generate_images(
        prompts=[example_prompts[0]],  # Generate just the first one
        num_inference_steps=28,  # Good balance of quality and speed
        guidance_scale=7.0,  # Standard value for SD3
        height=1024,  # SD3.5 works best at 1024x1024
        width=1024,
        seed=42,  # For reproducibility
    )

    print("\n" + "=" * 60)
    print("Tips for better results:")
    print("=" * 60)
    print("• Be specific and descriptive in your prompts")
    print("• Include style keywords (e.g., 'photorealistic', 'oil painting', 'anime')")
    print(
        "• Mention lighting, composition, and quality (e.g., '8k', 'highly detailed')"
    )
    print("• Use guidance_scale 7-9 for best prompt following")
    print("• Increase num_inference_steps (40-50) for higher quality")
    print("• SD3.5 excels at text rendering and complex compositions")
    print("\n" + "=" * 60)
    print("Alternative Models to Try:")
    print("=" * 60)
    print("• FLUX.1 [dev]: 'black-forest-labs/FLUX.1-dev' (best quality)")
    print("• SDXL Turbo: 'stabilityai/sdxl-turbo' (fastest, 1 step)")
    print("• Stable Diffusion 3 Medium: 'stabilityai/stable-diffusion-3-medium'")
    print("\nTo use a different model, change the model_id parameter.")


if __name__ == "__main__":
    main()
