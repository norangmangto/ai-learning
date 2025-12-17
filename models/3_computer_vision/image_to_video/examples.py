"""
Example script demonstrating how to use the Image-to-Video generator.

This script shows practical usage examples with different scenarios and
parameter configurations.
"""

from generate_video import ImageToVideoGenerator
from PIL import Image
import numpy as np
from pathlib import Path


def create_sample_image(output_path: str = "sample_image.jpg"):
    """
    Create a simple sample image for testing (if you don't have one).
    
    In practice, you would use your own images.
    """
    # Create a gradient image with a circle
    size = (512, 512)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Create gradient background
    for i in range(size[1]):
        image[i, :, 0] = int(255 * (i / size[1]))  # Red gradient
        image[i, :, 2] = int(255 * (1 - i / size[1]))  # Blue gradient
    
    # Add a circle
    center = (size[0] // 2, size[1] // 2)
    radius = 100
    y, x = np.ogrid[:size[1], :size[0]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    image[mask] = [255, 255, 0]  # Yellow circle
    
    # Save image
    Image.fromarray(image).save(output_path)
    print(f"Created sample image: {output_path}")
    return output_path


def example_basic_usage():
    """
    Basic example: Generate a simple video from an image.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Create or use an existing image
    image_path = "sample_image.jpg"
    if not Path(image_path).exists():
        image_path = create_sample_image(image_path)
    
    # Initialize generator
    print("\nInitializing generator...")
    # Use a smaller model on CPU/MPS for lower memory usage
    import torch
    model_name = (
        "stabilityai/stable-video-diffusion-img2vid-xt"
        if torch.cuda.is_available()
        else "stabilityai/stable-video-diffusion-img2vid"
    )
    generator = ImageToVideoGenerator(model_name=model_name, enable_cpu_offload=True)
    
    # Generate video with basic parameters
    print("\nGenerating video...")
    frames = generator.generate(
        image=image_path,
        prompt="gentle motion, subtle movement",
        num_frames=2,  # Tiny run to fit CPU/MPS memory
        fps=7,
        motion_bucket_id=127,
        target_size=(128, 72),
        num_inference_steps=8,
        decode_chunk_size=2
    )
    
    # Save video
    generator.save_video(frames, "output_basic.mp4", fps=7)
    print("\n✓ Basic example complete!")


def example_high_motion():
    """
    Example: Generate video with high motion.
    """
    print("\n" + "="*60)
    print("Example 2: High Motion Video")
    print("="*60)
    
    image_path = "sample_image.jpg"
    if not Path(image_path).exists():
        image_path = create_sample_image(image_path)
    
    generator = ImageToVideoGenerator(enable_cpu_offload=True)
    
    # High motion bucket ID for more dynamic video
    frames = generator.generate(
        image=image_path,
        prompt="fast motion, dynamic movement, zooming in",
        num_frames=14,
        fps=7,
        motion_bucket_id=180,  # Higher = more motion
        noise_aug_strength=0.05  # More variation from original
    )
    
    generator.save_video(frames, "output_high_motion.mp4", fps=7)
    print("\n✓ High motion example complete!")


def example_low_motion():
    """
    Example: Generate video with minimal motion (more similar to input).
    """
    print("\n" + "="*60)
    print("Example 3: Low Motion Video")
    print("="*60)
    
    image_path = "sample_image.jpg"
    if not Path(image_path).exists():
        image_path = create_sample_image(image_path)
    
    generator = ImageToVideoGenerator(enable_cpu_offload=True)
    
    # Low motion bucket ID for subtle changes
    frames = generator.generate(
        image=image_path,
        prompt="very subtle motion, almost static, slight breathing",
        num_frames=14,
        fps=7,
        motion_bucket_id=50,  # Lower = less motion
        noise_aug_strength=0.01  # Stay closer to original
    )
    
    generator.save_video(frames, "output_low_motion.mp4", fps=7)
    print("\n✓ Low motion example complete!")


def example_longer_video():
    """
    Example: Generate a longer video sequence.
    """
    print("\n" + "="*60)
    print("Example 4: Longer Video")
    print("="*60)
    
    image_path = "sample_image.jpg"
    if not Path(image_path).exists():
        image_path = create_sample_image(image_path)
    
    generator = ImageToVideoGenerator(enable_cpu_offload=True)
    
    # Generate longer sequence (requires more memory and time)
    frames = generator.generate(
        image=image_path,
        prompt="smooth continuous motion",
        num_frames=25,  # Maximum for standard SVD
        fps=7,
        motion_bucket_id=127
    )
    
    generator.save_video(frames, "output_long.mp4", fps=7)
    
    # Also save individual frames
    generator.save_frames(frames, "output_frames")
    
    print("\n✓ Longer video example complete!")


def example_deterministic():
    """
    Example: Generate reproducible videos using seed.
    """
    print("\n" + "="*60)
    print("Example 5: Deterministic Generation (with seed)")
    print("="*60)
    
    image_path = "sample_image.jpg"
    if not Path(image_path).exists():
        image_path = create_sample_image(image_path)
    
    generator = ImageToVideoGenerator(enable_cpu_offload=True)
    
    # Generate with fixed seed - will produce same result every time
    seed = 42
    frames = generator.generate(
        image=image_path,
        prompt="consistent motion",
        num_frames=14,
        fps=7,
        seed=seed
    )
    
    generator.save_video(frames, "output_seeded.mp4", fps=7)
    print(f"\n✓ Generated with seed {seed} - run again with same seed for identical results!")


def example_batch_processing():
    """
    Example: Process multiple images in sequence.
    """
    print("\n" + "="*60)
    print("Example 6: Batch Processing")
    print("="*60)
    
    # Create sample images
    image_paths = []
    for i in range(3):
        path = f"sample_image_{i}.jpg"
        if not Path(path).exists():
            path = create_sample_image(path)
        image_paths.append(path)
    
    generator = ImageToVideoGenerator(enable_cpu_offload=True)
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}...")
        
        frames = generator.generate(
            image=image_path,
            prompt=f"motion example {i+1}",
            num_frames=14,
            fps=7
        )
        
        output_path = f"output_batch_{i}.mp4"
        generator.save_video(frames, output_path, fps=7)
    
    print("\n✓ Batch processing complete!")


def example_different_sizes():
    """
    Example: Working with different image sizes.
    """
    print("\n" + "="*60)
    print("Example 7: Different Image Sizes")
    print("="*60)
    
    generator = ImageToVideoGenerator(enable_cpu_offload=True)
    
    # Create images of different sizes
    sizes = [(512, 512), (768, 512), (1024, 576)]
    
    for i, size in enumerate(sizes):
        # Create sample image
        image = np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)
        image_path = f"sample_{size[0]}x{size[1]}.jpg"
        Image.fromarray(image).save(image_path)
        
        print(f"\nProcessing {size[0]}x{size[1]} image...")
        
        frames = generator.generate(
            image=image_path,
            prompt=f"{size[0]}x{size[1]} resolution test",
            num_frames=14,
            fps=7
        )
        
        output_path = f"output_{size[0]}x{size[1]}.mp4"
        generator.save_video(frames, output_path, fps=7)
    
    print("\n✓ Different sizes example complete!")


def main():
    """
    Run all examples or select specific ones.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Image-to-Video Examples")
    parser.add_argument(
        "--example",
        type=str,
        choices=["all", "basic", "high_motion", "low_motion", "longer", 
                 "deterministic", "batch", "sizes"],
        default="basic",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    examples = {
        "basic": example_basic_usage,
        "high_motion": example_high_motion,
        "low_motion": example_low_motion,
        "longer": example_longer_video,
        "deterministic": example_deterministic,
        "batch": example_batch_processing,
        "sizes": example_different_sizes
    }
    
    print("="*60)
    print("Image-to-Video Generation Examples")
    print("="*60)
    print("\nNote: First run will download the model (~7GB)")
    print("Subsequent runs will use the cached model.")
    
    if args.example == "all":
        for name, func in examples.items():
            func()
    else:
        examples[args.example]()
    
    print("\n" + "="*60)
    print("All examples complete!")
    print("="*60)
    print("\nTips for best results:")
    print("  - Use high-quality, well-lit input images")
    print("  - Experiment with motion_bucket_id (0-255)")
    print("  - Lower noise_aug_strength stays closer to input")
    print("  - Use seed for reproducible results")
    print("  - Enable CPU offload to reduce VRAM usage")


if __name__ == "__main__":
    main()
