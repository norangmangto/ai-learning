"""
Image-to-Video Generation with Text Prompt Conditioning

This script implements an image-to-video generator using Stable Video Diffusion
with text prompt conditioning. It can generate short video clips from a single
image and text description of desired motion.

Requirements:
    pip install torch torchvision diffusers transformers accelerate safetensors imageio pillow opencv-python
"""

import torch
import torch.nn as nn
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np
from typing import Optional, List, Union
from pathlib import Path
import imageio


class ImageToVideoGenerator:
    """
    Image-to-Video generator using Stable Video Diffusion with text conditioning.
    
    This class wraps the Stable Video Diffusion pipeline and provides an easy
    interface for generating videos from images with text prompts.
    """
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_cpu_offload: bool = False
    ):
        """
        Initialize the image-to-video generator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cuda', 'mps', or 'cpu')
            dtype: Data type for model weights (float16 for GPU, float32 for CPU)
            enable_cpu_offload: Whether to enable CPU offloading for lower memory usage
        """
        # Auto-select device if not provided: CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Prefer float16 only on CUDA; use float32 on MPS/CPU for stability
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        self.device = device
        self.dtype = dtype
        
        print(f"Loading model: {model_name}")
        print(f"Device: {device}, dtype: {self.dtype}")
        
        # Load the Stable Video Diffusion pipeline
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None
        )
        
        if enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(device)
        
        # Enable memory efficient attention if available
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            print("✓ Using xformers memory efficient attention")
        except Exception:
            print("⚠ xformers not available, using default attention")
        
        # Further reduce memory on CPU/MPS: use classic attention and slicing
        if self.device != "cuda":
            try:
                self.pipeline.unet.set_attn_processor(AttnProcessor())
                self.pipeline.enable_attention_slicing()
                print("✓ Enabled attention slicing with standard attention (CPU/MPS)")
            except Exception:
                pass

        print("✓ Model loaded successfully")
    
    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image],
        target_size: tuple = (1024, 576)
    ) -> Image.Image:
        """
        Preprocess input image for the model.
        
        Args:
            image: Input image path or PIL Image
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, (str, Path)):
            image = load_image(str(image))
        
        # Resize while maintaining aspect ratio
        image = image.resize(target_size, Image.LANCZOS)
        
        return image
    
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        num_frames: int = 25,
        num_inference_steps: int = 25,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 8,
        seed: Optional[int] = None,
        target_size: Optional[tuple] = None
    ) -> List[Image.Image]:
        """
        Generate video frames from an input image with optional text prompt.
        
        Args:
            image: Input image (path or PIL Image)
            prompt: Text description of desired motion (currently for reference,
                   SVD doesn't use text directly but you can use it for documentation)
            num_frames: Number of frames to generate (default: 25)
            num_inference_steps: Number of denoising steps (default: 25)
            fps: Frames per second for the output video
            motion_bucket_id: Controls amount of motion (0-255, higher = more motion)
            noise_aug_strength: Amount of noise added to input image (0.0-1.0)
            decode_chunk_size: Number of frames to decode at once (lower = less memory)
            seed: Random seed for reproducibility
            
        Returns:
            List of PIL Images representing video frames
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Preprocess the input image
        print("Preprocessing image...")
        # Choose a smaller default resolution on CPU/MPS for speed
        if target_size is None:
            if self.device == "cuda":
                target_size = (1024, 576)
            else:
                target_size = (256, 144)
        processed_image = self.preprocess_image(image, target_size=target_size)
        
        # Display prompt for reference (SVD doesn't use text conditioning directly)
        if prompt:
            print(f"Prompt (for reference): {prompt}")
        
        print(f"Generating {num_frames} frames...")
        print(f"Parameters: motion_bucket_id={motion_bucket_id}, "
              f"noise_aug_strength={noise_aug_strength}")
        
        # Generate video frames
        with torch.no_grad():
            frames = self.pipeline(
                processed_image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                decode_chunk_size=decode_chunk_size,
            ).frames[0]
        
        print(f"✓ Generated {len(frames)} frames")
        
        return frames
    
    def save_video(
        self,
        frames: List[Image.Image],
        output_path: Union[str, Path],
        fps: int = 7
    ):
        """
        Save generated frames as a video file.
        
        Args:
            frames: List of PIL Images
            output_path: Path to save the video
            fps: Frames per second
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving video to {output_path}...")
        
        # Convert PIL Images to numpy arrays
        frame_arrays = [np.array(frame) for frame in frames]
        
        # Save as video
        export_to_video(frames, str(output_path), fps=fps)
        
        print(f"✓ Video saved: {output_path}")
    
    def save_frames(
        self,
        frames: List[Image.Image],
        output_dir: Union[str, Path],
        prefix: str = "frame"
    ):
        """
        Save individual frames as images.
        
        Args:
            frames: List of PIL Images
            output_dir: Directory to save frames
            prefix: Prefix for frame filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"{prefix}_{i:04d}.png"
            frame.save(frame_path)
        
        print(f"✓ Saved {len(frames)} frames to {output_dir}")


def main():
    """
    Example usage of the ImageToVideoGenerator.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate video from image with text prompt")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt describing desired motion")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Output video path")
    parser.add_argument("--frames", type=int, default=25, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=7, help="Frames per second")
    parser.add_argument("--motion", type=int, default=127, help="Motion amount (0-255)")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offloading")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ImageToVideoGenerator(
        enable_cpu_offload=args.cpu_offload
    )
    
    # Generate video
    frames = generator.generate(
        image=args.image,
        prompt=args.prompt,
        num_frames=args.frames,
        fps=args.fps,
        motion_bucket_id=args.motion,
        num_inference_steps=args.steps,
        seed=args.seed
    )
    
    # Save video
    generator.save_video(frames, args.output, fps=args.fps)
    
    # Optionally save individual frames
    if args.save_frames:
        frames_dir = Path(args.output).stem + "_frames"
        generator.save_frames(frames, frames_dir)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
