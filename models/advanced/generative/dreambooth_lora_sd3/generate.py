import torch
from diffusers import StableDiffusion3Pipeline
import argparse
import os

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a photo of sks dog in a bucket")
    parser.add_argument("--lora_path", type=str, default="lora_weights")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--output_path", type=str, default="output.png")
    args = parser.parse_args()

    print(f"Generating image for prompt: '{args.prompt}'")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Base Pipeline
    print(f"Loading Base SD3 Model: {args.base_model}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    pipe.to(device)

    # Load LoRA Adapter
    if os.path.exists(args.lora_path):
        print(f"Loading LoRA weights from {args.lora_path}...")
        try:
            pipe.load_lora_weights(args.lora_path)
            print("LoRA loaded successfully.")
        except Exception as e:
            print(f"Failed to load LoRA: {e}")
    else:
        print(f"Warning: LoRA path '{args.lora_path}' does not exist. Generating with base model only.")

    # Generate
    print("Generating...")
    image = pipe(args.prompt, num_inference_steps=28, guidance_scale=4.5).images[0]

    image.save(args.output_path)
    print(f"Image saved to {args.output_path}")

if __name__ == "__main__":
    generate()
