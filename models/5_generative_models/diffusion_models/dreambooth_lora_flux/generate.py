import torch
from diffusers import FluxPipeline
import argparse
import os

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a photo of sks dog in a bucket")
    parser.add_argument("--lora_path", type=str, default="flux_lora_weights")
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-dev")
    # Flux Dev is gated/needs license acceptance, users might need token.
    # Schnell is apache 2.0. We default to dev as requested "powerful".
    parser.add_argument("--output_path", type=str, default="output.png")
    args = parser.parse_args()

    print(f"Generating image for prompt: '{args.prompt}'")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading Base Flux Model: {args.base_model}...")
    # Flux requires bfloat16
    dtype = torch.bfloat16
    pipe = FluxPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
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

    print("Generating...")
    # Flux guidance scale 3.5 is standard
    image = pipe(args.prompt, num_inference_steps=28, guidance_scale=3.5).images[0]

    image.save(args.output_path)
    print(f"Image saved to {args.output_path}")

if __name__ == "__main__":
    generate()
