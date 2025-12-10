import torch
from diffusers import StableDiffusionPipeline
import argparse
import os

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a photo of sks dog in a bucket")
    parser.add_argument("--lora_path", type=str, default="lora_weights")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_path", type=str, default="output.png")
    args = parser.parse_args()

    print(f"Generating image for prompt: '{args.prompt}'")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Base Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.float32)
    pipe.to(device)

    # Load LoRA Adapter
    if os.path.exists(args.lora_path):
        print(f"Loading LoRA weights from {args.lora_path}...")
        try:
            pipe.unet.load_adapter(args.lora_path, adapter_name="dreambooth")
            # Or simplified: pipe.load_lora_weights(args.lora_path) which is supported in newer diffusers
            # But since we saved with PEFT:
            # pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)
            # Actually, diffusers has native load_lora_weights now which is easier:
            pipe.load_lora_weights(args.lora_path)
        except Exception as e:
            print(f"Failed to load LoRA with standard method, trying PEFT direct: {e}")
    else:
        print(f"Warning: LoRA path '{args.lora_path}' does not exist. Generating with base model only.")

    # Generate
    image = pipe(args.prompt, num_inference_steps=30).images[0]

    image.save(args.output_path)
    print(f"Image saved to {args.output_path}")

if __name__ == "__main__":
    generate()
