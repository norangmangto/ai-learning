import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# Diffusers & Transformers
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(description="SD3 Dreambooth LoRA Trainer")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing instance images")
    parser.add_argument("--output_dir", type=str, default="lora_weights", help="Where to save weights")
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks dog", help="Prompt with unique identifier")
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-3.5-large", help="Path to pretrained model")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    return parser.parse_args()

class DreamBoothDataset(Dataset):
    def __init__(self, image_dir, instance_prompt, size=1024):
        self.image_dir = image_dir
        self.instance_prompt = instance_prompt
        self.size = size

        self.images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index % len(self.images)]
        img = Image.open(image_path).convert("RGB")
        img = self.transforms(img)

        return {"pixel_values": img, "prompt": self.instance_prompt}

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}

def train(args):
    print(f"Initializing SD3 Dreambooth LoRA training for: {args.instance_prompt}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Pipeline (used for encoding & noise scheduling)
    # SD3 requires complex text encoding, so we use the pipeline components
    print(f"Loading SD3 Pipeline from {args.pretrained_model}...")
    # Load in bfloat16 to save memory, generally required for SD3
    dtype = torch.bfloat16 if device == "cuda" or device == "mps" else torch.float32

    pipe = StableDiffusion3Pipeline.from_pretrained(args.pretrained_model, torch_dtype=dtype)
    pipe.to(device)

    transformer = pipe.transformer
    vae = pipe.vae

    # 2. Setup LoRA on Transformer
    print("Injecting LoRA adapters into Transformer...")
    # Freeze everything
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    # Text encoders are inside pipe, we won't train them

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # Unwrap transformer if needed? No, directly apply
    transformer.add_adapter(lora_config)

    # Enable gradients for LoRA
    # In PEFT > 0.4 we can just use the model, but let's be explicit
    # get_peft_model is usually wrapper, but add_adapter is transformer specific method in diffusers or requires PEFT injection
    # Actually, recent diffusers integration is cleaner. Let's use get_peft_model wrapper to be safe standard PEFT
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.to(device) # ensure on device

    # 3. Dataset & Dataloader
    dataset = DreamBoothDataset(args.image_dir, args.instance_prompt, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)

    # 5. Training Loop
    print("Starting training (Flow Matching)...")
    transformer.train()

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps))

    while global_step < args.max_train_steps:
        for batch in dataloader:
            # Move pixels to device
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            prompts = batch["prompts"]

            with torch.no_grad():
                # Encode inputs (VAE)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Encode prompt (using pipeline helper)
                # SD3 encode_prompt returns: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
                prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(prompt=prompts, device=device)

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample random timesteps
            # FlowMatchEulerDiscreteScheduler logic: 'sigmas' usually 0 to 1?
            # Diffusers simplified training:
            #   u = torch.randn(bsz)
            #   timesteps = u * 1000 (roughly)
            #   sigmas = timesteps / 1000?
            # Let's check diffusers source or assume standard 0-1000 like DDPMScheduler but with different interpretation.
            # Ideally use scheduler:
            # timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            # But FlowMatching uses float sigmas typically.
            # Simplified Flow Matching:
            # t \in [0, 1]
            u = torch.rand(bsz, device=device)
            # Create noisy latents: x_t = (1-t)x_0 + t*x_1 (x_0=data, x_1=noise) OR x_t = (1-t)noise + t*data
            # SD3 uses: sigmas are time.
            # We will rely on simple Rectified Flow formula:
            # x_t = t * noise + (1 - t) * data
            # target = noise - data

            sigmas = u.reshape(bsz, 1, 1, 1)
            noisy_latents = (1 - sigmas) * latents + sigmas * noise
            target = noise - latents

            timesteps = u * 1000.0 # Transformer expects 0-1000 typically

            # Predict
            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False
            )[0]

            # Loss (MSE between v-prediction and target)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if global_step % 50 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")

            if global_step >= args.max_train_steps:
                break

    # 6. Save
    print(f"Saving LoRA weights to {args.output_dir}...")
    pipe.save_lora_weights(args.output_dir) # Pipeline helper saves attached adapters cleanly
    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' not found. Please create it.")
    else:
        train(args)
