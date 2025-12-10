import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# Diffusers & Transformers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(description="Simple Dreambooth LoRA Trainer")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing instance images")
    parser.add_argument("--output_dir", type=str, default="lora_weights", help="Where to save weights")
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks dog", help="Prompt with unique identifier")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to pretrained model")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    return parser.parse_args()

class DreamBoothDataset(Dataset):
    def __init__(self, image_dir, instance_prompt, tokenizer, size=512):
        self.image_dir = image_dir
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
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

        # Tokenize prompt
        inputs = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        return {"pixel_values": img, "input_ids": inputs.input_ids[0]}

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def train(args):
    print(f"Initializing Dreambooth LoRA training for: {args.instance_prompt}")

    # 1. Setup Accelerator / Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Models
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    # Freeze standard models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 3. Inject LoRA
    print("Injecting LoRA adapters...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.to(device)

    # 4. Dataset & Dataloader
    dataset = DreamBoothDataset(args.image_dir, args.instance_prompt, tokenizer, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # 6. Training Loop
    print("Starting training...")
    unet.train()

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps))

    while global_step < args.max_train_steps:
        for batch in dataloader:
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Encode inputs
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Get text embeddings
            encoder_hidden_states = text_encoder(input_ids)[0]

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

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

    # 7. Save
    print(f"Saving LoRA weights to {args.output_dir}...")
    unet.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' not found. Please create it and add 3-5 images of your subject.")
    else:
        train(args)
