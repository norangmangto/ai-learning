"""
Training script for Image-to-Video Generation Model

This script implements training for an image-to-video generation model with
text prompt conditioning. It uses a latent diffusion approach with temporal
layers for generating coherent video sequences.

Note: This is a simplified training setup. Production training would require
substantial compute resources and large-scale video datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json


class VideoDataset(Dataset):
    """
    Dataset for video sequences with text descriptions.
    
    Expected structure:
        data_dir/
            video_001/
                frame_0000.jpg
                frame_0001.jpg
                ...
                metadata.json  # Contains 'caption' field
            video_002/
            ...
    """
    
    def __init__(
        self,
        data_dir: str,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (256, 256),
        frame_stride: int = 1
    ):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        
        # Find all video directories
        self.video_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self) -> int:
        return len(self.video_dirs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_dir = self.video_dirs[idx]
        
        # Load frames
        frame_files = sorted(list(video_dir.glob("frame_*.jpg")))
        
        # Sample frames with stride
        sampled_indices = list(range(0, len(frame_files), self.frame_stride))[:self.num_frames]
        
        # Pad if necessary
        while len(sampled_indices) < self.num_frames:
            sampled_indices.append(sampled_indices[-1])
        
        frames = []
        for idx in sampled_indices:
            frame = Image.open(frame_files[idx]).convert("RGB")
            frame = self.transform(frame)
            frames.append(frame)
        
        video = torch.stack(frames)  # Shape: (num_frames, C, H, W)
        
        # Load caption
        metadata_path = video_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                caption = metadata.get("caption", "")
        else:
            caption = ""
        
        return {
            "video": video,
            "caption": caption,
            "first_frame": video[0]  # Conditioning image
        }


class TemporalAttentionBlock(nn.Module):
    """
    Temporal attention layer for processing video sequences.
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(
            channels, 
            num_heads, 
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Reshape for temporal attention
        x = x.permute(0, 3, 4, 2, 1)  # (B, H, W, T, C)
        x = x.reshape(B * H * W, T, C)
        
        # Apply attention
        residual = x
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x + residual
        
        # Reshape back
        x = x.reshape(B, H, W, T, C)
        x = x.permute(0, 4, 3, 1, 2)  # (B, C, T, H, W)
        
        return x


class ImageToVideoModel(nn.Module):
    """
    Image-to-Video generation model with text conditioning.
    
    Architecture:
        1. Encode first frame and text
        2. Generate latent video sequence using temporal U-Net
        3. Decode to pixel space
    """
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        unet_channels: int = 320,
        num_frames: int = 16
    ):
        super().__init__()
        
        self.vae = vae
        self.text_encoder = text_encoder
        self.num_frames = num_frames
        
        # Freeze VAE and text encoder
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Temporal layers (simplified version)
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(unet_channels // 2),
            TemporalAttentionBlock(unet_channels),
            TemporalAttentionBlock(unet_channels * 2)
        ])
        
        # Frame expansion network
        self.frame_expander = nn.Sequential(
            nn.Conv3d(4, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        latent = self.vae.encode(image).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent
    
    @torch.no_grad()
    def encode_text(self, text_inputs: Dict) -> torch.Tensor:
        """Encode text to embeddings."""
        text_embeddings = self.text_encoder(**text_inputs)[0]
        return text_embeddings
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space."""
        latents = latents / self.vae.config.scaling_factor
        frames = []
        for i in range(latents.shape[2]):
            frame = self.vae.decode(latents[:, :, i]).sample
            frames.append(frame)
        return torch.stack(frames, dim=2)
    
    def forward(
        self,
        first_frame: torch.Tensor,
        text_embeddings: torch.Tensor,
        target_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate video latents from first frame and text.
        
        Args:
            first_frame: (B, C, H, W) - First frame
            text_embeddings: (B, seq_len, D) - Text embeddings
            target_frames: (B, C, T, H, W) - Target frames for training
            
        Returns:
            (B, C, T, H, W) - Generated video latents
        """
        B = first_frame.shape[0]
        
        # Encode first frame to latent
        first_latent = self.encode_image(first_frame)  # (B, 4, H//8, W//8)
        
        # Expand to temporal dimension
        # Repeat first frame
        latents = first_latent.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
        
        # Generate temporal variations
        latents = self.frame_expander(latents)
        
        return latents


class ImageToVideoTrainer:
    """
    Trainer for image-to-video generation model.
    """
    
    def __init__(
        self,
        model: ImageToVideoModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        device: str = "cuda",
        output_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        # Noise scheduler for diffusion
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear"
        )
        
        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    
    def train_step(self, batch: Dict) -> float:
        """Single training step."""
        self.model.train()
        
        videos = batch["video"].to(self.device)  # (B, T, C, H, W)
        first_frames = batch["first_frame"].to(self.device)
        captions = batch["caption"]
        
        # Tokenize captions
        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode text
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_inputs)
        
        # Encode video frames to latents
        B, T, C, H, W = videos.shape
        videos_flat = videos.view(B * T, C, H, W)
        
        with torch.no_grad():
            latents_flat = self.model.encode_image(videos_flat)
        
        latents = latents_flat.view(B, T, *latents_flat.shape[1:])
        latents = latents.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # Add noise for diffusion
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=self.device
        ).long()
        
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        predicted_latents = self.model(first_frames, text_embeddings, noisy_latents)
        
        # Compute loss
        loss = F.mse_loss(predicted_latents, latents)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": avg_loss
                }, checkpoint_path)
                print(f"✓ Saved checkpoint: {checkpoint_path}")


def main():
    """
    Example training script.
    """
    # Configuration
    config = {
        "data_dir": "path/to/video_dataset",
        "num_frames": 16,
        "frame_size": (256, 256),
        "batch_size": 2,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print("Initializing dataset...")
    train_dataset = VideoDataset(
        data_dir=config["data_dir"],
        num_frames=config["num_frames"],
        frame_size=config["frame_size"]
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    
    print("Loading pre-trained models...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    
    print("Initializing model...")
    model = ImageToVideoModel(
        vae=vae,
        text_encoder=text_encoder,
        num_frames=config["num_frames"]
    )
    
    print("Initializing trainer...")
    trainer = ImageToVideoTrainer(
        model=model,
        train_dataloader=train_dataloader,
        learning_rate=config["learning_rate"],
        num_epochs=config["num_epochs"],
        device=config["device"]
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
