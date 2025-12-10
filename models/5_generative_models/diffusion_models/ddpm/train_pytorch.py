"""
Denoising Diffusion Probabilistic Models (DDPM)

This script demonstrates:
1. Forward diffusion process (adding noise)
2. Reverse diffusion process (denoising)
3. Training a noise prediction network
4. Sampling from learned distribution
5. Understanding the mathematical foundation

Dataset: MNIST
Reference: Ho et al., "Denoising Diffusion Probabilistic Models"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm

# Configuration
CONFIG = {
    'dataset': 'mnist',
    'image_size': 28,
    'channels': 1,
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    # Diffusion parameters
    'timesteps': 1000,  # Number of diffusion steps
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'output_dir': 'results/ddpm'
}

class LinearSchedule:
    """Linear noise schedule"""
    def __init__(self, timesteps, beta_start, beta_end):
        self.timesteps = timesteps

        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Coefficients
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod_m_one', torch.sqrt(1.0 / alphas_cumprod - 1.0))

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.float())

class SimpleUNet(nn.Module):
    """Simple U-Net for noise prediction"""
    def __init__(self, channels=1, time_dim=128):
        super(SimpleUNet, self).__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.enc1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)

        # Bottleneck
        self.bottleneck = nn.Conv2d(128, 128, 3, padding=1)

        # Decoder
        self.dec1 = nn.Conv2d(256, 64, 3, padding=1)  # +128 from skip
        self.dec2 = nn.Conv2d(128, channels, 3, padding=1)  # +64 from skip

        self.act = nn.SiLU()
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1))
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # (B, time_dim, 1, 1)

        # Encoder
        x1 = self.act(self.enc1(x) + t_emb)
        x = self.pool(x1)

        x2 = self.act(self.enc2(x) + t_emb)
        x = self.pool(x2)

        # Bottleneck
        x = self.act(self.bottleneck(x) + t_emb)

        # Decoder
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.act(self.dec1(x) + t_emb)

        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)

        return x

class DDPM:
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, model, schedule, device):
        self.model = model
        self.schedule = schedule
        self.device = device

    def add_noise(self, x0, t):
        """
        Forward diffusion: add noise to clean image
        q(x_t | x_0) = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        """
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1, 1)

        epsilon = torch.randn_like(x0)
        x_t = sqrt_alpha * x0 + sqrt_one_minus_alpha * epsilon

        return x_t, epsilon

    def sample_t(self, batch_size):
        """Sample random timestep"""
        return torch.randint(0, self.schedule.timesteps, (batch_size,)).to(self.device)

    def p_loss(self, x0, t):
        """Compute prediction loss"""
        # Forward diffusion
        x_t, noise = self.add_noise(x0, t)

        # Predict noise
        noise_pred = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """Reverse diffusion: denoise one step"""
        betas = self.schedule.betas
        alphas = self.schedule.alphas
        alphas_cumprod = self.schedule.alphas_cumprod

        # Reshape
        betas_t = betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas[t]).view(-1, 1, 1, 1)

        # Predict noise
        noise_pred = self.model(x_t, t)

        # Mean
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alpha_cumprod_t)

        # Variance
        posterior_variance_t = (
            (1 - alphas_cumprod[t - 1]) / (1 - alphas_cumprod[t]) * betas[t]
        )
        posterior_variance_t = posterior_variance_t.view(-1, 1, 1, 1)

        # Sample
        if t.min() > 0:
            noise = torch.randn_like(x_t)
            x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x_prev = model_mean

        return x_prev

    @torch.no_grad()
    def sample(self, batch_size, shape):
        """Generate images from noise"""
        device = self.device

        # Start from noise
        x = torch.randn(batch_size, *shape).to(device)

        # Reverse diffusion
        for t in tqdm(reversed(range(self.schedule.timesteps)), total=self.schedule.timesteps, desc="Sampling"):
            t_batch = torch.full((batch_size,), t, dtype=torch.long).to(device)
            x = self.p_sample(x, t_batch)

        return x.clamp(-1, 1)

def load_data(dataset_name='mnist', image_size=28):
    """Load dataset"""
    print("="*80)
    print("LOADING DATASET")
    print("="*80)

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root='data',
            train=True,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers']
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Total images: {len(dataset)}")
    print(f"Batches: {len(dataloader)}")

    return dataloader

def train_ddpm(ddpm, dataloader, device):
    """Train DDPM"""
    print("\n" + "="*80)
    print("TRAINING DDPM")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Timesteps: {CONFIG['timesteps']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Device: {device}")

    # Optimizer
    optimizer = optim.Adam(ddpm.model.parameters(), lr=CONFIG['learning_rate'])

    # Output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    losses = []

    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        total_loss = 0

        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            # Sample random timestep
            t = ddpm.sample_t(x.size(0))

            # Compute loss
            loss = ddpm.p_loss(x, t)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Loss: {avg_loss:.4f} "
              f"Time: {epoch_time:.2f}s")

        # Sample periodically
        if (epoch + 1) % 10 == 0:
            ddpm.model.eval()
            with torch.no_grad():
                samples = ddpm.sample(16, (CONFIG['channels'], CONFIG['image_size'], CONFIG['image_size']))

            grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
            save_image(grid, output_dir / f'epoch_{epoch+1:03d}.png')

            ddpm.model.train()

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDPM Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'losses.png', dpi=150, bbox_inches='tight')
    print(f"\nLoss plot saved to: {output_dir}/losses.png")
    plt.close()

    return ddpm

def main():
    print("="*80)
    print("DENOISING DIFFUSION PROBABILISTIC MODELS (DDPM)")
    print("="*80)

    print(f"\nDevice: {CONFIG['device']}")

    # Load data
    dataloader = load_data(CONFIG['dataset'], CONFIG['image_size'])

    # Create schedule
    print("\n" + "="*80)
    print("CREATING NOISE SCHEDULE")
    print("="*80)

    schedule = LinearSchedule(
        timesteps=CONFIG['timesteps'],
        beta_start=CONFIG['beta_start'],
        beta_end=CONFIG['beta_end']
    )

    print(f"\nNoise schedule:")
    print(f"  Timesteps: {CONFIG['timesteps']}")
    print(f"  Beta start: {CONFIG['beta_start']}")
    print(f"  Beta end: {CONFIG['beta_end']}")

    # Create model
    print("\n" + "="*80)
    print("CREATING NOISE PREDICTION NETWORK")
    print("="*80)

    model = SimpleUNet(channels=CONFIG['channels']).to(CONFIG['device'])

    print(f"\nParameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Create DDPM
    ddpm = DDPM(model, schedule, CONFIG['device'])

    # Train
    ddpm = train_ddpm(ddpm, dataloader, CONFIG['device'])

    # Generate final samples
    print("\n" + "="*80)
    print("GENERATING SAMPLES")
    print("="*80)

    ddpm.model.eval()
    with torch.no_grad():
        samples = ddpm.sample(64, (CONFIG['channels'], CONFIG['image_size'], CONFIG['image_size']))

    grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
    output_dir = Path(CONFIG['output_dir'])
    save_image(grid, output_dir / 'final_samples.png')
    print(f"\nFinal samples saved to: {output_dir}/final_samples.png")

    # Save model
    torch.save(model.state_dict(), output_dir / 'ddpm.pth')
    print(f"Model saved to: {output_dir}/ddpm.pth")

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    print("\nDDPM Key Concepts:")
    print("✓ Forward process: gradually add noise to data")
    print("✓ Reverse process: iteratively denoise to generate samples")
    print("✓ Training: predict noise at each timestep")
    print("✓ Sampling: start from noise, denoise in reverse")

    print("\nMathematical Foundation:")
    print("- Forward: q(x_t|x_0) = sqrt(α_t)*x_0 + sqrt(1-α_t)*ε")
    print("- Reverse: p(x_{t-1}|x_t) = N(μ, Σ)")
    print("- Loss: E||ε - ε_θ(x_t, t)||²")

    print("\nAdvantages:")
    print("+ Stable training (no adversarial dynamics)")
    print("+ High-quality generations")
    print("+ Can handle unconditional and conditional")
    print("+ Better than GANs on many benchmarks")

    print("\nLimitations:")
    print("- Slow sampling (many reverse steps)")
    print("- High training cost")
    print("- Requires noise prediction network")

    print("\nVariants:")
    print("- DDIM: Faster sampling with fewer steps")
    print("- Conditional DDPM: Class-guided generation")
    print("- Latent Diffusion: Efficient in latent space")
    print("- Score-based: Alternative formulation")

if __name__ == '__main__':
    main()
