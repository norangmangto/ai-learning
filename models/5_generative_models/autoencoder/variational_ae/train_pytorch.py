"""
Variational Autoencoder (VAE) Implementation

This script demonstrates:
1. Building a VAE from scratch
2. Learning a latent distribution
3. Sampling and reconstructing images
4. Interpolation in latent space
5. Understanding KL divergence

Dataset: MNIST, Fashion-MNIST
Architecture: VAE with Gaussian latent space
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Configuration
CONFIG = {
    "dataset": "mnist",  # 'mnist' or 'fashion_mnist'
    "image_size": 28,
    "channels": 1,
    "latent_dim": 20,
    "hidden_dim": 400,
    "batch_size": 128,
    "epochs": 30,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "beta": 1.0,  # KL divergence weight
    "output_dir": "results/vae",
}


class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, image_size=28, channels=1, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input dimension (flattened)
        input_dim = channels * image_size * image_size

        # ===== ENCODER =====
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ===== DECODER =====
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Encoder: x -> mu, logvar"""
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def decode(self, z):
        """Decoder: z -> reconstruction"""
        h = F.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(h))
        return x_recon

    def forward(self, x):
        """Full VAE pass"""
        x_flat = x.view(x.size(0), -1)

        # Encode
        mu, logvar = self.encode(x_flat)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decode(z)
        x_recon = x_recon.view(x.size())

        return x_recon, mu, logvar, z

    def sample(self, num_samples=64):
        """Sample from standard normal distribution"""
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        samples = self.decode(z)
        return samples


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + beta * KL Divergence

    Reconstruction Loss: Binary Cross-Entropy
    KL Divergence: Measure of difference from standard normal distribution
    """
    # Reconstruction loss (BCE for pixel values in [0, 1])
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="mean")

    # KL divergence
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def load_data(dataset_name="mnist", image_size=28):
    """Load dataset"""
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    transform = T.Compose([T.Resize(image_size), T.ToTensor()])

    if dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST(
            root="data", train=True, download=True, transform=transform
        )
    elif dataset_name == "fashion_mnist":
        dataset = torchvision.datasets.FashionMNIST(
            root="data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Total images: {len(dataset)}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batches: {len(dataloader)}")

    return dataloader


def train_vae(model, dataloader, device):
    """Train VAE"""
    print("\n" + "=" * 80)
    print("TRAINING VAE")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Learning rate: {CONFIG['lr']}")
    print(f"  Latent dimension: {CONFIG['latent_dim']}")
    print(f"  Beta (KL weight): {CONFIG['beta']}")
    print(f"  Device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    losses = {"total": [], "recon": [], "kl": []}

    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_start = time.time()

        total_loss = 0
        total_recon = 0
        total_kl = 0

        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)

            # Forward pass
            x_recon, mu, logvar, z = model(x)

            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, CONFIG["beta"])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        n_batches = len(dataloader)
        epoch_time = time.time() - epoch_start

        # Average losses
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches

        losses["total"].append(avg_loss)
        losses["recon"].append(avg_recon)
        losses["kl"].append(avg_kl)

        print(
            f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
            f"Loss: {avg_loss:.4f} "
            f"Recon: {avg_recon:.4f} "
            f"KL: {avg_kl:.4f} "
            f"Time: {epoch_time:.2f}s"
        )

        # Save samples
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Reconstruction samples
                x_sample, _ = next(iter(dataloader))
                x_sample = x_sample[:16].to(device)
                x_recon, _, _, _ = model(x_sample)

                # Combine original and reconstructions
                combined = torch.cat([x_sample, x_recon], dim=0)
                grid = make_grid(combined, nrow=8)
                save_image(grid, output_dir / f"recon_epoch_{epoch+1:03d}.png")

                # Generated samples
                samples = model.sample(64)
                grid = make_grid(samples, nrow=8)
                save_image(grid, output_dir / f"generated_epoch_{epoch+1:03d}.png")

    # Plot losses
    plot_losses(losses, output_dir)

    return model, losses


def plot_losses(losses, output_dir):
    """Plot training losses"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(losses["total"])
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(losses["recon"])
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(losses["kl"])
    plt.title("KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "losses.png", dpi=150, bbox_inches="tight")
    print(f"\nLoss plot saved to: {output_dir}/losses.png")
    plt.close()


def visualize_latent_space(model, dataloader, device):
    """Visualize 2D latent space"""
    print("\n" + "=" * 80)
    print("VISUALIZING LATENT SPACE")
    print("=" * 80)

    if CONFIG["latent_dim"] != 2:
        print("Latent dimension is not 2, skipping visualization")
        return

    model.eval()

    all_z = []
    all_y = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mu, _ = model.encode(x.view(x.size(0), -1))
            all_z.append(mu.cpu().numpy())
            all_y.append(y.numpy())

    z = np.concatenate(all_z)
    y = np.concatenate(all_y)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=5, alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("VAE Latent Space")
    plt.grid(True, alpha=0.3)

    output_dir = Path(CONFIG["output_dir"])
    plt.savefig(output_dir / "latent_space.png", dpi=150, bbox_inches="tight")
    print(f"Latent space plot saved to: {output_dir}/latent_space.png")
    plt.close()


def interpolate_latent(model, device, num_steps=10):
    """Interpolate between two points in latent space"""
    print("\n" + "=" * 80)
    print("LATENT SPACE INTERPOLATION")
    print("=" * 80)

    model.eval()

    z1 = torch.randn(1, CONFIG["latent_dim"], device=device)
    z2 = torch.randn(1, CONFIG["latent_dim"], device=device)

    alphas = torch.linspace(0, 1, num_steps, device=device)

    images = []
    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            img = model.decode(z)
            images.append(img)

    images = torch.cat(images, dim=0)
    grid = make_grid(images, nrow=num_steps)

    output_dir = Path(CONFIG["output_dir"])
    save_image(grid, output_dir / "interpolation.png")
    print(f"Interpolation saved to: {output_dir}/interpolation.png")


def main():
    print("=" * 80)
    print("VARIATIONAL AUTOENCODER (VAE)")
    print("=" * 80)

    print(f"\nDevice: {CONFIG['device']}")

    # Load data
    dataloader = load_data(CONFIG["dataset"], CONFIG["image_size"])

    # Create model
    print("\n" + "=" * 80)
    print("CREATING VAE MODEL")
    print("=" * 80)

    model = VAE(
        image_size=CONFIG["image_size"],
        channels=CONFIG["channels"],
        hidden_dim=CONFIG["hidden_dim"],
        latent_dim=CONFIG["latent_dim"],
    ).to(CONFIG["device"])

    print(f"\nParameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"\nModel architecture:")
    print(model)

    # Train
    model, losses = train_vae(model, dataloader, CONFIG["device"])

    # Visualizations
    if CONFIG["latent_dim"] == 2:
        visualize_latent_space(model, dataloader, CONFIG["device"])

    interpolate_latent(model, CONFIG["device"], num_steps=15)

    # Save model
    output_dir = Path(CONFIG["output_dir"])
    torch.save(model.state_dict(), output_dir / "vae.pth")
    print(f"\nModel saved to: {output_dir}/vae.pth")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    print("\nVAE Key Concepts:")
    print("✓ Probabilistic model for unsupervised learning")
    print("✓ Learns latent distribution p(z)")
    print("✓ Reconstruction loss: measures pixel-level accuracy")
    print("✓ KL divergence: regularizes latent space")
    print("✓ Reparameterization trick: enables backpropagation through sampling")

    print("\nVAE Components:")
    print("- Encoder: maps data x to distribution q(z|x)")
    print("- Latent space: z sampled from q(z|x)")
    print("- Decoder: generates x from z")
    print("- Loss: Reconstruction + KL divergence")

    print("\nApplications:")
    print("- Image generation and synthesis")
    print("- Dimensionality reduction")
    print("- Anomaly detection")
    print("- Semi-supervised learning")
    print("- Data augmentation")

    print("\nHyperparameters:")
    print(f"  Beta: Controls reconstruction vs KL tradeoff")
    print(f"    - Beta=0: Only reconstruction (no regularization)")
    print(f"    - Beta=1: Standard VAE")
    print(f"    - Beta>1: More regularization (smoother latent space)")


if __name__ == "__main__":
    main()
