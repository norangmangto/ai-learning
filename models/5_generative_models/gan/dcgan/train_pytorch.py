import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import numpy as np

# Hyperparameters
latent_dim = 100
batch_size = 64
lr = 0.0002
epochs = 5  # Brief training


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),  # Output -1 to 1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.main(x)


def train():
    print("Training GAN with PyTorch (MNIST)...")
    os.makedirs("gan_images_pytorch", exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )

    dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator()
    discriminator = Discriminator()

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Real labels: 1, Fake labels: 0
            real = torch.ones(imgs.size(0), 1)
            fake = torch.zeros(imgs.size(0), 1)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real
            d_loss_real = criterion(discriminator(imgs), real)

            # Fake
            z = torch.randn(imgs.size(0), latent_dim)
            fake_imgs = generator(z)
            d_loss_fake = criterion(discriminator(fake_imgs.detach()), fake)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_imgs), real)  # Trick discriminator
            g_loss.backward()
            optimizer_G.step()

            if i % 200 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{
        len(dataloader)}] [D loss: {
            d_loss.item():.4f}] [G loss: {
                g_loss.item():.4f}]"
                )

        # Save sample images
        save_image(
            fake_imgs.data[:25],
            f"gan_images_pytorch/{epoch}.png",
            nrow=5,
            normalize=True,
        )

    print("PyTorch GAN Training Complete. Images saved to 'gan_images_pytorch/'.")

    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")

    # Generate final samples for evaluation
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        # Generate samples
        num_samples = 100
        z_test = torch.randn(num_samples, latent_dim)
        generated_samples = generator(z_test)

        # Test discriminator on generated samples
        d_fake_scores = discriminator(generated_samples).numpy().flatten()

        # Test discriminator on real samples
        real_samples = next(iter(dataloader))[0][:num_samples]
        d_real_scores = discriminator(real_samples).numpy().flatten()

    # Calculate metrics
    print(f"\nDiscriminator Performance:")
    print(
        f"Mean score on real images: {
        d_real_scores.mean():.4f} (should be close to 1)"
    )
    print(
        f"Mean score on fake images: {
        d_fake_scores.mean():.4f} (should be close to 0.5 for good GAN)"
    )
    print(f"\nFinal Losses:")
    print(f"Discriminator loss: {d_loss.item():.4f}")
    print(f"Generator loss: {g_loss.item():.4f}")

    # Sanity checks
    print("\n--- Sanity Checks ---")

    # Check 1: Generated images are in valid range [-1, 1]
    gen_min = generated_samples.numpy().min()
    gen_max = generated_samples.numpy().max()
    print(f"\nGenerated image range: [{gen_min:.3f}, {gen_max:.3f}]")
    if gen_min >= -1.5 and gen_max <= 1.5:
        print("✓ Generated images are in reasonable range")
    else:
        print("⚠ WARNING: Generated images have unusual value range")

    # Check 2: Discriminator scores are in valid range [0, 1]
    if np.all((d_fake_scores >= 0) & (d_fake_scores <= 1)) and np.all(
        (d_real_scores >= 0) & (d_real_scores <= 1)
    ):
        print("✓ All discriminator scores are in valid range [0, 1]")
    else:
        print("✗ WARNING: Some discriminator scores are outside [0, 1]!")

    # Check 3: GAN is learning (discriminator can distinguish real from fake)
    if d_real_scores.mean() > 0.6:
        print(
            f"✓ Discriminator recognizes real images well: {
        d_real_scores.mean():.4f}"
        )
    else:
        print(
            f"⚠ Discriminator struggling with real images: {
        d_real_scores.mean():.4f}"
        )

    # Check 4: Generator is fooling discriminator reasonably
    if 0.2 < d_fake_scores.mean() < 0.8:
        print(
            f"✓ Generator creating believable fakes: discriminator score = {
        d_fake_scores.mean():.4f}"
        )
    elif d_fake_scores.mean() < 0.2:
        print(
            f"⚠ Generator struggling: discriminator easily detects fakes ({
        d_fake_scores.mean():.4f})"
        )
    else:
        print(
            f"⚠ Possible mode collapse: discriminator score too high ({
        d_fake_scores.mean():.4f})"
        )

    # Check 5: Loss balance (neither network is dominating)
    loss_ratio = d_loss.item() / (g_loss.item() + 1e-8)
    print(f"\nLoss ratio (D/G): {loss_ratio:.4f}")
    if 0.5 < loss_ratio < 2.0:
        print("✓ Losses are reasonably balanced")
    else:
        print("⚠ Loss imbalance detected - one network may be dominating")

    # Check 6: No NaN or Inf in generated images
    if np.all(np.isfinite(generated_samples.numpy())):
        print("✓ All generated values are finite (no NaN or Inf)")
    else:
        print("✗ WARNING: Some generated values are NaN or Inf!")

    # Check 7: Generated images have reasonable variance
    gen_std = generated_samples.numpy().std()
    if gen_std > 0.1:
        print(f"✓ Generated images have good variance: {gen_std:.4f}")
    else:
        print(
            f"⚠ WARNING: Low variance in generated images: {
        gen_std:.4f} (possible mode collapse)"
        )

    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all(np.isfinite(generated_samples.numpy()))
        and np.all((d_fake_scores >= 0) & (d_fake_scores <= 1))
        and d_real_scores.mean() > 0.5
        and gen_std > 0.1
        and 0.1 < d_fake_scores.mean() < 0.9
    )

    if validation_passed:
        print("✓ Model validation PASSED - GAN is performing as expected")
    else:
        print(
            "✗ Model validation FAILED - Review training dynamics and generated samples"
        )

    print("\nNote: GANs are notoriously difficult to train. Consider:")
    print("  - Training for more epochs")
    print("  - Adjusting learning rates")
    print("  - Using techniques like gradient penalty or spectral normalization")
    print("\nGenerated images saved to: gan_images_pytorch/")


if __name__ == "__main__":
    train()
