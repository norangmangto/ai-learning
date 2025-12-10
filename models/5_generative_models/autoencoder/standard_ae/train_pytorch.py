import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),  # Latent bottleneck
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # Start with Sigmoid for 0-1 pixel values
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train():
    print("Training Autoencoder with PyTorch (MNIST)...")
    os.makedirs("autoencoder_images_pytorch", exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3
    for epoch in range(epochs):
        for data in loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Saving reconstruction example...")
    # Visualize verification
    with torch.no_grad():
        test_img, _ = next(iter(loader))
        test_flat = test_img.view(test_img.size(0), -1)
        recon = model(test_flat)

        # Save first image original vs recon
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(test_img[0].squeeze(), cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(recon[0].view(28, 28).numpy(), cmap="gray")
        axes[1].set_title("Reconstructed")
        plt.savefig("autoencoder_images_pytorch/result.png")
        plt.close()

    print("PyTorch Autoencoder Training Complete.")

    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")

    # Evaluate reconstruction quality on test batch
    model.eval()
    with torch.no_grad():
        # Get a larger test batch
        test_images = []
        for batch_idx, (img, _) in enumerate(loader):
            if batch_idx >= 5:  # Use 5 batches for evaluation
                break
            test_images.append(img)

        test_images = torch.cat(test_images, dim=0)
        test_flat = test_images.view(test_images.size(0), -1)
        reconstructed = model(test_flat)

        # Calculate reconstruction error
        criterion(reconstructed, test_flat).item()

    # Convert to numpy for metrics
    original_np = test_flat.numpy()
    reconstructed_np = reconstructed.numpy()

    # Calculate comprehensive metrics
    mse = mean_squared_error(original_np, reconstructed_np)
    mae = mean_absolute_error(original_np, reconstructed_np)
    rmse = np.sqrt(mse)

    print(f"\nReconstruction Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Final training loss: {loss.item():.6f}")

    # Sanity checks
    print("\n--- Sanity Checks ---")

    # Check 1: Reconstructions are in valid range [0, 1]
    if np.all((reconstructed_np >= 0) & (reconstructed_np <= 1)):
        print("✓ All reconstructed pixel values are in valid range [0, 1]")
    else:
        out_of_range = np.sum((reconstructed_np < 0) | (reconstructed_np > 1))
        print(f"⚠ WARNING: {out_of_range} pixel values are outside [0, 1] range")

    # Check 2: Reconstruction quality
    if mse < 0.01:
        print(f"✓ Excellent reconstruction quality: MSE = {mse:.6f}")
    elif mse < 0.05:
        print(f"✓ Good reconstruction quality: MSE = {mse:.6f}")
    elif mse < 0.1:
        print(f"⚠ Moderate reconstruction quality: MSE = {mse:.6f}")
    else:
        print(f"✗ WARNING: Poor reconstruction quality: MSE = {mse:.6f}")

    # Check 3: Check if model is learning (loss decreased)
    if loss.item() < 0.1:
        print(f"✓ Model converged well - Final loss: {loss.item():.6f}")
    else:
        print(f"⚠ Model loss: {loss.item():.6f} (may need more training)")

    # Check 4: No NaN or Inf in reconstructions
    if np.all(np.isfinite(reconstructed_np)):
        print("✓ All reconstructed values are finite (no NaN or Inf)")
    else:
        print("✗ WARNING: Some reconstructed values are NaN or Inf!")

    # Check 5: Latent space dimensionality
    with torch.no_grad():
        sample = test_flat[:10]
        latent = model.encoder(sample)
        print(
            f"\n✓ Latent space dimension: {
        latent.shape[1]} (compressed from {
            test_flat.shape[1]})"
        )
        print(
            f"  Compression ratio: {
        test_flat.shape[1] /
         latent.shape[1]:.1f}x"
        )

    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all(np.isfinite(reconstructed_np))
        and mse < 0.2
        and np.sum((reconstructed_np < -0.1) | (reconstructed_np > 1.1)) == 0
    )

    if validation_passed:
        print("✓ Model validation PASSED - Autoencoder is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")

    print(
        "\nReconstruction visualization saved to: autoencoder_images_pytorch/result.png"
    )


if __name__ == "__main__":
    train()
