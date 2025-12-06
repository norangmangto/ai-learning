import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3) # Latent bottleneck
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid() # Start with Sigmoid for 0-1 pixel values
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train():
    print("Training Autoencoder with PyTorch (MNIST)...")
    os.makedirs("autoencoder_images_pytorch", exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
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
        axes[0].imshow(test_img[0].squeeze(), cmap='gray')
        axes[0].set_title('Original')
        axes[1].imshow(recon[0].view(28, 28).numpy(), cmap='gray')
        axes[1].set_title('Reconstructed')
        plt.savefig("autoencoder_images_pytorch/result.png")
        plt.close()

    print("PyTorch Autoencoder Training Complete.")

if __name__ == "__main__":
    train()
