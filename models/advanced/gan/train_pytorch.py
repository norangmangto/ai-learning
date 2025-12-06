import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

# Hyperparameters
latent_dim = 100
batch_size = 64
lr = 0.0002
epochs = 5 # Brief training

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
            nn.Linear(1024, 28*28),
            nn.Tanh() # Output -1 to 1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.main(x)

def train():
    print("Training GAN with PyTorch (MNIST)...")
    os.makedirs("gan_images_pytorch", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])

    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
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
            g_loss = criterion(discriminator(fake_imgs), real) # Trick discriminator
            g_loss.backward()
            optimizer_G.step()

            if i % 200 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Save sample images
        save_image(fake_imgs.data[:25], f"gan_images_pytorch/{epoch}.png", nrow=5, normalize=True)

    print("PyTorch GAN Training Complete. Images saved to 'gan_images_pytorch/'.")

if __name__ == "__main__":
    train()
