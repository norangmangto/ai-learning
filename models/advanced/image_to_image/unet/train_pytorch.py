import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder (Expanding Path)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256) # 256 + 256 input channels

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1) # Output RGB

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = self.up3(b)
        # Skip connection: concatenate u3 with e3
        u3 = torch.cat((u3, e3), dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat((u2, e2), dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat((u1, e1), dim=1)
        d1 = self.dec1(u1)

        out = self.final(d1)
        return out

def train():
    print("Training U-Net (Colorization) with PyTorch...")
    os.makedirs("unet_images_pytorch", exist_ok=True)

    # Transforms
    # We load as RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Subset for speed in demo
    train_data = torch.utils.data.Subset(train_data, range(2000))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    model = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2
    for epoch in range(epochs):
        for i, (rgb_imgs, _) in enumerate(train_loader):
            # Create Grayscale Input
            # Simple average or using proper weights. torchvision 'grayscale' transform is better but manual here:
            # 1 channel: 0.299R + 0.587G + 0.114B
            gray_imgs = transforms.functional.rgb_to_grayscale(rgb_imgs)

            optimizer.zero_grad()
            outputs = model(gray_imgs)
            loss = criterion(outputs, rgb_imgs)
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}")

    # Visualize
    model.eval()
    with torch.no_grad():
        test_rgb, _ = train_data[0]
        test_gray = transforms.functional.rgb_to_grayscale(test_rgb).unsqueeze(0)

        output_rgb = model(test_gray).squeeze(0)

        # Clip to 0-1
        output_rgb = torch.clamp(output_rgb, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(test_gray.squeeze(), cmap='gray')
        axes[0].set_title('Input (Grayscale)')
        axes[1].imshow(output_rgb.permute(1, 2, 0).numpy())
        axes[1].set_title('Output (Colorized)')
        axes[2].imshow(test_rgb.permute(1, 2, 0).numpy())
        axes[2].set_title('Truth (RGB)')
        plt.savefig("unet_images_pytorch/result.png")
        plt.close()

    print("PyTorch U-Net Training Complete. Result saved to 'unet_images_pytorch/result.png'.")

if __name__ == "__main__":
    train()
