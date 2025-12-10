"""
U-Net for Image-to-Image Translation

This script demonstrates:
1. General image-to-image translation with U-Net
2. Paired image translation (e.g., edges → photos)
3. Encoder-decoder with skip connections
4. Various tasks: colorization, denoising, enhancement
5. Conditional generation

Architecture: U-Net (encoder-decoder with skip connections)
Applications: Colorization, denoising, enhancement, edge-to-photo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    'task': 'colorization',  # 'colorization', 'denoising', 'enhancement'
    'image_size': 256,
    'batch_size': 16,
    'epochs': 50,
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
            # ...rest of code...
        self.down7 = UNetBlock(512, 512, down=True)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        # Decoder (upsampling with skip connections)
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up4 = UNetBlock(1024, 512, down=False)
        self.up5 = UNetBlock(1024, 256, down=False)
        self.up6 = UNetBlock(512, 128, down=False)
        self.up7 = UNetBlock(256, 64, down=False)

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

        print(f"Parameters: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self, x):
        # Encoder with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # Bottleneck
        bottleneck = self.bottleneck(d7)

        # Decoder with skip connections (concatenate)
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d7], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)

        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)

        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        # Output
        output = self.final(u7)

        return output

class ImageToImageDataset(Dataset):
    """Dataset for image-to-image translation"""

    def __init__(self, image_paths, task='colorization', transform=None):
        self.image_paths = image_paths
        self.task = task
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def create_input_target_pair(self, img):
        """Create input and target based on task"""
        if self.task == 'colorization':
            # Input: grayscale, Target: color
            target = self.transform(img)
            input_img = transforms.Grayscale()(img)
            input_img = transforms.ToTensor()(input_img)
            input_img = transforms.Normalize([0.5], [0.5])(input_img)

        elif self.task == 'denoising':
            # Input: noisy, Target: clean
            target = self.transform(img)

            # Add noise to input
            input_img = target.clone()
            noise = torch.randn_like(input_img) * 0.1
            input_img = input_img + noise
            input_img = torch.clamp(input_img, -1, 1)

        elif self.task == 'enhancement':
            # Input: degraded (low contrast), Target: enhanced
            target = self.transform(img)

            # Reduce contrast for input
            input_img = target * 0.5  # Reduce dynamic range

        else:
            raise ValueError(f"Unknown task: {self.task}")

        return input_img, target

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        input_img, target = self.create_input_target_pair(img)
        return input_img, target

def create_sample_dataset(num_samples=500):
    """Create sample colorful images"""
    print("\n" + "="*80)
    print("CREATING SAMPLE DATASET")
    print("="*80)

    sample_dir = Path('data/image_to_image_samples')
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for i in range(num_samples):
        # Create colorful image
        img = Image.new('RGB', (256, 256))
        pixels = img.load()

        # Different patterns
        pattern = i % 4

        for y in range(256):
            for x in range(256):
                if pattern == 0:  # Gradient
                    r = int(255 * x / 256)
                    g = int(255 * y / 256)
                    b = int(255 * (1 - x / 256))
                    pixels[x, y] = (r, g, b)

                elif pattern == 1:  # Circles
                    dist = np.sqrt((x - 128)**2 + (y - 128)**2)
                    r = int(128 + 127 * np.sin(dist / 10))
                    g = int(128 + 127 * np.cos(dist / 10))
                    b = 255 - r
                    pixels[x, y] = (r, g, b)

                elif pattern == 2:  # Waves
                    r = int(128 + 127 * np.sin(x / 20))
                    g = int(128 + 127 * np.sin(y / 20))
                    b = int(128 + 127 * np.sin((x + y) / 30))
                    pixels[x, y] = (r, g, b)

                else:  # Checkerboard with colors
                    if (x // 32 + y // 32) % 2 == 0:
                        pixels[x, y] = (255, 100, 100)
                    else:
                        pixels[x, y] = (100, 100, 255)

        img_path = sample_dir / f'sample_{i}.jpg'
        img.save(img_path)
        image_paths.append(str(img_path))

    print(f"Created {len(image_paths)} sample images")
    return image_paths

def train_unet(model, train_loader, val_loader, device):
    """Train U-Net for image-to-image translation"""
    print("\n" + "="*80)
    print(f"TRAINING U-NET FOR {CONFIG['task'].upper()}")
    print("="*80)

    model.to(device)

    # Loss functions
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(CONFIG['beta1'], 0.999)
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for input_imgs, target_imgs in progress_bar:
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            # Forward pass
            output_imgs = model(input_imgs)

            # Combined loss (L1 + L2)
            loss = l1_loss(output_imgs, target_imgs) + 0.1 * l2_loss(output_imgs, target_imgs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for input_imgs, target_imgs in val_loader:
                input_imgs = input_imgs.to(device)
                target_imgs = target_imgs.to(device)

                output_imgs = model(input_imgs)
                loss = l1_loss(output_imgs, target_imgs) + 0.1 * l2_loss(output_imgs, target_imgs)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        scheduler.step()

    return model, history

def visualize_translation(model, val_loader, device, task='colorization', num_samples=4):
    """Visualize image-to-image translation results"""
    model.eval()

    input_imgs, target_imgs = next(iter(val_loader))
    input_imgs = input_imgs[:num_samples].to(device)
    target_imgs = target_imgs[:num_samples].to(device)

    with torch.no_grad():
        output_imgs = model(input_imgs)

    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples*4))

    for i in range(num_samples):
        # Input
        if task == 'colorization':
            input_img = input_imgs[i].cpu().squeeze().numpy()
            input_img = (input_img + 1) / 2
            axes[i, 0].imshow(input_img, cmap='gray')
        else:
            input_img = input_imgs[i].cpu().permute(1, 2, 0).numpy()
            input_img = (input_img + 1) / 2
            axes[i, 0].imshow(np.clip(input_img, 0, 1))

        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        # Output
        output_img = output_imgs[i].cpu().permute(1, 2, 0).numpy()
        output_img = (output_img + 1) / 2
        axes[i, 1].imshow(np.clip(output_img, 0, 1))
        axes[i, 1].set_title('Output')
        axes[i, 1].axis('off')

        # Target
        target_img = target_imgs[i].cpu().permute(1, 2, 0).numpy()
        target_img = (target_img + 1) / 2
        axes[i, 2].imshow(np.clip(target_img, 0, 1))
        axes[i, 2].set_title('Target')
        axes[i, 2].axis('off')

    plt.tight_layout()

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{task}_results.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved results to {output_dir / f'{task}_results.png'}")

def main():
    print("="*80)
    print("IMAGE-TO-IMAGE TRANSLATION WITH U-NET")
    print("="*80)

    print(f"\nTask: {CONFIG['task']}")
    print(f"Device: {CONFIG['device']}")

    # Create dataset
    image_paths = create_sample_dataset(500)

    # Split
    split = int(0.8 * len(image_paths))
    train_paths = image_paths[:split]
    val_paths = image_paths[split:]

    # Transform
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Datasets
    train_dataset = ImageToImageDataset(train_paths, CONFIG['task'], transform)
    val_dataset = ImageToImageDataset(val_paths, CONFIG['task'], transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    # Model
    if CONFIG['task'] == 'colorization':
        model = UNetGenerator(in_channels=1, out_channels=3)
    else:
        model = UNetGenerator(in_channels=3, out_channels=3)

    # Train
    model, history = train_unet(model, train_loader, val_loader, CONFIG['device'])

    # Visualize
    visualize_translation(model, val_loader, CONFIG['device'], CONFIG['task'])

    print("\n" + "="*80)
    print("IMAGE-TO-IMAGE TRANSLATION COMPLETED")
    print("="*80)

    print("\nKey Concepts:")
    print("✓ Encoder-decoder architecture")
    print("✓ Skip connections preserve details")
    print("✓ Paired training (input-target pairs)")
    print("✓ L1 loss for pixel-wise accuracy")

    print("\nU-Net Advantages:")
    print("- Skip connections preserve spatial info")
    print("- Works well with limited data")
    print("- Good for dense prediction tasks")
    print("- Fast inference")

    print("\nApplications:")
    print("- Image colorization")
    print("- Image denoising")
    print("- Image enhancement")
    print("- Edge-to-photo")
    print("- Segmentation masks")
    print("- Medical image translation")

if __name__ == '__main__':
    main()
