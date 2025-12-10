"""
Image Inpainting

This script demonstrates:
1. Filling missing/masked regions in images
2. Partial convolutions for irregular masks
3. Context-aware image reconstruction
4. Training with random masks
5. Perceptual loss for realistic inpainting

Architecture: U-Net with partial convolutions or GAN-based
Applications: Photo restoration, object removal, image editing
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
import random

# Configuration
CONFIG = {
    'image_size': 256,
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.0002,
    'mask_ratio': 0.3,  # Percentage of image to mask
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'output_dir': 'results/inpainting'
}

class PartialConv2d(nn.Module):
    """
    Partial Convolution layer

    Handles irregular masks by updating mask based on valid pixels
    Reference: "Image Inpainting for Irregular Holes Using Partial Convolutions"
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=False
        )

        # Mask update conv (sum of 1s in kernel)
        self.mask_conv = nn.Conv2d(
            1, 1,
            kernel_size, stride, padding, bias=False
        )

        # Initialize mask conv weights to 1
        nn.init.constant_(self.mask_conv.weight, 1.0)

        # Make mask conv non-trainable
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        """
        Args:
            x: Input tensor [B, C, H, W]
            mask: Binary mask [B, 1, H, W] (1=valid, 0=masked)

        Returns:
            output: Inpainted tensor
            updated_mask: Updated mask
        """
        # Apply mask to input
        masked_input = x * mask

        # Compute convolution
        output = self.conv(masked_input)

        # Update mask
        with torch.no_grad():
            # Sum of valid pixels in each kernel window
            mask_sum = self.mask_conv(mask)

            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-8)

            # Normalize by number of valid pixels
            # If all pixels in kernel are valid, no scaling needed
            kernel_size = self.mask_conv.weight.shape[2]
            scaling = kernel_size * kernel_size / mask_sum

            # Update mask (1 if any valid pixel in kernel)
            updated_mask = (mask_sum > 0).float()

        # Scale output
        output = output * scaling

        return output, updated_mask

class InpaintingUNet(nn.Module):
    """U-Net with partial convolutions for inpainting"""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        print("\n" + "="*80)
        print("BUILDING INPAINTING MODEL")
        print("="*80)

        # Encoder
        self.enc1 = PartialConv2d(in_channels, 64, 7, 2, 3)
        self.enc2 = PartialConv2d(64, 128, 5, 2, 2)
        self.enc3 = PartialConv2d(128, 256, 5, 2, 2)
        self.enc4 = PartialConv2d(256, 512, 3, 2, 1)
        self.enc5 = PartialConv2d(512, 512, 3, 2, 1)

        # Decoder
        self.dec5 = PartialConv2d(512, 512, 3, 1, 1)
        self.dec4 = PartialConv2d(512 + 512, 256, 3, 1, 1)
        self.dec3 = PartialConv2d(256 + 256, 128, 3, 1, 1)
        self.dec2 = PartialConv2d(128 + 128, 64, 3, 1, 1)
        self.dec1 = nn.Conv2d(64 + 64, out_channels, 3, 1, 1)

        self.bn_enc = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(512)
        ])

        self.bn_dec = nn.ModuleList([
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64)
        ])

        self.activation = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        print(f"Parameters: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self, x, mask):
        # Encoder
        x1, m1 = self.enc1(x, mask)
        x1 = self.bn_enc[0](x1)
        x1 = self.activation(x1)

        x2, m2 = self.enc2(x1, m1)
        x2 = self.bn_enc[1](x2)
        x2 = self.activation(x2)

        x3, m3 = self.enc3(x2, m2)
        x3 = self.bn_enc[2](x3)
        x3 = self.activation(x3)

        x4, m4 = self.enc4(x3, m3)
        x4 = self.bn_enc[3](x4)
        x4 = self.activation(x4)

        x5, m5 = self.enc5(x4, m4)
        x5 = self.bn_enc[4](x5)
        x5 = self.activation(x5)

        # Decoder with skip connections
        x = self.upsample(x5)
        m = self.upsample(m5)
        x = torch.cat([x, x4], dim=1)
        x, m = self.dec5(x, m)
        x = self.bn_dec[0](x)
        x = self.activation(x)

        x = self.upsample(x)
        m = self.upsample(m)
        x = torch.cat([x, x3], dim=1)
        x, m = self.dec4(x, m)
        x = self.bn_dec[1](x)
        x = self.activation(x)

        x = self.upsample(x)
        m = self.upsample(m)
        x = torch.cat([x, x2], dim=1)
        x, m = self.dec3(x, m)
        x = self.bn_dec[2](x)
        x = self.activation(x)

        x = self.upsample(x)
        m = self.upsample(m)
        x = torch.cat([x, x1], dim=1)
        x, m = self.dec2(x, m)
        x = self.bn_dec[3](x)
        x = self.activation(x)

        x = self.upsample(x)
        x = torch.cat([x, x * mask], dim=1)
        x = self.dec1(x)
        x = torch.tanh(x)

        return x

class InpaintingDataset(Dataset):
    """Dataset for image inpainting"""

    def __init__(self, image_paths, transform=None, mask_ratio=0.3):
        self.image_paths = image_paths
        self.transform = transform
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.image_paths)

    def create_random_mask(self, size):
        """Create random irregular mask"""
        mask = np.ones((size, size), dtype=np.float32)

        # Random number of rectangular holes
        num_holes = random.randint(1, 5)

        for _ in range(num_holes):
            # Random hole size
            hole_h = random.randint(size // 8, size // 3)
            hole_w = random.randint(size // 8, size // 3)

            # Random position
            y = random.randint(0, size - hole_h)
            x = random.randint(0, size - hole_w)

            mask[y:y+hole_h, x:x+hole_w] = 0

        # Add some random strokes
        num_strokes = random.randint(5, 15)
        for _ in range(num_strokes):
            x1 = random.randint(0, size-1)
            y1 = random.randint(0, size-1)
            x2 = random.randint(0, size-1)
            y2 = random.randint(0, size-1)

            thickness = random.randint(2, 8)

            # Draw line (simplified)
            steps = max(abs(x2-x1), abs(y2-y1))
            if steps > 0:
                for i in range(steps):
                    x = int(x1 + (x2-x1) * i / steps)
                    y = int(y1 + (y2-y1) * i / steps)

                    y_min = max(0, y - thickness//2)
                    y_max = min(size, y + thickness//2)
                    x_min = max(0, x - thickness//2)
                    x_max = min(size, x + thickness//2)

                    mask[y_min:y_max, x_min:x_max] = 0

        return mask

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Create mask
        size = image.shape[1]
        mask = self.create_random_mask(size)
        mask = torch.FloatTensor(mask).unsqueeze(0)

        # Masked image
        masked_image = image * mask

        return masked_image, mask, image

def create_sample_dataset(num_samples=500):
    """Create sample dataset"""
    print("\n" + "="*80)
    print("CREATING SAMPLE DATASET")
    print("="*80)

    sample_dir = Path('data/inpainting_samples')
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for i in range(num_samples):
        # Create colorful image with patterns
        img = Image.new('RGB', (256, 256))
        pixels = img.load()

        for y in range(256):
            for x in range(256):
                r = int(128 + 127 * np.sin(x / 10 + i))
                g = int(128 + 127 * np.sin(y / 10 + i))
                b = int(128 + 127 * np.sin((x + y) / 10 + i))
                pixels[x, y] = (r, g, b)

        img_path = sample_dir / f'sample_{i}.jpg'
        img.save(img_path)
        image_paths.append(str(img_path))

    print(f"Created {len(image_paths)} sample images")
    return image_paths

def train_inpainting_model(model, train_loader, val_loader, device):
    """Train inpainting model"""
    print("\n" + "="*80)
    print("TRAINING INPAINTING MODEL")
    print("="*80)

    model.to(device)

    # Loss functions
    l1_loss = nn.L1Loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for masked_img, mask, target in progress_bar:
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # Forward pass
            output = model(masked_img, mask)

            # Loss only on masked regions
            loss_hole = l1_loss(output * (1 - mask), target * (1 - mask))
            loss_valid = l1_loss(output * mask, target * mask)

            # Total loss (weight hole region more)
            loss = 6 * loss_hole + loss_valid

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
            for masked_img, mask, target in val_loader:
                masked_img = masked_img.to(device)
                mask = mask.to(device)
                target = target.to(device)

                output = model(masked_img, mask)

                loss_hole = l1_loss(output * (1 - mask), target * (1 - mask))
                loss_valid = l1_loss(output * mask, target * mask)
                loss = 6 * loss_hole + loss_valid

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        scheduler.step()

    return model, history

def visualize_inpainting(model, val_loader, device, num_samples=4):
    """Visualize inpainting results"""
    model.eval()

    # Get samples
    masked_imgs, masks, targets = next(iter(val_loader))
    masked_imgs = masked_imgs[:num_samples].to(device)
    masks = masks[:num_samples].to(device)
    targets = targets[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(masked_imgs, masks)

    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples*4))

    for i in range(num_samples):
        # Masked image
        masked = masked_imgs[i].cpu().permute(1, 2, 0).numpy()
        masked = (masked + 1) / 2  # Denormalize
        axes[i, 0].imshow(np.clip(masked, 0, 1))
        axes[i, 0].set_title('Masked Input')
        axes[i, 0].axis('off')

        # Inpainted
        inpainted = outputs[i].cpu().permute(1, 2, 0).numpy()
        inpainted = (inpainted + 1) / 2
        axes[i, 1].imshow(np.clip(inpainted, 0, 1))
        axes[i, 1].set_title('Inpainted')
        axes[i, 1].axis('off')

        # Ground truth
        target = targets[i].cpu().permute(1, 2, 0).numpy()
        target = (target + 1) / 2
        axes[i, 2].imshow(np.clip(target, 0, 1))
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')

    plt.tight_layout()

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'inpainting_results.png', dpi=150, bbox_inches='tight')
    print(f"Saved results to {output_dir / 'inpainting_results.png'}")

def main():
    print("="*80)
    print("IMAGE INPAINTING")
    print("="*80)

    # Create dataset
    image_paths = create_sample_dataset(500)

    # Split
    split = int(0.8 * len(image_paths))
    train_paths = image_paths[:split]
    val_paths = image_paths[split:]

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Datasets
    train_dataset = InpaintingDataset(train_paths, transform, CONFIG['mask_ratio'])
    val_dataset = InpaintingDataset(val_paths, transform, CONFIG['mask_ratio'])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    # Model
    model = InpaintingUNet()

    # Train
    model, history = train_inpainting_model(model, train_loader, val_loader, CONFIG['device'])

    # Visualize
    visualize_inpainting(model, val_loader, CONFIG['device'])

    print("\n" + "="*80)
    print("INPAINTING COMPLETED")
    print("="*80)

    print("\nKey Concepts:")
    print("✓ Partial convolutions handle irregular masks")
    print("✓ Higher loss weight on hole regions")
    print("✓ Skip connections preserve details")
    print("✓ Context-aware reconstruction")

    print("\nApplications:")
    print("- Object removal from photos")
    print("- Photo restoration")
    print("- Image editing tools")
    print("- Corrupted image recovery")

if __name__ == '__main__':
    main()
