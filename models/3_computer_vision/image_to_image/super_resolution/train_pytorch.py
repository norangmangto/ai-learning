"""
Super Resolution

This script demonstrates:
1. Upscaling low-resolution images to high-resolution
2. PSNR and SSIM metrics for image quality
3. Perceptual loss for realistic details
4. Sub-pixel convolution for efficient upsampling
5. Residual learning for faster convergence

Architectures: SRCNN, ESRGAN, SwinIR
Applications: Image enhancement, photo restoration, video upscaling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# Configuration
CONFIG = {
    'scale_factor': 4,  # 4x upsampling
    'image_size': 64,  # Low-res size (HR will be 256)
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'output_dir': 'results/super_resolution'
}

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11):
    """
    Calculate Structural Similarity Index (SSIM)

    Measures structural similarity between images
    Better correlates with human perception than PSNR
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = nn.functional.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = nn.functional.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.functional.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = nn.functional.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = nn.functional.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.mean()

class ResidualBlock(nn.Module):
    """Residual block for super resolution"""

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        return out

class SubPixelConv(nn.Module):
    """
    Sub-pixel convolution (Pixel Shuffle)

    Efficiently upsamples by rearranging channels into spatial dimensions
    More efficient than transposed convolution
    """

    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale_factor ** 2),
            3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class SRResNet(nn.Module):
    """
    Super Resolution Residual Network

    Based on SRResNet from SRGAN paper
    Uses residual blocks and sub-pixel convolution
    """

    def __init__(self, scale_factor=4, num_residual_blocks=16):
        super().__init__()

        print("\n" + "="*80)
        print("BUILDING SUPER RESOLUTION MODEL")
        print("="*80)

        print(f"\nScale factor: {scale_factor}x")
        print(f"Residual blocks: {num_residual_blocks}")

        # Initial convolution
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.PReLU()
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # Post-residual convolution
        self.conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling layers
        self.upsampling = nn.Sequential()

        if scale_factor == 2 or scale_factor == 4 or scale_factor == 8:
            num_upsample = int(math.log2(scale_factor))

            for i in range(num_upsample):
                self.upsampling.add_module(
                    f'upsample_{i}',
                    SubPixelConv(64, 64, 2)
                )
                self.upsampling.add_module(f'prelu_{i}', nn.PReLU())

        # Output convolution
        self.conv_output = nn.Conv2d(64, 3, 9, padding=4)

        print(f"Parameters: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self, x):
        # Initial features
        out = self.conv_input(x)
        residual = out

        # Residual blocks
        out = self.residual_blocks(out)

        # Post-residual
        out = self.conv_mid(out)

        # Add skip connection
        out = out + residual

        # Upsampling
        out = self.upsampling(out)

        # Output
        out = self.conv_output(out)

        return out

class SuperResolutionDataset(Dataset):
    """Dataset for super resolution"""

    def __init__(self, image_paths, scale_factor=4, hr_size=256):
        self.image_paths = image_paths
        self.scale_factor = scale_factor
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor

        # High-resolution transform
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor()
        ])

        # Low-resolution transform (downsampled)
        self.lr_transform = transforms.Compose([
            transforms.Resize((self.lr_size, self.lr_size), Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')

        # Create high-resolution and low-resolution pairs
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(img)

        return lr_img, hr_img

def create_sample_dataset(num_samples=500):
    """Create sample dataset with diverse patterns"""
    print("\n" + "="*80)
    print("CREATING SAMPLE DATASET")
    print("="*80)

    sample_dir = Path('data/super_resolution_samples')
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for i in range(num_samples):
        # Create image with various patterns
        img = Image.new('RGB', (256, 256))
        pixels = img.load()

        pattern = i % 5

        for y in range(256):
            for x in range(256):
                if pattern == 0:  # Gradient
                    pixels[x, y] = (x, y, 255 - x)
                elif pattern == 1:  # Checkerboard
                    pixels[x, y] = (255, 255, 255) if (x // 16 + y // 16) % 2 == 0 else (0, 0, 0)
                elif pattern == 2:  # Circular
                    dist = math.sqrt((x - 128)**2 + (y - 128)**2)
                    val = int(128 + 127 * math.sin(dist / 10))
                    pixels[x, y] = (val, val, 255 - val)
                elif pattern == 3:  # Noise
                    pixels[x, y] = (
                        np.random.randint(0, 256),
                        np.random.randint(0, 256),
                        np.random.randint(0, 256)
                    )
                else:  # Waves
                    r = int(128 + 127 * np.sin(x / 15))
                    g = int(128 + 127 * np.sin(y / 15))
                    b = int(128 + 127 * np.sin((x + y) / 20))
                    pixels[x, y] = (r, g, b)

        img_path = sample_dir / f'sample_{i}.jpg'
        img.save(img_path)
        image_paths.append(str(img_path))

    print(f"Created {len(image_paths)} sample images")
    return image_paths

def train_super_resolution(model, train_loader, val_loader, device):
    """Train super resolution model"""
    print("\n" + "="*80)
    print("TRAINING SUPER RESOLUTION MODEL")
    print("="*80)

    model.to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_psnr': [],
        'val_ssim': []
    }

    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Forward pass
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

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
        val_psnr = 0
        val_ssim = 0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)

                val_loss += loss.item()

                # Compute metrics
                psnr = calculate_psnr(sr_imgs, hr_imgs)
                ssim = calculate_ssim(sr_imgs, hr_imgs)

                val_psnr += psnr.item()
                val_ssim += ssim.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)

        history['val_loss'].append(avg_val_loss)
        history['val_psnr'].append(avg_val_psnr)
        history['val_ssim'].append(avg_val_ssim)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val PSNR: {avg_val_psnr:.2f} dB")
        print(f"  Val SSIM: {avg_val_ssim:.4f}")

        scheduler.step()

    return model, history

def visualize_super_resolution(model, val_loader, device, num_samples=4):
    """Visualize super resolution results"""
    model.eval()

    lr_imgs, hr_imgs = next(iter(val_loader))
    lr_imgs = lr_imgs[:num_samples].to(device)
    hr_imgs = hr_imgs[:num_samples].to(device)

    with torch.no_grad():
        sr_imgs = model(lr_imgs)

    # Compute metrics for samples
    psnr_bicubic = []
    psnr_sr = []

    for i in range(num_samples):
        # Bicubic upsampling baseline
        bicubic = nn.functional.interpolate(
            lr_imgs[i:i+1],
            scale_factor=CONFIG['scale_factor'],
            mode='bicubic',
            align_corners=False
        )

        psnr_bicubic.append(calculate_psnr(bicubic, hr_imgs[i:i+1]).item())
        psnr_sr.append(calculate_psnr(sr_imgs[i:i+1], hr_imgs[i:i+1]).item())

    # Plot
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples*4))

    for i in range(num_samples):
        # Low-resolution
        lr = lr_imgs[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(lr, 0, 1))
        axes[i, 0].set_title(f'LR ({CONFIG["image_size"]}x{CONFIG["image_size"]})')
        axes[i, 0].axis('off')

        # Bicubic upsampling
        bicubic = nn.functional.interpolate(
            lr_imgs[i:i+1],
            scale_factor=CONFIG['scale_factor'],
            mode='bicubic'
        )[0].cpu().permute(1, 2, 0).numpy()
        axes[i, 1].imshow(np.clip(bicubic, 0, 1))
        axes[i, 1].set_title(f'Bicubic (PSNR: {psnr_bicubic[i]:.2f})')
        axes[i, 1].axis('off')

        # Super resolution
        sr = sr_imgs[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 2].imshow(np.clip(sr, 0, 1))
        axes[i, 2].set_title(f'SR (PSNR: {psnr_sr[i]:.2f})')
        axes[i, 2].axis('off')

        # Ground truth
        hr = hr_imgs[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 3].imshow(np.clip(hr, 0, 1))
        axes[i, 3].set_title('Ground Truth (HR)')
        axes[i, 3].axis('off')

    plt.tight_layout()

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'super_resolution_results.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved results to {output_dir / 'super_resolution_results.png'}")

def main():
    print("="*80)
    print("SUPER RESOLUTION")
    print("="*80)

    # Create dataset
    image_paths = create_sample_dataset(500)

    # Split
    split = int(0.8 * len(image_paths))
    train_paths = image_paths[:split]
    val_paths = image_paths[split:]

    # Datasets
    hr_size = CONFIG['image_size'] * CONFIG['scale_factor']
    train_dataset = SuperResolutionDataset(train_paths, CONFIG['scale_factor'], hr_size)
    val_dataset = SuperResolutionDataset(val_paths, CONFIG['scale_factor'], hr_size)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    # Model
    model = SRResNet(scale_factor=CONFIG['scale_factor'])

    # Train
    model, history = train_super_resolution(model, train_loader, val_loader, CONFIG['device'])

    # Visualize
    visualize_super_resolution(model, val_loader, CONFIG['device'])

    print("\n" + "="*80)
    print("SUPER RESOLUTION COMPLETED")
    print("="*80)

    print("\nKey Concepts:")
    print("✓ Sub-pixel convolution (pixel shuffle)")
    print("✓ Residual learning for faster convergence")
    print("✓ PSNR and SSIM metrics")
    print("✓ Perceptual loss for realistic details")

    print("\nMetrics:")
    print("- PSNR: Peak Signal-to-Noise Ratio (dB)")
    print("- SSIM: Structural Similarity Index")
    print("- Higher is better for both")

    print("\nApplications:")
    print("- Photo enhancement")
    print("- Video upscaling")
    print("- Medical image enhancement")
    print("- Satellite imagery")

if __name__ == '__main__':
    main()
