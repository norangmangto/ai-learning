"""
U-Net for Semantic Segmentation

This script demonstrates:
1. U-Net architecture for semantic segmentation
2. Encoder-decoder with skip connections
3. Training on Pascal VOC or Cityscapes dataset
4. Pixel-level classification
5. Evaluation with mIoU metric

Dataset: Pascal VOC 2012
Model: U-Net with various backbones
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Configuration
CONFIG = {
    'num_classes': 21,  # Pascal VOC classes
    'image_size': 512,
    'batch_size': 4,
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'output_dir': 'results/unet'
}

class DoubleConv(nn.Module):
    """Double Convolution Block"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downsampling (Encoder) Block"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling (Decoder) Block with Skip Connection"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1, x2):
        """
        x1: Upsampled feature map
        x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch
        if x2.shape[-1] != x1.shape[-1]:
            diff_h = x2.shape[-2] - x1.shape[-2]
            diff_w = x2.shape[-1] - x1.shape[-1]
            x1 = F.pad(x1, (diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2))

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """U-Net Architecture for Semantic Segmentation"""
    def __init__(self, in_channels=3, num_classes=21, bilinear=False):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits

def load_data():
    """Load Pascal VOC dataset"""
    print("="*80)
    print("LOADING DATASET")
    print("="*80)

    transform_image = T.Compose([
        T.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])

    transform_target = T.Compose([
        T.Resize((CONFIG['image_size'], CONFIG['image_size']), T.InterpolationMode.NEAREST),
        T.ToTensor()
    ])

    try:
        # Try to load Pascal VOC
        dataset = VOCSegmentation(
            root='data',
            year='2012',
            image_set='train',
            download=True,
            transforms=(transform_image, transform_target)
        )

        print(f"Loaded Pascal VOC 2012")
        print(f"Total images: {len(dataset)}")

        dataloader = DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=CONFIG['num_workers']
        )

        return dataloader

    except Exception as e:
        print(f"Could not load Pascal VOC: {e}")
        print("Using dummy dataset for demonstration...")

        # Create dummy dataset
        from torch.utils.data import TensorDataset

        x = torch.randn(100, 3, CONFIG['image_size'], CONFIG['image_size'])
        y = torch.randint(0, CONFIG['num_classes'], (100, CONFIG['image_size'], CONFIG['image_size']))

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True
        )

        return dataloader

def compute_iou(predictions, targets, num_classes):
    """Compute Intersection over Union (IoU)"""
    ious = []

    for class_id in range(num_classes):
        intersection = ((predictions == class_id) & (targets == class_id)).sum().item()
        union = ((predictions == class_id) | (targets == class_id)).sum().item()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        ious.append(iou)

    return np.array(ious)

def train_unet(model, dataloader, device):
    """Train U-Net model"""
    print("\n" + "="*80)
    print("TRAINING U-NET")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Image size: {CONFIG['image_size']}")
    print(f"  Device: {device}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    losses = []
    ious = []

    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_start = time.time()

        total_loss = 0
        total_iou = []

        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device).long()

            # Squeeze channel dimension from target if needed
            if y.ndim == 4 and y.shape[1] == 1:
                y = y.squeeze(1)

            # Forward pass
            logits = model(x)

            # Compute loss
            loss = criterion(logits, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Compute IoU
            predictions = logits.argmax(dim=1)
            batch_iou = compute_iou(predictions, y, CONFIG['num_classes'])
            total_iou.append(batch_iou.mean())

        # Update learning rate
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Average metrics
        avg_loss = total_loss / len(dataloader)
        avg_iou = np.mean(total_iou)

        losses.append(avg_loss)
        ious.append(avg_iou)

        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Loss: {avg_loss:.4f} "
              f"mIoU: {avg_iou:.4f} "
              f"Time: {epoch_time:.2f}s")

    # Plot training curves
    plot_metrics(losses, ious, output_dir)

    return model

def plot_metrics(losses, ious, output_dir):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(ious)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('Mean Intersection over Union')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics.png', dpi=150)
    print(f"\nMetrics plot saved to: {output_dir}/metrics.png")
    plt.close()

def segment_image(model, image, device):
    """Segment an image"""
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])

    if isinstance(image, str):
        from PIL import Image
        image = Image.open(image).convert('RGB')

    # Resize
    image = image.resize((CONFIG['image_size'], CONFIG['image_size']))
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(image_tensor)

    # Get predictions
    predictions = logits.argmax(dim=1)[0].cpu().numpy()

    return predictions

def main():
    print("="*80)
    print("U-NET FOR SEMANTIC SEGMENTATION")
    print("="*80)

    print(f"\nDevice: {CONFIG['device']}")

    # Load data
    dataloader = load_data()

    # Create model
    print("\n" + "="*80)
    print("CREATING U-NET MODEL")
    print("="*80)

    model = UNet(
        in_channels=3,
        num_classes=CONFIG['num_classes'],
        bilinear=True
    ).to(CONFIG['device'])

    print(f"\nParameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Train
    model = train_unet(model, dataloader, CONFIG['device'])

    # Save model
    output_dir = Path(CONFIG['output_dir'])
    torch.save(model.state_dict(), output_dir / 'unet.pth')
    print(f"\nModel saved to: {output_dir}/unet.pth")

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    print("\nU-Net Architecture:")
    print("✓ Encoder-Decoder structure")
    print("✓ Skip connections to preserve spatial information")
    print("✓ Upsampling path mirrors downsampling")
    print("✓ Output has same spatial resolution as input")

    print("\nKey Features:")
    print("- Encoder: Extracts features at multiple scales")
    print("- Decoder: Upsamples and refines predictions")
    print("- Skip connections: Concatenate encoder features with decoder")
    print("- Each level: Double convolution blocks")

    print("\nApplications:")
    print("- Semantic segmentation (pixel-level classification)")
    print("- Medical image segmentation")
    print("- Instance segmentation")
    print("- Depth estimation")
    print("- Object boundary detection")

    print("\nMetrics:")
    print("- Intersection over Union (IoU): per-class and mean")
    print("- Pixel Accuracy: percentage of correctly classified pixels")
    print("- Frequency Weighted IoU: weighted by class frequency")

if __name__ == '__main__':
    main()
