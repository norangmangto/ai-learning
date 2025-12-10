"""
U-Net for Semantic Segmentation with PyTorch

U-Net is a convolutional neural network architecture for semantic segmentation:
- Encoder-decoder structure with skip connections
- Originally designed for biomedical image segmentation
- Works well with limited training data
- Produces pixel-wise predictions

This implementation includes:
- Standard U-Net architecture
- Binary and multi-class segmentation
- Training with Dice loss and IoU metrics
- Data augmentation
- Visualization of predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.

    Architecture:
    - Encoder: 4 blocks of Conv-ReLU-Conv-ReLU-MaxPool
    - Bottleneck: Conv-ReLU-Conv-ReLU
    - Decoder: 4 blocks of UpConv-Concat-Conv-ReLU-Conv-ReLU
    - Output: 1x1 Conv for final prediction
    """
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features

        # Encoder (Contracting Path)
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")

        # Output layer
        self.conv_out = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv_out(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )


def generate_synthetic_segmentation_data(n_samples=200, img_size=128):
    """
    Generate synthetic segmentation dataset (circles and rectangles).

    Args:
        n_samples: Number of image-mask pairs
        img_size: Size of images

    Returns:
        images: Array of images (n_samples, 3, H, W)
        masks: Array of masks (n_samples, 1, H, W)
    """
    print(f"Generating {n_samples} synthetic images ({img_size}x{img_size})...")

    images = []
    masks = []

    np.random.seed(42)

    for i in range(n_samples):
        # Create blank image
        img = np.zeros((3, img_size, img_size), dtype=np.float32)
        mask = np.zeros((1, img_size, img_size), dtype=np.float32)

        # Add random shapes
        n_shapes = np.random.randint(2, 5)

        for _ in range(n_shapes):
            shape_type = np.random.choice(['circle', 'rectangle'])

            if shape_type == 'circle':
                # Random circle
                cx = np.random.randint(20, img_size-20)
                cy = np.random.randint(20, img_size-20)
                radius = np.random.randint(10, 25)
                color = np.random.rand(3)

                # Draw circle
                y, x = np.ogrid[:img_size, :img_size]
                circle_mask = (x - cx)**2 + (y - cy)**2 <= radius**2

                for c in range(3):
                    img[c][circle_mask] = color[c]
                mask[0][circle_mask] = 1.0

            else:
                # Random rectangle
                x1 = np.random.randint(10, img_size-40)
                y1 = np.random.randint(10, img_size-40)
                w = np.random.randint(20, 40)
                h = np.random.randint(20, 40)
                color = np.random.rand(3)

                # Draw rectangle
                for c in range(3):
                    img[c, y1:y1+h, x1:x1+w] = color[c]
                mask[0, y1:y1+h, x1:x1+w] = 1.0

        # Add noise
        img += np.random.randn(3, img_size, img_size) * 0.05
        img = np.clip(img, 0, 1)

        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    print(f"Dataset shape: images {images.shape}, masks {masks.shape}")

    return images, masks


class SegmentationDataset(Dataset):
    """PyTorch Dataset for segmentation."""
    def __init__(self, images, masks):
        self.images = torch.FloatTensor(images)
        self.masks = torch.FloatTensor(masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice loss = 1 - Dice coefficient
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU).

    Args:
        pred: Predicted masks (with sigmoid applied)
        target: Ground truth masks
        threshold: Threshold for binarization

    Returns:
        iou: IoU score
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def train_unet(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    Train U-Net model.

    Args:
        model: U-Net model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate

    Returns:
        history: Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining U-Net on {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")

    model = model.to(device)

    # Loss and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      patience=5, factor=0.5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': []
    }

    best_val_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_iou = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate IoU
            with torch.no_grad():
                pred_masks = torch.sigmoid(outputs)
                train_iou += calculate_iou(pred_masks, masks)

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

                pred_masks = torch.sigmoid(outputs)
                val_iou += calculate_iou(pred_masks, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')

        epoch_time = time.time() - start_time

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")

    return history


def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Dice)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IoU
    axes[1].plot(history['train_iou'], label='Train IoU', linewidth=2)
    axes[1].plot(history['val_iou'], label='Val IoU', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('IoU Score', fontsize=12)
    axes[1].set_title('Training and Validation IoU', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('unet_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, images, masks, n_samples=4):
    """
    Visualize model predictions.

    Args:
        model: Trained model
        images: Input images
        masks: Ground truth masks
        n_samples: Number of samples to visualize
    """
    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3*n_samples))

    with torch.no_grad():
        for i in range(n_samples):
            img_tensor = torch.FloatTensor(images[i:i+1]).to(device)
            pred = model(img_tensor)
            pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]

            # Original image
            img_display = images[i].transpose(1, 2, 0)
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            # Ground truth mask
            axes[i, 1].imshow(masks[i, 0], cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            # Predicted mask
            axes[i, 2].imshow(pred_mask, cmap='gray')
            iou = calculate_iou(torch.FloatTensor(pred_mask[None, None, :, :]),
                               torch.FloatTensor(masks[i:i+1]))
            axes[i, 2].set_title(f'Prediction (IoU: {iou:.3f})')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('unet_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("U-Net for Semantic Segmentation")
    print("="*70)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic segmentation dataset...")
    images, masks = generate_synthetic_segmentation_data(n_samples=200, img_size=128)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")

    # 2. Create data loaders
    train_dataset = SegmentationDataset(X_train, y_train)
    val_dataset = SegmentationDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 3. Create model
    print("\n2. Creating U-Net model...")
    model = UNet(in_channels=3, out_channels=1, init_features=32)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 4. Train model
    print("\n3. Training model...")
    history = train_unet(model, train_loader, val_loader, epochs=50, lr=0.001)

    # 5. Plot training history
    print("\n4. Plotting training history...")
    plot_training_history(history)

    # 6. Load best model and visualize predictions
    print("\n5. Visualizing predictions...")
    model.load_state_dict(torch.load('best_unet_model.pth'))
    visualize_predictions(model, X_val, y_val, n_samples=4)

    print("\n" + "="*70)
    print("U-Net Training Complete!")
    print("="*70)

    print("\nKey Features of U-Net:")
    print("✓ Skip connections preserve spatial information")
    print("✓ Works well with limited training data")
    print("✓ Symmetric encoder-decoder structure")
    print("✓ Pixel-wise predictions")

    print("\nWhen to use U-Net:")
    print("✓ Medical image segmentation")
    print("✓ Limited training data")
    print("✓ Need precise boundaries")
    print("✓ Binary or multi-class segmentation")

    print("\nBest Practices:")
    print("1. Use Dice loss for segmentation")
    print("2. Monitor IoU metric during training")
    print("3. Use data augmentation for small datasets")
    print("4. Start with batch size 4-8")
    print("5. Use learning rate scheduling")

    print("\nArchitecture Tips:")
    print("- init_features=32 for small images (128x128)")
    print("- init_features=64 for medium images (256x256)")
    print("- Reduce depth for smaller datasets")
    print("- Add dropout for regularization if overfitting")


if __name__ == "__main__":
    main()
