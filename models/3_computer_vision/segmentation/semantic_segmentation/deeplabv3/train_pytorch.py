"""
DeepLabV3 for Semantic Segmentation with PyTorch

DeepLabV3 uses atrous (dilated) convolutions for semantic segmentation:
- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale features
- Encoder with dilated ResNet backbone
- No need for decoder (unlike U-Net)
- State-of-the-art performance on many benchmarks

This implementation includes:
- DeepLabV3 with ResNet50 backbone
- ASPP module for multi-scale context
- Transfer learning from ImageNet
- Training and evaluation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn.functional as F
import time


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.

    Captures multi-scale information using parallel atrous convolutions
    with different dilation rates.
    """

    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()

        modules = []

        # 1x1 convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Atrous convolutions with different rates
        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global average pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)

        # Projection layer
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Upsample global pooling result to match feature size
        res[-1] = F.interpolate(
            res[-1], size=x.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate all branches
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3(nn.Module):
    """
    DeepLabV3 architecture for semantic segmentation.

    Uses ResNet as backbone with atrous convolutions and ASPP module.
    """

    def __init__(self, num_classes=1, backbone="resnet50", pretrained=True):
        super(DeepLabV3, self).__init__()

        # Load pretrained ResNet backbone
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove fully connected layers
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Modify layer4 to use atrous convolutions
        for module in self.layer4.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (2, 2)
                    module.padding = (2, 2)

        # ASPP module
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Encoder
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Classifier
        x = self.classifier(x)

        # Upsample to input size
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x


def generate_synthetic_data(n_samples=200, img_size=256):
    """Generate synthetic segmentation dataset."""
    print(f"Generating {n_samples} synthetic images ({img_size}x{img_size})...")

    images = []
    masks = []

    np.random.seed(42)

    for i in range(n_samples):
        # Create blank image
        img = np.zeros((3, img_size, img_size), dtype=np.float32)
        mask = np.zeros((1, img_size, img_size), dtype=np.float32)

        # Add random shapes
        n_shapes = np.random.randint(3, 6)

        for _ in range(n_shapes):
            shape_type = np.random.choice(["circle", "rectangle", "ellipse"])

            if shape_type == "circle":
                cx = np.random.randint(30, img_size - 30)
                cy = np.random.randint(30, img_size - 30)
                radius = np.random.randint(15, 40)
                color = np.random.rand(3)

                y, x = np.ogrid[:img_size, :img_size]
                circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

                for c in range(3):
                    img[c][circle_mask] = color[c]
                mask[0][circle_mask] = 1.0

            elif shape_type == "rectangle":
                x1 = np.random.randint(20, img_size - 60)
                y1 = np.random.randint(20, img_size - 60)
                w = np.random.randint(30, 60)
                h = np.random.randint(30, 60)
                color = np.random.rand(3)

                for c in range(3):
                    img[c, y1 : y1 + h, x1 : x1 + w] = color[c]
                mask[0, y1 : y1 + h, x1 : x1 + w] = 1.0

            else:  # ellipse
                cx = np.random.randint(40, img_size - 40)
                cy = np.random.randint(40, img_size - 40)
                a = np.random.randint(20, 50)
                b = np.random.randint(15, 40)
                color = np.random.rand(3)

                y, x = np.ogrid[:img_size, :img_size]
                ellipse_mask = ((x - cx) ** 2 / a**2 + (y - cy) ** 2 / b**2) <= 1

                for c in range(3):
                    img[c][ellipse_mask] = color[c]
                mask[0][ellipse_mask] = 1.0

        # Add noise
        img += np.random.randn(3, img_size, img_size) * 0.05
        img = np.clip(img, 0, 1)

        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

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


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss."""

    def __init__(self, dice_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # BCE loss
        bce_loss = self.bce(pred, target)

        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)
        dice_loss = 1 - dice

        # Combined loss
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss


def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU metric."""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def train_deeplabv3(model, train_loader, val_loader, epochs=30, lr=0.001):
    """Train DeepLabV3 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining DeepLabV3 on {device}")

    model = model.to(device)
    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_iou = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_deeplabv3_model.pth")

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f} | "
                f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}"
            )

    return history


def visualize_predictions(model, images, masks, n_samples=4):
    """Visualize predictions."""
    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

    with torch.no_grad():
        for i in range(n_samples):
            img_tensor = torch.FloatTensor(images[i : i + 1]).to(device)
            pred = model(img_tensor)
            pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]

            # Input
            axes[i, 0].imshow(images[i].transpose(1, 2, 0))
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            # Ground truth
            axes[i, 1].imshow(masks[i, 0], cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            # Prediction
            axes[i, 2].imshow(pred_mask, cmap="gray")
            iou = calculate_iou(
                torch.FloatTensor(pred_mask[None, None, :, :]),
                torch.FloatTensor(masks[i : i + 1]),
            )
            axes[i, 2].set_title(f"Prediction (IoU: {iou:.3f})")
            axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("deeplabv3_predictions.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("DeepLabV3 for Semantic Segmentation")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic data...")
    from sklearn.model_selection import train_test_split

    images, masks = generate_synthetic_data(n_samples=150, img_size=256)
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    # Create dataloaders
    train_dataset = SegmentationDataset(X_train, y_train)
    val_dataset = SegmentationDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Create model
    print("\n2. Creating DeepLabV3 model...")
    model = DeepLabV3(num_classes=1, backbone="resnet50", pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print("\n3. Training model...")
    train_deeplabv3(model, train_loader, val_loader, epochs=30, lr=0.0001)

    # Visualize
    print("\n4. Visualizing predictions...")
    model.load_state_dict(torch.load("best_deeplabv3_model.pth"))
    visualize_predictions(model, X_val, y_val, n_samples=4)

    print("\n" + "=" * 70)
    print("DeepLabV3 Training Complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("✓ Atrous convolutions capture multi-scale context")
    print("✓ ASPP module for parallel multi-rate processing")
    print("✓ No decoder needed (simpler than U-Net)")
    print("✓ State-of-the-art performance")
    print("\nBest for: Large-scale segmentation, when accuracy is critical")


if __name__ == "__main__":
    main()
