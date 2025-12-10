"""
Fully Convolutional Network (FCN) for Semantic Segmentation with PyTorch

FCN is the pioneering architecture for semantic segmentation:
- Replaces fully connected layers with convolutions
- Uses skip connections to combine features at different scales
- Three variants: FCN-32s, FCN-16s, FCN-8s (increasing accuracy)
- Simple yet effective architecture

This implementation includes:
- FCN-8s with VGG16 backbone
- Skip connections from pool3, pool4, pool5
- Bilinear upsampling for output
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


class FCN8s(nn.Module):
    """
    FCN-8s architecture for semantic segmentation.

    Uses VGG16 as backbone with skip connections from pool3, pool4, pool5.
    Provides finer segmentation than FCN-32s or FCN-16s.
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(FCN8s, self).__init__()

        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())

        # Divide into stages
        self.pool3 = nn.Sequential(*features[:17])  # Up to pool3
        self.pool4 = nn.Sequential(*features[17:24])  # Up to pool4
        self.pool5 = nn.Sequential(*features[24:])  # Up to pool5

        # Replace classifier with convolutions (fc6, fc7)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Score layers for each pool
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_fc7 = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Upsampling layers
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Initialize upsampling with bilinear kernel
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize transposed convolutions with bilinear interpolation."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c = m.weight.size(0)
                f = m.kernel_size[0]
                factor = (f + 1) // 2
                center = f / 2.0 - 0.5
                og = np.ogrid[:f, :f]
                filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                m.weight.data.copy_(torch.from_numpy(filt).view(1, 1, f, f).repeat(c, c, 1, 1))

    def forward(self, x):
        input_size = x.size()[2:]

        # Encoder with skip connections
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        # Fully convolutional layers
        fc6 = self.drop6(self.relu6(self.fc6(pool5)))
        fc7 = self.drop7(self.relu7(self.fc7(fc6)))

        # Score layers
        score_fc7 = self.score_fc7(fc7)
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        # Upsample and add skip connections
        upscore2 = self.upscore2(score_fc7)
        score_pool4c = score_pool4[:, :, 5:5+upscore2.size(2), 5:5+upscore2.size(3)]
        fuse_pool4 = upscore2 + score_pool4c

        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3c = score_pool3[:, :, 9:9+upscore_pool4.size(2), 9:9+upscore_pool4.size(3)]
        fuse_pool3 = upscore_pool4 + score_pool3c

        # Final upsampling to input size
        out = self.upscore8(fuse_pool3)
        out = out[:, :, 31:31+input_size[0], 31:31+input_size[1]].contiguous()

        return out


class FCN16s(nn.Module):
    """
    FCN-16s architecture (fewer skip connections than FCN-8s).
    Faster but slightly less accurate than FCN-8s.
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(FCN16s, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())

        self.pool4 = nn.Sequential(*features[:24])
        self.pool5 = nn.Sequential(*features[24:])

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_fc7 = nn.Conv2d(4096, num_classes, kernel_size=1)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c = m.weight.size(0)
                f = m.kernel_size[0]
                factor = (f + 1) // 2
                center = f / 2.0 - 0.5
                og = np.ogrid[:f, :f]
                filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                m.weight.data.copy_(torch.from_numpy(filt).view(1, 1, f, f).repeat(c, c, 1, 1))

    def forward(self, x):
        input_size = x.size()[2:]

        pool4 = self.pool4(x)
        pool5 = self.pool5(pool4)

        fc6 = self.drop6(self.relu6(self.fc6(pool5)))
        fc7 = self.drop7(self.relu7(self.fc7(fc6)))

        score_fc7 = self.score_fc7(fc7)
        score_pool4 = self.score_pool4(pool4)

        upscore2 = self.upscore2(score_fc7)
        score_pool4c = score_pool4[:, :, 5:5+upscore2.size(2), 5:5+upscore2.size(3)]
        fuse_pool4 = upscore2 + score_pool4c

        out = self.upscore16(fuse_pool4)
        out = out[:, :, 27:27+input_size[0], 27:27+input_size[1]].contiguous()

        return out


def generate_synthetic_data(n_samples=200, img_size=224):
    """Generate synthetic segmentation dataset."""
    print(f"Generating {n_samples} synthetic images ({img_size}x{img_size})...")

    images = []
    masks = []

    np.random.seed(42)

    for i in range(n_samples):
        img = np.zeros((3, img_size, img_size), dtype=np.float32)
        mask = np.zeros((1, img_size, img_size), dtype=np.float32)

        n_shapes = np.random.randint(2, 5)

        for _ in range(n_shapes):
            shape_type = np.random.choice(['circle', 'rectangle'])

            if shape_type == 'circle':
                cx = np.random.randint(30, img_size-30)
                cy = np.random.randint(30, img_size-30)
                radius = np.random.randint(20, 50)
                color = np.random.rand(3)

                y, x = np.ogrid[:img_size, :img_size]
                circle_mask = (x - cx)**2 + (y - cy)**2 <= radius**2

                for c in range(3):
                    img[c][circle_mask] = color[c]
                mask[0][circle_mask] = 1.0
            else:
                x1 = np.random.randint(20, img_size-80)
                y1 = np.random.randint(20, img_size-80)
                w = np.random.randint(40, 70)
                h = np.random.randint(40, 70)
                color = np.random.rand(3)

                for c in range(3):
                    img[c, y1:y1+h, x1:x1+w] = color[c]
                mask[0, y1:y1+h, x1:x1+w] = 1.0

        img += np.random.randn(3, img_size, img_size) * 0.05
        img = np.clip(img, 0, 1)

        images.append(img)
        masks.append(mask)

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.FloatTensor(images)
        self.masks = torch.FloatTensor(masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)
        dice_loss = 1 - dice

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


def train_fcn(model, train_loader, val_loader, epochs=30, lr=0.001):
    """Train FCN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining FCN on {device}")

    model = model.to(device)
    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    best_val_loss = float('inf')

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

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_fcn_model.pth')

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f} | "
                  f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

    return history


def compare_fcn_variants(X_train, y_train, X_val, y_val):
    """Compare FCN-8s and FCN-16s."""
    print("\n" + "="*70)
    print("Comparing FCN Variants")
    print("="*70)

    train_dataset = SegmentationDataset(X_train, y_train)
    val_dataset = SegmentationDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    variants = {
        'FCN-8s': FCN8s(num_classes=1, pretrained=False),
        'FCN-16s': FCN16s(num_classes=1, pretrained=False)
    }

    results = {}

    for name, model in variants.items():
        print(f"\n{name}:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

        history = train_fcn(model, train_loader, val_loader, epochs=15, lr=0.0001)
        results[name] = history

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, history in results.items():
        axes[0].plot(history['train_loss'], label=f'{name} Train')
        axes[0].plot(history['val_loss'], '--', label=f'{name} Val')
        axes[1].plot(history['train_iou'], label=f'{name} Train')
        axes[1].plot(history['val_iou'], '--', label=f'{name} Val')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training/Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('Training/Validation IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fcn_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, images, masks, n_samples=4):
    """Visualize predictions."""
    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3*n_samples))

    with torch.no_grad():
        for i in range(n_samples):
            img_tensor = torch.FloatTensor(images[i:i+1]).to(device)
            pred = model(img_tensor)
            pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]

            axes[i, 0].imshow(images[i].transpose(1, 2, 0))
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(masks[i, 0], cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_mask, cmap='gray')
            iou = calculate_iou(torch.FloatTensor(pred_mask[None, None, :, :]),
                               torch.FloatTensor(masks[i:i+1]))
            axes[i, 2].set_title(f'Prediction (IoU: {iou:.3f})')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('fcn_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("Fully Convolutional Networks (FCN) for Semantic Segmentation")
    print("="*70)

    # Generate data
    print("\n1. Generating synthetic data...")
    from sklearn.model_selection import train_test_split
    images, masks = generate_synthetic_data(n_samples=150, img_size=224)
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Compare variants
    print("\n2. Comparing FCN variants...")
    compare_fcn_variants(X_train, y_train, X_val, y_val)

    # Train best model
    print("\n3. Training FCN-8s (best variant)...")
    train_dataset = SegmentationDataset(X_train, y_train)
    val_dataset = SegmentationDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = FCN8s(num_classes=1, pretrained=False)
    history = train_fcn(model, train_loader, val_loader, epochs=20, lr=0.0001)

    # Visualize
    print("\n4. Visualizing predictions...")
    model.load_state_dict(torch.load('best_fcn_model.pth'))
    visualize_predictions(model, X_val, y_val, n_samples=4)

    print("\n" + "="*70)
    print("FCN Training Complete!")
    print("="*70)
    print("\nKey Features:")
    print("✓ Pioneering end-to-end segmentation architecture")
    print("✓ Replaces FC layers with convolutions")
    print("✓ Skip connections combine coarse and fine features")
    print("✓ FCN-8s provides finer segmentation than FCN-16s/32s")
    print("\nBest for: Real-time applications, simpler baseline models")


if __name__ == "__main__":
    main()
