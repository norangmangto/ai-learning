"""
Fully Convolutional Network (FCN) for Semantic Segmentation

This script demonstrates:
1. First end-to-end network for semantic segmentation
2. Fully convolutional architecture (no dense layers)
3. Skip connections for combining coarse and fine features
4. Upsampling through transposed convolutions
5. FCN-32s, FCN-16s, FCN-8s variants

Architecture: VGG backbone + upsampling + skip connections
Applications: Scene understanding, autonomous driving, medical imaging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    "num_classes": 5,  # Including background
    "model_variant": "fcn8s",  # 'fcn32s', 'fcn16s', 'fcn8s'
    "image_size": 256,
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 0.0001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "output_dir": "results/fcn_segmentation",
}


class FCN(nn.Module):
    """
    Fully Convolutional Network for Semantic Segmentation

    Variants:
    - FCN-32s: 32x upsampling from final layer
    - FCN-16s: Combines pool4 (16x) with final layer
    - FCN-8s: Combines pool3 (8x), pool4 (16x) with final layer
    """

    def __init__(self, num_classes=21, variant="fcn8s", pretrained=True):
        super().__init__()

        print("\n" + "=" * 80)
        print(f"BUILDING FCN-{variant.upper()}")
        print("=" * 80)

        print(f"\nNumber of classes: {num_classes}")
        print(f"Variant: {variant}")

        self.variant = variant

        # Load VGG16 backbone
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())

        # Split into different resolution levels
        # Pool3: 1/8, Pool4: 1/16, Pool5: 1/32
        self.pool3 = nn.Sequential(*features[:17])  # Up to pool3
        self.pool4 = nn.Sequential(*features[17:24])  # Pool3 to pool4
        self.pool5 = nn.Sequential(*features[24:])  # Pool4 to pool5

        # Replace classifier with convolutions
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, num_classes, 1)

        # Upsampling layers
        if variant == "fcn32s":
            # 32x upsampling
            self.upscore = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=64, stride=32, bias=False
            )

        elif variant == "fcn16s":
            # Score pool4
            self.score_pool4 = nn.Conv2d(512, num_classes, 1)

            # 2x upsampling to match pool4
            self.upscore2 = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=4, stride=2, bias=False
            )

            # 16x upsampling
            self.upscore16 = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=32, stride=16, bias=False
            )

        elif variant == "fcn8s":
            # Score pool4 and pool3
            self.score_pool4 = nn.Conv2d(512, num_classes, 1)
            self.score_pool3 = nn.Conv2d(256, num_classes, 1)

            # 2x upsampling to match pool4
            self.upscore2 = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=4, stride=2, bias=False
            )

            # 2x upsampling to match pool3
            self.upscore_pool4 = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=4, stride=2, bias=False
            )

            # 8x upsampling
            self.upscore8 = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=16, stride=8, bias=False
            )

        print(f"Parameters: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self, x):
        # Encoder
        pool3 = self.pool3(x)  # 1/8
        pool4 = self.pool4(pool3)  # 1/16
        pool5 = self.pool5(pool4)  # 1/32

        # Fully convolutional layers
        fc6 = self.relu6(self.fc6(pool5))
        fc6 = self.dropout6(fc6)

        fc7 = self.relu7(self.fc7(fc6))
        fc7 = self.dropout7(fc7)

        score_fr = self.score_fr(fc7)

        # Upsampling with skip connections
        if self.variant == "fcn32s":
            # Simple 32x upsampling
            output = self.upscore(score_fr)

        elif self.variant == "fcn16s":
            # Add pool4 skip connection
            score_pool4 = self.score_pool4(pool4)
            upscore2 = self.upscore2(score_fr)

            # Crop to match sizes
            upscore2 = self._crop(upscore2, score_pool4)

            # Combine
            fuse_pool4 = upscore2 + score_pool4

            # 16x upsampling
            output = self.upscore16(fuse_pool4)

        elif self.variant == "fcn8s":
            # Add pool4 skip connection
            score_pool4 = self.score_pool4(pool4)
            upscore2 = self.upscore2(score_fr)
            upscore2 = self._crop(upscore2, score_pool4)
            fuse_pool4 = upscore2 + score_pool4

            # Add pool3 skip connection
            score_pool3 = self.score_pool3(pool3)
            upscore_pool4 = self.upscore_pool4(fuse_pool4)
            upscore_pool4 = self._crop(upscore_pool4, score_pool3)
            fuse_pool3 = upscore_pool4 + score_pool3

            # 8x upsampling
            output = self.upscore8(fuse_pool3)

        # Crop to input size
        output = self._crop(output, x)

        return output

    def _crop(self, x, target):
        """Crop x to match target size"""
        _, _, h, w = target.shape
        return x[:, :, :h, :w]


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation"""

    def __init__(self, num_samples=1000, image_size=256, num_classes=5):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

        # Generate samples
        self.samples = []
        for _ in range(num_samples):
            # Random shapes
            num_shapes = np.random.randint(3, 8)
            shapes = []

            for _ in range(num_shapes):
                shape_type = np.random.choice(["rect", "circle", "polygon"])
                cls = np.random.randint(1, num_classes)  # 0 is background

                if shape_type == "rect":
                    w = np.random.randint(30, 100)
                    h = np.random.randint(30, 100)
                    x = np.random.randint(0, image_size - w)
                    y = np.random.randint(0, image_size - h)
                    shapes.append(("rect", [x, y, x + w, y + h], cls))

                elif shape_type == "circle":
                    r = np.random.randint(15, 50)
                    x = np.random.randint(r, image_size - r)
                    y = np.random.randint(r, image_size - r)
                    shapes.append(("circle", [x - r, y - r, x + r, y + r], cls))

                else:  # polygon
                    points = []
                    cx = np.random.randint(50, image_size - 50)
                    cy = np.random.randint(50, image_size - 50)
                    for _ in range(5):
                        angle = np.random.uniform(0, 2 * np.pi)
                        radius = np.random.randint(20, 40)
                        px = cx + int(radius * np.cos(angle))
                        py = cy + int(radius * np.sin(angle))
                        points.append((px, py))
                    shapes.append(("polygon", points, cls))

            self.samples.append(shapes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        shapes = self.samples[idx]

        # Create image
        img = Image.new(
            "RGB", (self.image_size, self.image_size), color=(255, 255, 255)
        )
        draw_img = ImageDraw.Draw(img)

        # Create mask
        mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for shape in shapes:
            shape_type, coords, cls = shape
            color = colors[(cls - 1) % len(colors)]

            if shape_type == "rect":
                draw_img.rectangle(coords, fill=color)
                x1, y1, x2, y2 = coords
                mask[y1:y2, x1:x2] = cls

            elif shape_type == "circle":
                draw_img.ellipse(coords, fill=color)
                x1, y1, x2, y2 = coords
                # Approximate circle mask
                cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
                r = (x2 - x1) // 2
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
                            mask[y, x] = cls

            else:  # polygon
                draw_img.polygon(coords, fill=color)
                # Approximate polygon mask (simplified)
                min_x = min(p[0] for p in coords)
                max_x = max(p[0] for p in coords)
                min_y = min(p[1] for p in coords)
                max_y = max(p[1] for p in coords)
                mask[min_y:max_y, min_x:max_x] = cls

        # Convert to tensors
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
            img_tensor
        )

        mask_tensor = torch.LongTensor(mask)

        return img_tensor, mask_tensor


def compute_iou(pred, target, num_classes):
    """Compute mean IoU"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append((intersection / union).item())

    # Mean IoU (ignore NaN)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0


def train_fcn(model, train_loader, val_loader, device):
    """Train FCN"""
    print("\n" + "=" * 80)
    print("TRAINING FCN")
    print("=" * 80)

    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(CONFIG["epochs"]):
        # Training
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for images, masks in progress_bar:
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
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

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

                # Compute IoU
                _, preds = torch.max(outputs, 1)
                iou = compute_iou(preds, masks, CONFIG["num_classes"])
                val_iou += iou

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        history["val_loss"].append(avg_val_loss)
        history["val_iou"].append(avg_val_iou)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val mIoU: {avg_val_iou:.4f}")

        scheduler.step()

    return model, history


def visualize_segmentation(model, val_loader, device, num_samples=4):
    """Visualize segmentation results"""
    model.eval()

    images, masks = next(iter(val_loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples]

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))

    for i in range(num_samples):
        # Image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        # Ground truth
        axes[i, 1].imshow(
            masks[i].numpy(), cmap="tab20", vmin=0, vmax=CONFIG["num_classes"] - 1
        )
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        # Prediction
        axes[i, 2].imshow(
            preds[i].cpu().numpy(), cmap="tab20", vmin=0, vmax=CONFIG["num_classes"] - 1
        )
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir
        / f'fcn_{
        CONFIG["model_variant"]}_results.png',
        dpi=150,
        bbox_inches="tight",
    )
    print(
        f"\nSaved results to {
        output_dir /
        f'fcn_{
            CONFIG['model_variant']}_results.png'}"
    )


def main():
    print("=" * 80)
    print("FULLY CONVOLUTIONAL NETWORK (FCN)")
    print("=" * 80)

    print(f"\nDevice: {CONFIG['device']}")

    # Create dataset
    train_dataset = SegmentationDataset(
        800, CONFIG["image_size"], CONFIG["num_classes"]
    )
    val_dataset = SegmentationDataset(200, CONFIG["image_size"], CONFIG["num_classes"])

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    # Model
    model = FCN(num_classes=CONFIG["num_classes"], variant=CONFIG["model_variant"])

    # Train
    model, history = train_fcn(model, train_loader, val_loader, CONFIG["device"])

    # Visualize
    visualize_segmentation(model, val_loader, CONFIG["device"])

    print("\n" + "=" * 80)
    print("FCN TRAINING COMPLETED")
    print("=" * 80)

    print("\nKey Concepts:")
    print("✓ First end-to-end semantic segmentation")
    print("✓ Fully convolutional (no dense layers)")
    print("✓ Skip connections combine features")
    print("✓ FCN-8s > FCN-16s > FCN-32s")

    print("\nArchitecture:")
    print("- VGG backbone for features")
    print("- Transposed convolutions for upsampling")
    print("- Skip connections from pool3, pool4")
    print("- Per-pixel classification")

    print("\nApplications:")
    print("- Scene understanding")
    print("- Autonomous driving")
    print("- Medical image analysis")
    print("- Satellite image segmentation")


if __name__ == "__main__":
    main()
