"""
Anchor-Free Object Detection (CenterNet)

This script demonstrates:
1. Anchor-free detection without predefined anchor boxes
2. Objects as keypoints (center points)
3. Heatmap-based object localization
4. Size and offset regression
5. Simpler than anchor-based methods (Faster R-CNN, YOLO)

Architecture: CenterNet with hourglass or ResNet backbone
Advantages: No anchor tuning, simpler, faster, one-stage
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
import matplotlib.patches as patches

# Configuration
CONFIG = {
    "num_classes": 3,  # Number of object categories
    "image_size": 512,
    "output_stride": 4,  # Downsampling factor
    "heatmap_size": 128,  # 512 / 4
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 0.0001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "output_dir": "results/anchor_free_detection",
}


def gaussian_2d(shape, sigma=1):
    """Generate 2D gaussian kernel"""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draw gaussian circle on heatmap

    Args:
        heatmap: Target heatmap
        center: Object center (x, y)
        radius: Gaussian radius
        k: Peak value (1.0 for object center)
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


class CenterNetBackbone(nn.Module):
    """ResNet-based backbone for CenterNet"""

    def __init__(self, pretrained=True):
        super().__init__()

        # Use ResNet18 as backbone
        resnet = models.resnet18(pretrained=pretrained)

        # Remove final layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class CenterNet(nn.Module):
    """
    CenterNet for anchor-free object detection

    Predicts:
    1. Heatmap: Object center probabilities
    2. Size: Object width and height
    3. Offset: Sub-pixel offset for precise localization
    """

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        print("\n" + "=" * 80)
        print("BUILDING CENTERNET MODEL")
        print("=" * 80)

        print(f"\nNumber of classes: {num_classes}")

        # Backbone
        self.backbone = CenterNetBackbone(pretrained=pretrained)

        # Upsampling layers (deconvolution)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Prediction heads
        # Heatmap: probability of object center for each class
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
            nn.Sigmoid(),
        )

        # Size: width and height of bounding box
        self.size_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 2, 1)
        )

        # Offset: sub-pixel offset for precise localization
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 2, 1)
        )

        print(f"Parameters: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self, x):
        # Backbone
        x = self.backbone(x)

        # Upsampling
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        # Predictions
        heatmap = self.heatmap_head(x)
        size = self.size_head(x)
        offset = self.offset_head(x)

        return heatmap, size, offset


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Most pixels are background (no object center)
    Focal loss down-weights easy examples
    """

    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, C, H, W]
            target: Target heatmap [B, C, H, W]
        """
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, self.beta)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


class AnchorFreeDataset(Dataset):
    """Dataset for anchor-free detection"""

    def __init__(self, num_samples=1000, image_size=512, num_classes=3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.heatmap_size = image_size // CONFIG["output_stride"]

        # Generate samples
        self.samples = []
        for _ in range(num_samples):
            # Random number of objects
            num_objects = np.random.randint(1, 6)

            objects = []
            for _ in range(num_objects):
                # Random bounding box
                w = np.random.randint(30, 150)
                h = np.random.randint(30, 150)
                x = np.random.randint(0, image_size - w)
                y = np.random.randint(0, image_size - h)

                # Random class
                cls = np.random.randint(0, num_classes)

                objects.append({"bbox": [x, y, w, h], "class": cls})

            self.samples.append(objects)

    def __len__(self):
        return self.num_samples

    def create_targets(self, objects):
        """Create heatmap, size, and offset targets"""
        # Initialize targets
        heatmap = np.zeros(
            (self.num_classes, self.heatmap_size, self.heatmap_size), dtype=np.float32
        )
        size = np.zeros((2, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        offset = np.zeros((2, self.heatmap_size, self.heatmap_size), dtype=np.float32)

        for obj in objects:
            x, y, w, h = obj["bbox"]
            cls = obj["class"]

            # Center point
            cx = (x + w / 2) / CONFIG["output_stride"]
            cy = (y + h / 2) / CONFIG["output_stride"]

            # Integer center
            cx_int = int(cx)
            cy_int = int(cy)

            if 0 <= cx_int < self.heatmap_size and 0 <= cy_int < self.heatmap_size:
                # Gaussian radius based on IoU
                radius = max(0, int(min(w, h) / CONFIG["output_stride"] / 4))
                radius = max(2, radius)

                # Draw gaussian on heatmap
                draw_gaussian(heatmap[cls], (cx_int, cy_int), radius)

                # Size target (normalized by output stride)
                size[0, cy_int, cx_int] = w / CONFIG["output_stride"]
                size[1, cy_int, cx_int] = h / CONFIG["output_stride"]

                # Offset target (sub-pixel offset)
                offset[0, cy_int, cx_int] = cx - cx_int
                offset[1, cy_int, cx_int] = cy - cy_int

        return heatmap, size, offset

    def __getitem__(self, idx):
        objects = self.samples[idx]

        # Create image with colored rectangles
        img = Image.new(
            "RGB", (self.image_size, self.image_size), color=(240, 240, 240)
        )
        draw = ImageDraw.Draw(img)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for obj in objects:
            x, y, w, h = obj["bbox"]
            cls = obj["class"]
            color = colors[cls]

            draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0), width=2)

        # Convert to tensor
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
            img_tensor
        )

        # Create targets
        heatmap, size, offset = self.create_targets(objects)

        return (
            img_tensor,
            torch.FloatTensor(heatmap),
            torch.FloatTensor(size),
            torch.FloatTensor(offset),
        )


def train_centernet(model, train_loader, val_loader, device):
    """Train CenterNet"""
    print("\n" + "=" * 80)
    print("TRAINING CENTERNET")
    print("=" * 80)

    model.to(device)

    # Loss functions
    heatmap_loss_fn = FocalLoss()
    size_loss_fn = nn.L1Loss()
    offset_loss_fn = nn.L1Loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(CONFIG["epochs"]):
        # Training
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for images, heatmaps, sizes, offsets in progress_bar:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            sizes = sizes.to(device)
            offsets = offsets.to(device)

            # Forward pass
            pred_heatmaps, pred_sizes, pred_offsets = model(images)

            # Compute losses
            hm_loss = heatmap_loss_fn(pred_heatmaps, heatmaps)
            size_loss = size_loss_fn(pred_sizes, sizes)
            offset_loss = offset_loss_fn(pred_offsets, offsets)

            # Total loss
            loss = hm_loss + 0.1 * size_loss + offset_loss

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

        with torch.no_grad():
            for images, heatmaps, sizes, offsets in val_loader:
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                sizes = sizes.to(device)
                offsets = offsets.to(device)

                pred_heatmaps, pred_sizes, pred_offsets = model(images)

                hm_loss = heatmap_loss_fn(pred_heatmaps, heatmaps)
                size_loss = size_loss_fn(pred_sizes, sizes)
                offset_loss = offset_loss_fn(pred_offsets, offsets)

                loss = hm_loss + 0.1 * size_loss + offset_loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        print(
            f"\nEpoch {
        epoch+
        1}: Train Loss = {
            avg_train_loss:.4f}, Val Loss = {
                avg_val_loss:.4f}"
        )

        scheduler.step()

    return model, history


def decode_predictions(heatmap, size, offset, threshold=0.3, top_k=100):
    """Decode predictions to bounding boxes"""
    batch_size, num_classes, height, width = heatmap.shape

    # Find local maxima (object centers)
    heatmap = torch.sigmoid(heatmap)  # Ensure in [0, 1]

    # Max pooling to find peaks
    kernel = 3
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()

    # Apply threshold
    heatmap = heatmap * keep
    heatmap = heatmap.view(batch_size, num_classes, -1)

    # Get top-k peaks
    topk_scores, topk_inds = torch.topk(heatmap, top_k)

    detections = []

    for b in range(batch_size):
        batch_dets = []

        for c in range(num_classes):
            scores = topk_scores[b, c]
            inds = topk_inds[b, c]

            # Filter by threshold
            keep_inds = scores > threshold
            scores = scores[keep_inds]
            inds = inds[keep_inds]

            # Convert indices to (y, x)
            ys = (inds // width).float()
            xs = (inds % width).float()

            for i in range(len(scores)):
                y, x = int(ys[i]), int(xs[i])

                # Get size and offset
                w = size[b, 0, y, x] * CONFIG["output_stride"]
                h = size[b, 1, y, x] * CONFIG["output_stride"]
                offset_x = offset[b, 0, y, x]
                offset_y = offset[b, 1, y, x]

                # Adjust center
                cx = (x + offset_x) * CONFIG["output_stride"]
                cy = (y + offset_y) * CONFIG["output_stride"]

                # Bounding box
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                batch_dets.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "score": float(scores[i]),
                        "class": c,
                    }
                )

        detections.append(batch_dets)

    return detections


def visualize_detections(model, val_loader, device, num_samples=4):
    """Visualize detection results"""
    model.eval()

    images, _, _, _ = next(iter(val_loader))
    images = images[:num_samples].to(device)

    with torch.no_grad():
        heatmaps, sizes, offsets = model(images)

    detections = decode_predictions(heatmaps, sizes, offsets, threshold=0.5)

    # Plot
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))

    colors = ["red", "green", "blue"]

    for i in range(num_samples):
        # Denormalize image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)

        # Draw detections
        for det in detections[i]:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class"]
            score = det["score"]

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=colors[cls],
                facecolor="none",
            )
            axes[i].add_patch(rect)
            axes[i].text(
                x1, y1, f"{score:.2f}", color=colors[cls], fontsize=10, weight="bold"
            )

        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis("off")

    plt.tight_layout()

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "detections.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved detections to {output_dir / 'detections.png'}")


def main():
    print("=" * 80)
    print("ANCHOR-FREE OBJECT DETECTION (CENTERNET)")
    print("=" * 80)

    print(f"\nDevice: {CONFIG['device']}")

    # Create dataset
    train_dataset = AnchorFreeDataset(800, CONFIG["image_size"], CONFIG["num_classes"])
    val_dataset = AnchorFreeDataset(200, CONFIG["image_size"], CONFIG["num_classes"])

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
    model = CenterNet(num_classes=CONFIG["num_classes"])

    # Train
    model, history = train_centernet(model, train_loader, val_loader, CONFIG["device"])

    # Visualize
    visualize_detections(model, val_loader, CONFIG["device"])

    print("\n" + "=" * 80)
    print("ANCHOR-FREE DETECTION COMPLETED")
    print("=" * 80)

    print("\nKey Concepts:")
    print("✓ Objects as keypoints (center points)")
    print("✓ Heatmap for object centers")
    print("✓ Size and offset regression")
    print("✓ No anchor boxes or NMS")

    print("\nAdvantages:")
    print("- Simpler than anchor-based methods")
    print("- No anchor hyperparameters")
    print("- One-stage detector (fast)")
    print("- Good for small objects")

    print("\nApplications:")
    print("- Real-time object detection")
    print("- Pedestrian detection")
    print("- Autonomous driving")
    print("- Surveillance systems")


if __name__ == "__main__":
    main()
