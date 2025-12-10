"""
DeepLabV3 for Semantic Segmentation

This script demonstrates:
1. Atrous (dilated) convolutions for multi-scale context
2. Atrous Spatial Pyramid Pooling (ASPP)
3. Encoder-decoder with strong features
4. Better boundary segmentation
5. State-of-the-art performance

Architecture: ResNet backbone + ASPP + decoder
Key innovation: Atrous convolutions capture multi-scale features
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
    "num_classes": 5,
    "image_size": 256,
    "output_stride": 16,  # 8 or 16
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 0.0001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "output_dir": "results/deeplabv3_segmentation",
}


class ASPPConv(nn.Sequential):
    """Atrous Spatial Pyramid Pooling convolution"""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    """Global average pooling branch"""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return nn.functional.interpolate(
            x, size=size, mode="bilinear", align_corners=False
        )


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling

    Key component of DeepLabV3
    Captures multi-scale context using parallel atrous convolutions
    with different dilation rates
    """

    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()

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
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))

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

        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3(nn.Module):
    """
    DeepLabV3 for Semantic Segmentation

    Architecture:
    1. ResNet backbone with atrous convolutions
    2. ASPP module for multi-scale features
    3. Decoder for upsampling
    4. Per-pixel classification
    """

    def __init__(self, num_classes=21, output_stride=16, pretrained=True):
        super().__init__()

        print("\n" + "=" * 80)
        print("BUILDING DEEPLABV3")
        print("=" * 80)

        print(f"\nNumber of classes: {num_classes}")
        print(f"Output stride: {output_stride}")

        # ResNet-50 backbone
        resnet = models.resnet50(pretrained=pretrained)

        # Modify stride in layer4 for output_stride=16
        if output_stride == 16:
            # Dilate layer4
            self._replace_stride_with_dilation(resnet.layer4, dilation=2)
        elif output_stride == 8:
            # Dilate layer3 and layer4
            self._replace_stride_with_dilation(resnet.layer3, dilation=2)
            self._replace_stride_with_dilation(resnet.layer4, dilation=4)

        # Encoder
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16 (or 1/8 if output_stride=8)
        self.layer4 = resnet.layer4  # 1/16 or 1/8

        # ASPP
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            atrous_rates = [6, 12, 18]

        self.aspp = ASPP(2048, atrous_rates)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1),
        )

        print(f"Parameters: {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def _replace_stride_with_dilation(self, layer, dilation):
        """Replace stride with dilation in a layer"""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                    if module.kernel_size == (3, 3):
                        module.dilation = (dilation, dilation)
                        module.padding = (dilation, dilation)
                elif module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x)

        # Upsample to input size
        x = nn.functional.interpolate(
            x, size=input_size, mode="bilinear", align_corners=False
        )

        return x


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
                cls = np.random.randint(1, num_classes)

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

                else:
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
                cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
                r = (x2 - x1) // 2
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
                            mask[y, x] = cls

            else:
                draw_img.polygon(coords, fill=color)
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

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0


def train_deeplabv3(model, train_loader, val_loader, device):
    """Train DeepLabV3"""
    print("\n" + "=" * 80)
    print("TRAINING DEEPLABV3")
    print("=" * 80)

    model.to(device)

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

            outputs = model(images)
            loss = criterion(outputs, masks)

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
    plt.savefig(output_dir / "deeplabv3_results.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved results to {output_dir / 'deeplabv3_results.png'}")


def main():
    print("=" * 80)
    print("DEEPLABV3 SEMANTIC SEGMENTATION")
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
    model = DeepLabV3(
        num_classes=CONFIG["num_classes"], output_stride=CONFIG["output_stride"]
    )

    # Train
    model, history = train_deeplabv3(model, train_loader, val_loader, CONFIG["device"])

    # Visualize
    visualize_segmentation(model, val_loader, CONFIG["device"])

    print("\n" + "=" * 80)
    print("DEEPLABV3 TRAINING COMPLETED")
    print("=" * 80)

    print("\nKey Concepts:")
    print("✓ Atrous (dilated) convolutions")
    print("✓ ASPP for multi-scale context")
    print("✓ ResNet backbone with modified stride")
    print("✓ State-of-the-art segmentation")

    print("\nASPP Module:")
    print("- Parallel atrous convolutions (rates: 6, 12, 18)")
    print("- Global average pooling branch")
    print("- Captures multi-scale features")
    print("- No information loss from pooling")

    print("\nAdvantages:")
    print("- Better boundary segmentation")
    print("- Multi-scale context modeling")
    print("- Efficient computation")
    print("- Strong feature representations")

    print("\nApplications:")
    print("- Autonomous driving")
    print("- Medical image segmentation")
    print("- Scene understanding")
    print("- Robot vision")


if __name__ == "__main__":
    main()
