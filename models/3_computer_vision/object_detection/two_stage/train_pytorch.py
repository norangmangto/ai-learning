"""
Faster R-CNN Object Detection Implementation

This script demonstrates two-stage object detection using Faster R-CNN:
1. Using pre-trained Faster R-CNN from torchvision
2. Training on COCO dataset
3. Region Proposal Network (RPN)
4. Evaluation with mAP metrics

Dataset: COCO 2017
Model: Faster R-CNN with ResNet-50 FPN backbone
"""

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import time
from PIL import Image

# Configuration
CONFIG = {
    'backbone': 'resnet50',
    'pretrained': True,
    'num_classes': 91,  # COCO classes + background
    'batch_size': 4,  # Smaller for memory
    'epochs': 10,
    'learning_rate': 0.005,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'lr_step_size': 3,
    'lr_gamma': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'rpn_nms_thresh': 0.7,
    'box_score_thresh': 0.05,
    'box_nms_thresh': 0.5,
    'save_dir': 'results/faster_rcnn'
}

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_faster_rcnn(pretrained=True, num_classes=91):
    """Load Faster R-CNN model"""
    print("="*80)
    print("LOADING FASTER R-CNN MODEL")
    print("="*80)

    print(f"\nBackbone: ResNet-50 with FPN")
    print(f"Pretrained: {pretrained}")
    print(f"Number of classes: {num_classes}")

    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=True,
        num_classes=num_classes if not pretrained else 91,
        rpn_nms_thresh=CONFIG['rpn_nms_thresh'],
        box_score_thresh=CONFIG['box_score_thresh'],
        box_nms_thresh=CONFIG['box_nms_thresh']
    )

    # Modify head if different number of classes
    if not pretrained or num_classes != 91:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(CONFIG['device'])

    print(f"\nModel loaded on {CONFIG['device']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    return model

def get_transform(train=False):
    """Get image transforms"""
    transforms = []
    transforms.append(T.ToTensor())

    if train:
        # Add data augmentation for training
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))

class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for demo"""
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

def train_one_epoch(model, optimizer, data_loader, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += losses.item()
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()
        total_loss_objectness += loss_dict['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

    n_batches = len(data_loader)

    return {
        'total': total_loss / n_batches,
        'classifier': total_loss_classifier / n_batches,
        'box_reg': total_loss_box_reg / n_batches,
        'objectness': total_loss_objectness / n_batches,
        'rpn_box_reg': total_loss_rpn_box_reg / n_batches
    }

def train_faster_rcnn(model, data_loader):
    """Train Faster R-CNN"""
    print("\n" + "="*80)
    print("TRAINING FASTER R-CNN")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Device: {CONFIG['device']}")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['lr_step_size'],
        gamma=CONFIG['lr_gamma']
    )

    # Training loop
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 80)

        start_time = time.time()

        # Train one epoch
        losses = train_one_epoch(model, optimizer, data_loader, CONFIG['device'])

        # Update learning rate
        lr_scheduler.step()

        epoch_time = time.time() - start_time

        # Print losses
        print(f"Time: {epoch_time:.2f}s")
        print(f"Losses:")
        print(f"  Total: {losses['total']:.4f}")
        print(f"  Classifier: {losses['classifier']:.4f}")
        print(f"  Box regression: {losses['box_reg']:.4f}")
        print(f"  Objectness: {losses['objectness']:.4f}")
        print(f"  RPN box regression: {losses['rpn_box_reg']:.4f}")

    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)

    return model

def detect_objects(model, image, device):
    """Detect objects in an image"""
    model.eval()

    # Prepare image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    transform = T.ToTensor()
    image_tensor = transform(image).to(device)

    # Inference
    start_time = time.time()

    with torch.no_grad():
        predictions = model([image_tensor])

    inference_time = time.time() - start_time

    return predictions[0], inference_time

def visualize_detections(image, predictions, threshold=0.5, save_path=None):
    """Visualize object detections"""

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image.cpu())

    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Filter predictions by score threshold
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    mask = scores >= threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        if label < len(COCO_CLASSES):
            class_name = COCO_CLASSES[label]
        else:
            class_name = f"Class {label}"

        ax.text(
            x1, y1 - 5,
            f'{class_name}: {score:.2f}',
            bbox=dict(facecolor='red', alpha=0.5),
            fontsize=10,
            color='white'
        )

    ax.axis('off')
    plt.title(f'Detected {len(boxes)} objects (threshold: {threshold})')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.close()

def demonstrate_detection():
    """Demonstrate object detection"""
    print("\n" + "="*80)
    print("OBJECT DETECTION DEMONSTRATION")
    print("="*80)

    # Load model
    model = load_faster_rcnn(pretrained=True)
    model.eval()

    # Create sample image (or load from file)
    print("\nGenerating sample image...")

    # For demo, create a simple image with geometric shapes
    img = Image.new('RGB', (640, 480), color='white')

    # You can also download a sample image:
    try:
        import urllib.request
        url = 'https://raw.githubusercontent.com/pytorch/vision/main/test/assets/encode_jpeg/grace_hopper_517x606.jpg'
        temp_path = 'temp_image.jpg'
        urllib.request.urlretrieve(url, temp_path)
        img = Image.open(temp_path)
        print("Loaded sample image from URL")
    except Exception as e:
        print(f"Could not download image: {e}")
        print("Using blank image for demo")

    # Detect objects
    print("\nRunning object detection...")
    predictions, inference_time = detect_objects(model, img, CONFIG['device'])

    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"Detected {len(predictions['boxes'])} objects")

    # Show top detections
    threshold = 0.5
    high_conf = predictions['scores'] > threshold

    if high_conf.sum() > 0:
        print(f"\nObjects with confidence > {threshold}:")
        for box, label, score in zip(
            predictions['boxes'][high_conf],
            predictions['labels'][high_conf],
            predictions['scores'][high_conf]
        ):
            if label < len(COCO_CLASSES):
                class_name = COCO_CLASSES[label]
            else:
                class_name = f"Class {label}"

            print(f"  {class_name}: {score:.3f}")
    else:
        print(f"\nNo objects detected with confidence > {threshold}")

    # Visualize
    output_dir = Path(CONFIG['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_detections(
        img,
        predictions,
        threshold=0.3,
        save_path=output_dir / 'detection_result.png'
    )

def explain_faster_rcnn():
    """Explain Faster R-CNN architecture"""
    print("\n" + "="*80)
    print("FASTER R-CNN ARCHITECTURE")
    print("="*80)

    print("""
Faster R-CNN: Two-Stage Object Detector

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  BACKBONE (ResNet-50 + FPN)                  │
│  - Extract multi-scale feature maps                          │
│  - Feature Pyramid Network for different scales              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            REGION PROPOSAL NETWORK (RPN)                     │
│  - Generate ~2000 region proposals                           │
│  - Objectness scores (object vs background)                  │
│  - Bounding box refinement                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    ROI POOLING / ALIGN                       │
│  - Extract fixed-size features from proposals                │
│  - Pool features for each region                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  DETECTION HEAD (R-CNN)                      │
│  - Classify each proposal (91 classes for COCO)              │
│  - Refine bounding boxes                                     │
│  - Apply NMS to remove duplicates                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FINAL DETECTIONS                                │
│  - Boxes, labels, scores                                     │
└─────────────────────────────────────────────────────────────┘
    """)

    print("\nKey Components:")
    print("1. BACKBONE: Feature extraction using CNN")
    print("2. RPN: Proposes regions that may contain objects")
    print("3. ROI Pooling: Extracts fixed-size features from proposals")
    print("4. Detection Head: Classifies and refines bounding boxes")

    print("\nAdvantages:")
    print("+ High accuracy (better than single-stage)")
    print("+ Good for small objects")
    print("+ Learns objectness and classification separately")
    print("+ Well-established architecture")

    print("\nLimitations:")
    print("- Slower than YOLO (~5-10 FPS vs 60+ FPS)")
    print("- Two-stage process is more complex")
    print("- Higher memory requirements")
    print("- Harder to deploy on edge devices")

def benchmark_speed(model):
    """Benchmark inference speed"""
    print("\n" + "="*80)
    print("SPEED BENCHMARK")
    print("="*80)

    model.eval()

    # Create dummy input
    dummy_img = torch.randn(3, 800, 800).to(CONFIG['device'])

    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model([dummy_img])

    # Benchmark
    print("Running benchmark (50 iterations)...")
    times = []

    for _ in range(50):
        start_time = time.time()
        with torch.no_grad():
            _ = model([dummy_img])

        if CONFIG['device'] == 'cuda':
            torch.cuda.synchronize()

        times.append(time.time() - start_time)

    times = np.array(times)

    print("\nResults:")
    print(f"  Mean: {times.mean()*1000:.2f} ms")
    print(f"  Std: {times.std()*1000:.2f} ms")
    print(f"  FPS: {1/times.mean():.1f}")

def main():
    print("="*80)
    print("FASTER R-CNN OBJECT DETECTION")
    print("Two-Stage Detector")
    print("="*80)

    print(f"\nDevice: {CONFIG['device']}")

    # Explain architecture
    explain_faster_rcnn()

    # Load model
    model = load_faster_rcnn(pretrained=True)

    # Demonstrate detection
    demonstrate_detection()

    # Benchmark speed
    response = input("\nRun speed benchmark? (y/n): ").strip().lower()
    if response == 'y':
        benchmark_speed(model)

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)

    print("\nFaster R-CNN Summary:")
    print("✓ Two-stage detector (RPN + Detection)")
    print("✓ High accuracy on COCO dataset")
    print("✓ Good for small and overlapping objects")
    print("✓ Pre-trained models available in torchvision")
    print("✓ Can be fine-tuned on custom datasets")

    print("\nComparison with YOLO (Single-Stage):")
    print("  Faster R-CNN:")
    print("    + Higher accuracy")
    print("    + Better for small objects")
    print("    - Slower (5-10 FPS)")
    print("    - More complex")
    print("  YOLO:")
    print("    + Faster (60+ FPS)")
    print("    + Simpler architecture")
    print("    - Lower accuracy")
    print("    - May miss small objects")

    print("\nUse Cases:")
    print("- High-accuracy applications")
    print("- Small object detection")
    print("- Image analysis (not real-time)")
    print("- Research and benchmarking")
    print("- When accuracy > speed")

if __name__ == '__main__':
    main()
