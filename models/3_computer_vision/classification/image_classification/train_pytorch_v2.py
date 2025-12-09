"""
Image Classification with EfficientNet (lightweight alternative)
Alternative to standard CNN approaches
"""

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train():
    print("Training Image Classification with EfficientNet...")

    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    except:
        print("Warning: Could not load CIFAR-10. Using synthetic data...")
        train_dataset = create_synthetic_dataset(2000, transform)
        test_dataset = create_synthetic_dataset(500, transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Dataset size - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # 2. Load Pre-trained EfficientNet Model
    model = models.efficientnet_b0(pretrained=True)

    # Modify final layer for 10 classes (CIFAR-10)
    model.classifier[-1] = nn.Linear(1280, 10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 3. Train Model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # 4. Evaluate
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nEfficientNet Classification Accuracy: {accuracy:.4f}")

    # 5. QA Validation
    print("\n=== QA Validation ===")
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"F1-Score (weighted): {f1:.4f}")

    print("\n--- Sanity Checks ---")
    if accuracy >= 0.7:
        print(f"✓ Good accuracy: {accuracy:.4f}")
    else:
        print(f"⚠ Moderate accuracy: {accuracy:.4f}")

    print("\n=== Overall Validation Result ===")
    validation_passed = accuracy >= 0.6

    if validation_passed:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")

    return model


def create_synthetic_dataset(size, transform):
    """Create synthetic image dataset"""
    from torch.utils.data import Dataset
    from PIL import Image
    import io

    class SyntheticImageDataset(Dataset):
        def __init__(self, size, transform):
            self.size = size
            self.transform = transform

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Create random image
            img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            if self.transform:
                img = self.transform(img)

            label = idx % 10
            return img, label

    return SyntheticImageDataset(size, transform)


if __name__ == "__main__":
    train()
