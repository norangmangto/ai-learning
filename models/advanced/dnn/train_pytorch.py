import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

def train():
    print("Training DNN (Deep Neural Network) with PyTorch (FashionMNIST)...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    try:
        train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Failed to download FashionMNIST: {e}")
        return

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    model = DeepNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2  # Short training for demonstration
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss {running_loss/100:.4f}")
                running_loss = 0.0

    # Evaluate
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    acc = 100. * correct / len(test_loader.dataset)
    print(f"\nPyTorch DNN Accuracy on FashionMNIST: {acc:.2f}%")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # FashionMNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Confusion Matrix (simplified view)
    cm = confusion_matrix(all_targets, all_preds)
    print(f"\nConfusion Matrix shape: {cm.shape}")
    print("Diagonal (correct predictions per class):")
    print(np.diag(cm))
    
    # Sanity checks
    print("\n--- Sanity Checks ---")
    
    # Check 1: Predictions are in valid range
    if np.all((all_preds >= 0) & (all_preds < 10)):
        print("✓ All predictions are in valid class range [0-9]")
    else:
        print("✗ WARNING: Some predictions are outside valid range!")
    
    # Check 2: Model accuracy for FashionMNIST
    if acc > 88:
        print(f"✓ Excellent accuracy: {acc:.2f}% (> 88%)")
    elif acc > 85:
        print(f"✓ Good accuracy: {acc:.2f}% (> 85%)")
    elif acc > 80:
        print(f"⚠ Moderate accuracy: {acc:.2f}% (room for improvement)")
    else:
        print(f"✗ WARNING: Poor accuracy: {acc:.2f}%")
    
    # Check 3: All classes are predicted
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 10:
        print("✓ Model predicts all 10 fashion classes")
    else:
        print(f"⚠ WARNING: Model only predicts {len(unique_preds)} out of 10 classes")
    
    # Check 4: Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        mask = all_targets == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_targets[mask]).sum() / mask.sum() * 100
            status = "✓" if class_acc > 75 else "⚠"
            print(f"  {status} {class_names[i]}: {class_acc:.2f}%")
    
    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all((all_preds >= 0) & (all_preds < 10)) and
        acc > 75 and
        len(unique_preds) >= 8
    )
    
    if validation_passed:
        print("✓ Model validation PASSED - DNN is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
