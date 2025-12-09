import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x # Softmax in CrossEntropyLoss

def train():
    print("Training CNN with PyTorch (MNIST)...")

    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download=True might fail in some rigid environments, but usually fine locally
    # We use a small subset or full dataset
    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    except Exception as e:
        print(f"Failed to download MNIST: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # 2. Build Model
    model = SimpleCNN()

    # 3. Train Model
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    epochs = 1 # Just 1 epoch for quick demonstration
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        scheduler.step()

    # 4. Evaluate
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print(f'\nPyTorch CNN Accuracy: {acc:.2f}%')
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=[str(i) for i in range(10)]))
    
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
    
    # Check 2: Model accuracy for MNIST (CNN should perform well)
    if acc > 95:
        print(f"✓ Excellent accuracy: {acc:.2f}% (> 95%)")
    elif acc > 90:
        print(f"✓ Good accuracy: {acc:.2f}% (> 90%)")
    elif acc > 80:
        print(f"⚠ Moderate accuracy: {acc:.2f}% (CNNs usually achieve >95% on MNIST)")
    else:
        print(f"✗ WARNING: Poor accuracy: {acc:.2f}% (check model architecture)")
    
    # Check 3: All classes are predicted
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 10:
        print("✓ Model predicts all 10 digit classes")
    else:
        print(f"⚠ WARNING: Model only predicts {len(unique_preds)} out of 10 classes")
    
    # Check 4: Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        mask = all_targets == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_targets[mask]).sum() / mask.sum() * 100
            status = "✓" if class_acc > 90 else "⚠"
            print(f"  {status} Digit {i}: {class_acc:.2f}%")
    
    # Check 5: Test loss validation
    if test_loss < 0.1:
        print(f"\n✓ Low test loss: {test_loss:.6f}")
    else:
        print(f"\n⚠ Test loss: {test_loss:.6f} (might benefit from more training)")
    
    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all((all_preds >= 0) & (all_preds < 10)) and
        acc > 80 and
        len(unique_preds) >= 8
    )
    
    if validation_passed:
        print("✓ Model validation PASSED - CNN is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
