import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Hyperparameters
INPUT_SIZE = 28  # Each row has 28 pixels
SEQUENCE_LENGTH = 28  # 28 rows
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def train():
    print("Training RNN (LSTM) with PyTorch (MNIST Sequence)...")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epochs = 1  # Quick demo
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Reshape images to (batch_size, sequence_length, input_size)
            images = images.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

    # Test
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    print(f"PyTorch RNN (LSTM) Accuracy: {acc:.2f}%")

    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=[str(i) for i in range(10)]
        )
    )

    # Confusion Matrix (simplified view)
    cm = confusion_matrix(all_labels, all_preds)
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

    # Check 2: Model accuracy for MNIST with RNN
    if acc > 95:
        print(f"✓ Excellent accuracy: {acc:.2f}% (> 95%)")
    elif acc > 90:
        print(f"✓ Good accuracy: {acc:.2f}% (> 90%)")
    elif acc > 85:
        print(
            f"⚠ Moderate accuracy: {
        acc:.2f}% (RNNs can achieve >95% on MNIST)"
        )
    else:
        print(f"✗ WARNING: Poor accuracy: {acc:.2f}%")

    # Check 3: All classes are predicted
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 10:
        print("✓ Model predicts all 10 digit classes")
    else:
        print(
            f"⚠ WARNING: Model only predicts {
        len(unique_preds)} out of 10 classes"
        )

    # Check 4: Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum() * 100
            status = "✓" if class_acc > 85 else "⚠"
            print(f"  {status} Digit {i}: {class_acc:.2f}%")

    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all((all_preds >= 0) & (all_preds < 10))
        and acc > 80
        and len(unique_preds) >= 8
    )

    if validation_passed:
        print("✓ Model validation PASSED - RNN is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")

    print("\nDone.")


if __name__ == "__main__":
    train()
