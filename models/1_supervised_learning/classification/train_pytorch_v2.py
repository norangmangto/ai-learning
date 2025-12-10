import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def train():
    print("Training MLP with PyTorch (Alternative Architecture)...")

    # 1. Prepare Data
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_test = (X_test - X_mean) / (X_std + 1e-8)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 2. Build Model with Batch Norm and Dropout
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(20, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.dropout1 = nn.Dropout(0.3)

            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.dropout2 = nn.Dropout(0.3)

            self.fc3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.dropout3 = nn.Dropout(0.2)

            self.fc4 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            x = torch.relu(self.bn3(self.fc3(x)))
            x = self.dropout3(x)
            x = self.sigmoid(self.fc4(x))
            return x

    model = MLP()

    # 3. Train Model
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 4. Evaluate
    model.eval()
    with torch.no_grad():
        predictions = (model(X_test) > 0.5).float().numpy()
        y_test_np = y_test.numpy()

    accuracy = accuracy_score(y_test_np, predictions)
    print(f"\nPyTorch MLP (v2) Accuracy: {accuracy:.4f}")

    # 5. QA Validation
    print("\n=== QA Validation ===")
    f1 = f1_score(y_test_np, predictions, average="binary")
    print(f"F1-Score: {f1:.4f}")

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


if __name__ == "__main__":
    train()
