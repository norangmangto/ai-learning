import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def train():
    print("Training XGBoost with PyTorch (Gradient Boosting Approximation)...")

    # 1. Prepare Data
    X, y = make_classification(
        n_samples=3000,
        n_features=30,
        n_informative=20,
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
    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    # 2. Build Model - Sequential boosting-like architecture
    class BoostNet(nn.Module):
        def __init__(self, input_size):
            super(BoostNet, self).__init__()
            self.boosters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(input_size, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 2),
                    )
                    for _ in range(3)
                ]
            )
            self.final = nn.Linear(2 * 3, 2)

        def forward(self, x):
            booster_outputs = []
            for booster in self.boosters:
                booster_outputs.append(booster(x))
            combined = torch.cat(booster_outputs, dim=1)
            return self.final(combined)

    model = BoostNet(30)

    # 3. Train Model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    epochs = 100
    batch_size = 32
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X_train):.4f}")

    # 4. Evaluate
    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(X_test), dim=1).numpy()
        y_test_np = y_test.numpy()

    accuracy = accuracy_score(y_test_np, predictions)
    print(f"\nPyTorch XGBoost (Boosting Approximation) Accuracy: {accuracy:.4f}")

    # 5. QA Validation
    print("\n=== QA Validation ===")
    f1 = f1_score(y_test_np, predictions, average="binary")
    print(f"F1-Score: {f1:.4f}")

    print("\n--- Sanity Checks ---")
    if accuracy >= 0.75:
        print(f"✓ Excellent accuracy: {accuracy:.4f}")
    elif accuracy >= 0.65:
        print(f"✓ Good accuracy: {accuracy:.4f}")
    else:
        print(f"⚠ Moderate accuracy: {accuracy:.4f}")

    print("\n=== Overall Validation Result ===")
    validation_passed = accuracy >= 0.65

    if validation_passed:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")

    return model


if __name__ == "__main__":
    train()
