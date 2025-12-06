import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def train():
    print("Training Logistic Regression with PyTorch...")

    # 1. Prepare Data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # y_test labels (just numpy is fine for accuracy check)

    # 2. Build Model
    model = LogisticRegressionModel(input_dim=20)

    # 3. Train Model
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 4. Evaluate
    model.eval()
    with torch.no_grad():
        predicted_probs = model(X_test_tensor)
        predictions = (predicted_probs >= 0.5).float().numpy().flatten()

    acc = accuracy_score(y_test, predictions)
    print(f"PyTorch Logistic Regression Accuracy: {acc:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()
