import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train():
    print("Training Linear Regression with PyTorch...")

    # 1. Prepare Data
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 2. Build Model
    model = nn.Linear(1, 1) # Simple linear layer

    # 3. Train Model
    criterion = nn.MSELoss()
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
        predictions = model(X_test_tensor)
        mse = criterion(predictions, y_test_tensor).item()

    print(f"PyTorch Linear Regression MSE: {mse:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()
