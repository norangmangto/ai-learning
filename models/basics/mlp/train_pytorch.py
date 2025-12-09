import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def train():
    print("Training MLP with PyTorch...")

    # 1. Prepare Data
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 2. Build Model
    model = MLP(input_dim=64, num_classes=10)

    # 3. Train Model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 4. Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs.data, 1)

    acc = accuracy_score(y_test, predictions.numpy())
    print(f"PyTorch MLP Accuracy: {acc:.4f}")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    
    y_pred = predictions.numpy()
    
    # Calculate comprehensive metrics
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[f"Digit {i}" for i in range(10)]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Sanity checks
    print("\n--- Sanity Checks ---")
    
    # Check 1: Predictions are in valid class range
    if np.all((y_pred >= 0) & (y_pred < 10)):
        print("✓ All predictions are in valid class range [0-9]")
    else:
        print("✗ WARNING: Some predictions are outside valid class range!")
    
    # Check 2: Model accuracy is reasonable for digits dataset
    if acc > 0.9:
        print(f"✓ Excellent accuracy: {acc:.4f} (> 0.9)")
    elif acc > 0.7:
        print(f"✓ Good accuracy: {acc:.4f} (> 0.7)")
    elif acc > 0.5:
        print(f"⚠ Moderate accuracy: {acc:.4f} (room for improvement)")
    else:
        print(f"✗ WARNING: Poor accuracy: {acc:.4f}")
    
    # Check 3: All classes are predicted
    unique_preds = np.unique(y_pred)
    if len(unique_preds) == 10:
        print("✓ Model predicts all 10 digit classes")
    else:
        print(f"⚠ WARNING: Model only predicts {len(unique_preds)} out of 10 classes: {unique_preds}")
    
    # Check 4: Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_test[mask]).sum() / mask.sum()
            status = "✓" if class_acc > 0.7 else "⚠"
            print(f"  {status} Digit {i}: {class_acc:.4f}")
    
    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all((y_pred >= 0) & (y_pred < 10)) and
        acc > 0.5 and
        len(unique_preds) >= 8  # At least most classes predicted
    )
    
    if validation_passed:
        print("✓ Model validation PASSED - Model is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
