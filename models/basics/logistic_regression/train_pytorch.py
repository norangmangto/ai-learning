import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

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
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    
    # Calculate comprehensive metrics
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')
    
    # Get probabilities for AUC
    probs_np = predicted_probs.numpy().flatten()
    auc_score = roc_auc_score(y_test, probs_np)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {auc_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Sanity checks
    print("\n--- Sanity Checks ---")
    
    # Check 1: All probabilities are in [0, 1]
    if np.all((probs_np >= 0) & (probs_np <= 1)):
        print("✓ All predicted probabilities are in valid range [0, 1]")
    else:
        print("✗ WARNING: Some probabilities are outside [0, 1]!")
    
    # Check 2: Predictions are binary
    if set(predictions) <= {0, 1}:
        print("✓ All predictions are binary (0 or 1)")
    else:
        print("✗ WARNING: Predictions contain non-binary values!")
    
    # Check 3: Model accuracy is better than random
    if acc > 0.6:
        print(f"✓ Good accuracy: {acc:.4f} (> 0.6)")
    elif acc > 0.5:
        print(f"⚠ Moderate accuracy: {acc:.4f} (better than random)")
    else:
        print(f"✗ WARNING: Poor accuracy: {acc:.4f} (not much better than random)")
    
    # Check 4: AUC score validation
    if auc_score > 0.8:
        print(f"✓ Excellent AUC score: {auc_score:.4f}")
    elif auc_score > 0.7:
        print(f"✓ Good AUC score: {auc_score:.4f}")
    elif auc_score > 0.5:
        print(f"⚠ Moderate AUC score: {auc_score:.4f}")
    else:
        print(f"✗ WARNING: Poor AUC score: {auc_score:.4f}")
    
    # Check 5: Class balance in predictions
    pred_class_0 = np.sum(predictions == 0)
    pred_class_1 = np.sum(predictions == 1)
    print(f"\nPrediction distribution: Class 0: {pred_class_0}, Class 1: {pred_class_1}")
    
    if pred_class_0 == 0 or pred_class_1 == 0:
        print("✗ WARNING: Model predicts only one class!")
    else:
        print("✓ Model predicts both classes")
    
    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all((probs_np >= 0) & (probs_np <= 1)) and
        set(predictions) <= {0, 1} and
        acc > 0.5 and
        pred_class_0 > 0 and pred_class_1 > 0
    )
    
    if validation_passed:
        print("✓ Model validation PASSED - Model is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
