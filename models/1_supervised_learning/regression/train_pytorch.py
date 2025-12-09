import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    
    # Convert predictions to numpy for detailed metrics
    y_pred_np = predictions.numpy().flatten()
    y_test_np = y_test
    
    # Calculate comprehensive metrics
    mae = mean_absolute_error(y_test_np, y_pred_np)
    r2 = r2_score(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Sanity checks
    print("\n--- Sanity Checks ---")
    
    # Check 1: Predictions are finite
    if np.all(np.isfinite(y_pred_np)):
        print("✓ All predictions are finite (no NaN or Inf)")
    else:
        print("✗ WARNING: Some predictions are NaN or Inf!")
    
    # Check 2: R² score is reasonable (should be > 0 for meaningful model)
    if r2 > 0.5:
        print(f"✓ Good R² score: {r2:.4f} (> 0.5)")
    elif r2 > 0:
        print(f"⚠ Moderate R² score: {r2:.4f} (model explains some variance)")
    else:
        print(f"✗ WARNING: Poor R² score: {r2:.4f} (model performs worse than mean baseline)")
    
    # Check 3: Prediction distribution
    pred_mean = np.mean(y_pred_np)
    pred_std = np.std(y_pred_np)
    test_mean = np.mean(y_test_np)
    test_std = np.std(y_test_np)
    
    print(f"\nPrediction stats - Mean: {pred_mean:.2f}, Std: {pred_std:.2f}")
    print(f"Test data stats - Mean: {test_mean:.2f}, Std: {test_std:.2f}")
    
    # Check 4: Residuals analysis
    residuals = y_test_np - y_pred_np
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    print(f"\nResiduals - Mean: {residual_mean:.4f}, Std: {residual_std:.4f}")
    if abs(residual_mean) < 0.1 * test_std:
        print("✓ Residuals are well-centered around zero")
    else:
        print("⚠ Residuals may have systematic bias")
    
    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all(np.isfinite(y_pred_np)) and
        r2 > 0 and
        mse < np.var(y_test_np) * 2  # MSE should be reasonable compared to variance
    )
    
    if validation_passed:
        print("✓ Model validation PASSED - Model is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
