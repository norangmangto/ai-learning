import tensorflow as tf
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("Training Linear Regression with TensorFlow...")

    # 1. Prepare Data
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])

    # 3. Compile and Train
    model.compile(optimizer='sgd', loss='mse')

    model.fit(X_train, y_train, epochs=100, verbose=0)

    # 4. Evaluate
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"TensorFlow Linear Regression MSE: {loss:.4f}")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    predictions = model.predict(X_test, verbose=0)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(loss)
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    print("\n--- Sanity Checks ---")
    if np.all(np.isfinite(predictions)):
        print("✓ All predictions are finite")
    else:
        print("✗ WARNING: Some predictions are NaN or Inf!")
    
    if r2 > 0.5:
        print(f"✓ Good R² score: {r2:.4f}")
    elif r2 > 0:
        print(f"⚠ Moderate R² score: {r2:.4f}")
    else:
        print(f"✗ WARNING: Poor R² score: {r2:.4f}")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = np.all(np.isfinite(predictions)) and r2 > 0 and loss < np.var(y_test) * 2
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
