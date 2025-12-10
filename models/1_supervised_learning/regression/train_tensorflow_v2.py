import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train():
    print("Training Linear Regression with TensorFlow...")

    # 1. Prepare Data
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = tf.constant(X_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    X_test = tf.constant(X_test, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)

    # 2. Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])

    # 3. Compile Model
    model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

    # 4. Train Model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # 5. Evaluate
    predictions = model.predict(X_test, verbose=0).flatten()
    mse = mean_squared_error(y_test.numpy(), predictions)
    r2 = r2_score(y_test.numpy(), predictions)

    print(f"TensorFlow Linear Regression MSE: {mse:.4f}")

    # 6. QA Validation
    print("\n=== QA Validation ===")
    print(f"R² Score: {r2:.4f}")

    print("\n--- Sanity Checks ---")
    if np.all(np.isfinite(predictions)):
        print("✓ All predictions are finite")
    else:
        print("✗ WARNING: Some predictions are NaN or Inf!")

    if r2 > 0.5:
        print(f"✓ Good R² score: {r2:.4f}")
    else:
        print(f"⚠ Moderate R² score: {r2:.4f}")

    print("\n=== Overall Validation Result ===")
    validation_passed = np.all(np.isfinite(predictions)) and r2 > 0

    if validation_passed:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")

    return model


if __name__ == "__main__":
    train()
