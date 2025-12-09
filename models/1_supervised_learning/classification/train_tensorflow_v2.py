import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train():
    print("Training Logistic Regression with TensorFlow...")

    # 1. Prepare Data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_test = (X_test - X_mean) / (X_std + 1e-8)

    X_train = tf.constant(X_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    X_test = tf.constant(X_test, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)

    # 2. Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 3. Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Train Model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # 5. Evaluate
    predictions = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test.numpy(), predictions)

    print(f"TensorFlow Logistic Regression Accuracy: {accuracy:.4f}")

    # 6. QA Validation
    print("\n=== QA Validation ===")
    f1 = f1_score(y_test.numpy(), predictions, average='binary')
    print(f"F1-Score: {f1:.4f}")

    print("\n--- Sanity Checks ---")
    if accuracy >= 0.5:
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
