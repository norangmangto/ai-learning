import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("Training MLP with TensorFlow...")

    # 1. Prepare Data
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode labels for categorical_crossentropy,
    # OR use sparse_categorical_crossentropy (easier)

    # 2. Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 3. Compile and Train
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, verbose=0)

    # 4. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"TensorFlow MLP Accuracy: {accuracy:.4f}")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    predictions_probs = model.predict(X_test, verbose=0)
    predictions = np.argmax(predictions_probs, axis=1)
    
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=[f"Digit {i}" for i in range(10)]))
    
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix shape: {cm.shape}")
    
    print("\n--- Sanity Checks ---")
    if np.all((predictions >= 0) & (predictions < 10)):
        print("✓ All predictions in valid range [0-9]")
    else:
        print("✗ WARNING: Some predictions outside valid range!")
    
    if accuracy > 0.9:
        print(f"✓ Excellent accuracy: {accuracy:.4f}")
    elif accuracy > 0.7:
        print(f"✓ Good accuracy: {accuracy:.4f}")
    else:
        print(f"⚠ Moderate accuracy: {accuracy:.4f}")
    
    unique_preds = np.unique(predictions)
    if len(unique_preds) == 10:
        print("✓ Model predicts all 10 classes")
    else:
        print(f"⚠ WARNING: Only predicts {len(unique_preds)} classes")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = np.all((predictions >= 0) & (predictions < 10)) and accuracy > 0.5 and len(unique_preds) >= 8
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
