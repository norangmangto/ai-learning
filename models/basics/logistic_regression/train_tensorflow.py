import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("Training Logistic Regression with TensorFlow...")

    # 1. Prepare Data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(20,))
    ])

    # 3. Compile and Train
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, verbose=0)

    # 4. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"TensorFlow Logistic Regression Accuracy: {accuracy:.4f}")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    predictions_probs = model.predict(X_test, verbose=0)
    predictions = (predictions_probs > 0.5).astype(int).flatten()
    
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')
    auc = roc_auc_score(y_test, predictions_probs)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:\n{cm}")
    
    print("\n--- Sanity Checks ---")
    if np.all((predictions_probs >= 0) & (predictions_probs <= 1)):
        print("✓ All probabilities are in [0, 1]")
    else:
        print("✗ WARNING: Some probabilities outside [0, 1]!")
    
    if set(predictions) <= {0, 1}:
        print("✓ All predictions are binary")
    else:
        print("✗ WARNING: Predictions contain non-binary values!")
    
    if accuracy > 0.6:
        print(f"✓ Good accuracy: {accuracy:.4f}")
    elif accuracy > 0.5:
        print(f"⚠ Moderate accuracy: {accuracy:.4f}")
    else:
        print(f"✗ WARNING: Poor accuracy: {accuracy:.4f}")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = set(predictions) <= {0, 1} and accuracy > 0.5 and auc > 0.5
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
