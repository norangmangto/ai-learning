import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def train():
    print("Training MLP with Scikit-Learn...")

    # 1. Prepare Data
    digits = load_digits()
    X, y = digits.data, digits.target

    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Model
    # 2 hidden layers with 64 and 32 neurons
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    # 3. Train Model
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Scikit-Learn MLP Accuracy: {acc:.4f}")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
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
    
    if acc > 0.9:
        print(f"✓ Excellent accuracy: {acc:.4f}")
    elif acc > 0.7:
        print(f"✓ Good accuracy: {acc:.4f}")
    else:
        print(f"⚠ Moderate accuracy: {acc:.4f}")
    
    unique_preds = np.unique(predictions)
    if len(unique_preds) == 10:
        print("✓ Model predicts all 10 classes")
    else:
        print(f"⚠ WARNING: Only predicts {len(unique_preds)} classes")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = np.all((predictions >= 0) & (predictions < 10)) and acc > 0.5 and len(unique_preds) >= 8
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
