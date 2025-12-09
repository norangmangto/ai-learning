import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def train():
    print("Training Logistic Regression with Scikit-Learn...")

    # 1. Prepare Data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Model
    model = LogisticRegression()

    # 3. Train Model
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Scikit-Learn Logistic Regression Accuracy: {acc:.4f}")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:\n{cm}")
    
    print("\n--- Sanity Checks ---")
    if set(predictions) <= {0, 1}:
        print("✓ All predictions are binary")
    else:
        print("✗ WARNING: Predictions contain non-binary values!")
    
    if acc > 0.6:
        print(f"✓ Good accuracy: {acc:.4f}")
    elif acc > 0.5:
        print(f"⚠ Moderate accuracy: {acc:.4f}")
    else:
        print(f"✗ WARNING: Poor accuracy: {acc:.4f}")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = set(predictions) <= {0, 1} and acc > 0.5 and auc > 0.5
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
