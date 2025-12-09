import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train():
    print("Training Random Forest with Scikit-Learn...")

    # 1. Prepare Data
    data = load_wine()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Model
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

    # 3. Train Model
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Scikit-Learn Random Forest Accuracy: {acc:.4f}")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    
    # Calculate comprehensive metrics
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=data.target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Sanity checks
    print("\n--- Sanity Checks ---")
    
    # Check 1: Predictions are in valid class range
    if np.all((predictions >= 0) & (predictions < 3)):
        print("✓ All predictions are in valid class range [0-2]")
    else:
        print("✗ WARNING: Some predictions are outside valid class range!")
    
    # Check 2: Model accuracy for wine dataset
    if acc > 0.9:
        print(f"✓ Excellent accuracy: {acc:.4f} (> 0.9)")
    elif acc > 0.7:
        print(f"✓ Good accuracy: {acc:.4f} (> 0.7)")
    else:
        print(f"⚠ Moderate accuracy: {acc:.4f} (Random Forest usually performs well on this dataset)")
    
    # Check 3: All classes are predicted
    unique_preds = np.unique(predictions)
    if len(unique_preds) == 3:
        print("✓ Model predicts all 3 wine classes")
    else:
        print(f"⚠ WARNING: Model only predicts {len(unique_preds)} out of 3 classes")
    
    # Check 4: Feature importance analysis
    print("\nTop 5 Important Features:")
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1][:5]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. Feature {idx} ({data.feature_names[idx]}): {feature_importance[idx]:.4f}")
    
    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all((predictions >= 0) & (predictions < 3)) and
        acc > 0.6 and
        len(unique_preds) == 3
    )
    
    if validation_passed:
        print("✓ Model validation PASSED - Model is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
