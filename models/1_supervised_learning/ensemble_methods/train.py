import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import numpy as np


def train():
    print("Training XGBoost (Breast Cancer Dataset)...")

    # 1. Prepare Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Build Model
    # XGBoost can be used via Scikit-Learn API or native API. SKLearn API is
    # easier for beginners.
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # 3. Train Model
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"XGBoost Accuracy: {accuracy:.4f}")

    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")

    # Calculate comprehensive metrics
    precision = precision_score(y_test, predictions, average="binary")
    recall = recall_score(y_test, predictions, average="binary")
    f1 = f1_score(y_test, predictions, average="binary")

    # Get prediction probabilities for AUC
    pred_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_probs)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=data.target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:\n{cm}")

    # Sanity checks
    print("\n--- Sanity Checks ---")

    # Check 1: Predictions are binary
    if set(predictions) <= {0, 1}:
        print("✓ All predictions are binary (0 or 1)")
    else:
        print("✗ WARNING: Predictions contain non-binary values!")

    # Check 2: Model accuracy for breast cancer dataset
    if accuracy > 0.95:
        print(f"✓ Excellent accuracy: {accuracy:.4f} (> 0.95)")
    elif accuracy > 0.9:
        print(f"✓ Good accuracy: {accuracy:.4f} (> 0.9)")
    elif accuracy > 0.8:
        print(f"⚠ Moderate accuracy: {accuracy:.4f} (room for improvement)")
    else:
        print(f"✗ WARNING: Poor accuracy: {accuracy:.4f}")

    # Check 3: AUC score validation
    if auc > 0.95:
        print(f"✓ Excellent AUC score: {auc:.4f}")
    elif auc > 0.85:
        print(f"✓ Good AUC score: {auc:.4f}")
    else:
        print(f"⚠ Moderate AUC score: {auc:.4f}")

    # Check 4: Both classes are predicted
    unique_preds = np.unique(predictions)
    if len(unique_preds) == 2:
        print("✓ Model predicts both classes (benign and malignant)")
    else:
        print(f"✗ WARNING: Model only predicts {len(unique_preds)} class!")

    # Check 5: Feature importance
    print("\nTop 5 Important Features:")
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1][:5]
    for i, idx in enumerate(indices):
        print(
            f"  {
        i+
        1}. Feature {idx} ({
            data.feature_names[idx]}): {
                feature_importance[idx]:.4f}"
        )

    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        set(predictions) <= {0, 1}
        and accuracy > 0.7
        and auc > 0.7
        and len(unique_preds) == 2
    )

    if validation_passed:
        print("✓ Model validation PASSED - Model is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")

    print("\nDone.")


if __name__ == "__main__":
    train()
