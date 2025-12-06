import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def train():
    print("Training XGBoost (Breast Cancer Dataset)...")

    # 1. Prepare Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Model
    # XGBoost can be used via Scikit-Learn API or native API. SKLearn API is easier for beginners.
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # 3. Train Model
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()
