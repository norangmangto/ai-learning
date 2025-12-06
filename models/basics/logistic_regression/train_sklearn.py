import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    print("Done.")

if __name__ == "__main__":
    train()
