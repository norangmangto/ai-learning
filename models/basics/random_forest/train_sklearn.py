import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    print("Done.")

if __name__ == "__main__":
    train()
