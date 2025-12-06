import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    print("Done.")

if __name__ == "__main__":
    train()
