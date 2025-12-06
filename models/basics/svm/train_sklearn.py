import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def train():
    print("Training SVM with Scikit-Learn...")

    # 1. Prepare Data
    iris = load_iris()
    X, y = iris.data, iris.target

    # SVM benefits greatly from scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Model
    # Using RBF kernel which is popular
    model = SVC(kernel='rbf', C=1.0, gamma='scale')

    # 3. Train Model
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Scikit-Learn SVM Accuracy: {acc:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()
