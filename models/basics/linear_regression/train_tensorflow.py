import tensorflow as tf
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("Training Linear Regression with TensorFlow...")

    # 1. Prepare Data
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])

    # 3. Compile and Train
    model.compile(optimizer='sgd', loss='mse')

    model.fit(X_train, y_train, epochs=100, verbose=0)

    # 4. Evaluate
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"TensorFlow Linear Regression MSE: {loss:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()
