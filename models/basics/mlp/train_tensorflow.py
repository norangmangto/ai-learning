import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("Training MLP with TensorFlow...")

    # 1. Prepare Data
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode labels for categorical_crossentropy,
    # OR use sparse_categorical_crossentropy (easier)

    # 2. Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 3. Compile and Train
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, verbose=0)

    # 4. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"TensorFlow MLP Accuracy: {accuracy:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()
