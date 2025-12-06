import tensorflow as tf
import numpy as np
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("Training CNN with TensorFlow (MNIST)...")

    # 1. Prepare Data
    # MNIST is built-in
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape to (28, 28, 1) and normalize
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

    # 2. Build Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 3. Compile and Train
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Just 1 epoch for quick demonstration
    model.fit(x_train, y_train, epochs=1, verbose=1)

    # 4. Evaluate
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"TensorFlow CNN Accuracy: {accuracy:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()
