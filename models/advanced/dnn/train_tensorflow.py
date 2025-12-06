import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("Training DNN (Deep Neural Network) with TensorFlow (FashionMNIST)...")

    # 1. Prepare Data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 2. Build Model (Deep Architecture)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(64, activation='relu'),

        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 3. Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Train
    model.fit(train_images, train_labels, epochs=2, verbose=1)

    # 5. Evaluate
    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\nTensorFlow DNN Accuracy on FashionMNIST: {acc:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()
