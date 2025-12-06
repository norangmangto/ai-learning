import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("Training RNN (LSTM) with TensorFlow (MNIST Sequence)...")

    # MNIST data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Input shape: (28, 28) -> (timesteps, features)

    model = tf.keras.Sequential([
        # LSTM Layer
        tf.keras.layers.LSTM(128, input_shape=(28, 28), return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"TensorFlow RNN (LSTM) Accuracy: {accuracy:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()
