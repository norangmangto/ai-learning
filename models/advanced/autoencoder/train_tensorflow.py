import tensorflow as tf
from tensorflow.keras import layers, losses
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train():
    print("Training Autoencoder with TensorFlow (MNIST)...")
    os.makedirs("autoencoder_images_tensorflow", exist_ok=True)

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    latent_dim = 64
    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(x_train, x_train,
                    epochs=3,
                    shuffle=True,
                    validation_data=(x_test, x_test), verbose=1)

    print("Saving reconstruction example...")
    encoded_imgs = autoencoder.encoder(x_test[:1]).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x_test[0], cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(decoded_imgs[0], cmap='gray')
    axes[1].set_title('Reconstructed')
    plt.savefig("autoencoder_images_tensorflow/result.png")
    plt.close()

    print("TensorFlow Autoencoder Training Complete.")

if __name__ == "__main__":
    train()
