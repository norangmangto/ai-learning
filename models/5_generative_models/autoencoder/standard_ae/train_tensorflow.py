import tensorflow as tf
from tensorflow.keras import layers, losses
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    
    # Evaluate reconstruction quality on test batch
    test_batch = x_test[:320]  # Use subset for evaluation
    reconstructed = autoencoder.predict(test_batch, verbose=0)
    
    # Calculate reconstruction error
    mse = mean_squared_error(test_batch.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(test_batch.flatten(), reconstructed.flatten())
    rmse = np.sqrt(mse)
    
    print(f"\nReconstruction Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    print("\n--- Sanity Checks ---")
    
    # Check 1: Reconstructions are in valid range [0, 1]
    if np.all((reconstructed >= 0) & (reconstructed <= 1)):
        print("✓ All reconstructed pixel values in valid range [0, 1]")
    else:
        out_of_range = np.sum((reconstructed < 0) | (reconstructed > 1))
        print(f"⚠ WARNING: {out_of_range} pixel values outside [0, 1] range")
    
    # Check 2: Reconstruction quality
    if mse < 0.01:
        print(f"✓ Excellent reconstruction quality: MSE = {mse:.6f}")
    elif mse < 0.05:
        print(f"✓ Good reconstruction quality: MSE = {mse:.6f}")
    elif mse < 0.1:
        print(f"⚠ Moderate reconstruction quality: MSE = {mse:.6f}")
    else:
        print(f"✗ WARNING: Poor reconstruction quality: MSE = {mse:.6f}")
    
    # Check 3: No NaN or Inf in reconstructions
    if np.all(np.isfinite(reconstructed)):
        print("✓ All reconstructed values are finite")
    else:
        print("✗ WARNING: Some reconstructed values are NaN or Inf!")
    
    # Check 4: Latent space dimensionality
    sample = x_test[:10]
    latent = autoencoder.encoder(sample).numpy()
    print(f"\n✓ Latent space dimension: {latent.shape[1]} (compressed from {28*28})")
    print(f"  Compression ratio: {28*28 / latent.shape[1]:.1f}x")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all(np.isfinite(reconstructed)) and
        mse < 0.2 and
        np.sum((reconstructed < -0.1) | (reconstructed > 1.1)) == 0
    )
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nReconstruction visualization saved to: autoencoder_images_tensorflow/result.png")

if __name__ == "__main__":
    train()
