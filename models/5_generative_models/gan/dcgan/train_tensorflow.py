import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))

    # Upsample to 14x14
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Upsample to 28x28
    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1)) # Logits
    return model

def train():
    print("Training GAN with TensorFlow (MNIST)...")
    os.makedirs("gan_images_tensorflow", exist_ok=True)

    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize to [-1, 1]

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    epochs = 2 # Short run
    for epoch in range(epochs):
        for image_batch in train_dataset:
            g_loss, d_loss = train_step(image_batch)

        print(f"Epoch {epoch+1}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")

        # Save one example
        noise = tf.random.normal([1, 100])
        gen_img = generator(noise, training=False)
        plt.imsave(f"gan_images_tensorflow/epoch_{epoch+1}.png", gen_img[0, :, :, 0] * 127.5 + 127.5, cmap='gray')

    print("TensorFlow GAN Training Complete.")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    
    # Generate final samples for evaluation
    num_samples = 100
    noise_test = tf.random.normal([num_samples, 100])
    generated_samples = generator(noise_test, training=False).numpy()
    
    # Test discriminator on generated samples
    d_fake_scores = tf.nn.sigmoid(discriminator(generated_samples, training=False)).numpy().flatten()
    
    # Test discriminator on real samples
    real_samples = next(iter(train_dataset))[:num_samples]
    d_real_scores = tf.nn.sigmoid(discriminator(real_samples, training=False)).numpy().flatten()
    
    print(f"\nDiscriminator Performance:")
    print(f"Mean score on real images: {d_real_scores.mean():.4f} (should be close to 1)")
    print(f"Mean score on fake images: {d_fake_scores.mean():.4f} (should be close to 0.5 for good GAN)")
    print(f"\nFinal Losses:")
    print(f"Discriminator loss: {d_loss:.4f}")
    print(f"Generator loss: {g_loss:.4f}")
    
    print("\n--- Sanity Checks ---")
    
    # Check 1: Generated images are in valid range [-1, 1]
    gen_min = generated_samples.min()
    gen_max = generated_samples.max()
    print(f"\nGenerated image range: [{gen_min:.3f}, {gen_max:.3f}]")
    if gen_min >= -1.5 and gen_max <= 1.5:
        print("✓ Generated images are in reasonable range")
    else:
        print("⚠ WARNING: Generated images have unusual value range")
    
    # Check 2: Discriminator scores are in valid range [0, 1]
    if np.all((d_fake_scores >= 0) & (d_fake_scores <= 1)) and np.all((d_real_scores >= 0) & (d_real_scores <= 1)):
        print("✓ All discriminator scores in valid range [0, 1]")
    else:
        print("✗ WARNING: Some discriminator scores outside [0, 1]!")
    
    # Check 3: GAN is learning
    if d_real_scores.mean() > 0.6:
        print(f"✓ Discriminator recognizes real images well: {d_real_scores.mean():.4f}")
    else:
        print(f"⚠ Discriminator struggling with real images: {d_real_scores.mean():.4f}")
    
    # Check 4: Generator is fooling discriminator reasonably
    if 0.2 < d_fake_scores.mean() < 0.8:
        print(f"✓ Generator creating believable fakes: {d_fake_scores.mean():.4f}")
    elif d_fake_scores.mean() < 0.2:
        print(f"⚠ Generator struggling: {d_fake_scores.mean():.4f}")
    else:
        print(f"⚠ Possible mode collapse: {d_fake_scores.mean():.4f}")
    
    # Check 5: No NaN or Inf in generated images
    if np.all(np.isfinite(generated_samples)):
        print("✓ All generated values are finite")
    else:
        print("✗ WARNING: Some generated values are NaN or Inf!")
    
    # Check 6: Generated images have reasonable variance
    gen_std = generated_samples.std()
    if gen_std > 0.1:
        print(f"✓ Generated images have good variance: {gen_std:.4f}")
    else:
        print(f"⚠ WARNING: Low variance: {gen_std:.4f} (possible mode collapse)")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all(np.isfinite(generated_samples)) and
        np.all((d_fake_scores >= 0) & (d_fake_scores <= 1)) and
        d_real_scores.mean() > 0.5 and
        gen_std > 0.1
    )
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nGenerated images saved to: gan_images_tensorflow/")

if __name__ == "__main__":
    train()
