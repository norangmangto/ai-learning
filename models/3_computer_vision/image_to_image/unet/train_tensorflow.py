import tensorflow as tf
from tensorflow.keras import layers, models
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def unet_model():
    inputs = layers.Input(shape=(32, 32, 1))

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(b)

    # Decoder
    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u2 = layers.concatenate([u2, c2])
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c1])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c4) # 3 channels RGB, 0-1 range

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def train():
    print("Training U-Net (Colorization) with TensorFlow...")
    os.makedirs("unet_images_tensorflow", exist_ok=True)

    # Load CIFAR-10
    (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images[:2000].astype('float32') / 255.0 # Subset for speed

    # Create Grayscale
    # tf.image.rgb_to_grayscale retains 3D shape (32,32,1)
    train_gray = tf.image.rgb_to_grayscale(train_images)

    model = unet_model()
    model.compile(optimizer='adam', loss='mse')

    model.fit(train_gray, train_images, epochs=2, batch_size=32, verbose=1)

     # Visualize
    # ...rest of code...
