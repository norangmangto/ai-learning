import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def train():
    print("Training Transformer with TensorFlow (Dummy Text Classification)...")

    vocab_size = 1000
    maxlen = 20
    embed_dim = 64
    num_heads = 4
    ff_dim = 64

    # Dummy Data
    x_train = np.random.randint(0, vocab_size, size=(100, maxlen))
    y_train = np.random.randint(0, 2, size=(100,))

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    x = embedding_layer(inputs)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)
    print("TensorFlow Transformer Training Complete.")

if __name__ == "__main__":
    train()
