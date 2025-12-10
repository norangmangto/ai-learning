from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

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

    history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)
    print("TensorFlow Transformer Training Complete.")
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    
    # Generate test data
    x_test = np.random.randint(0, vocab_size, size=(50, maxlen))
    y_test = np.random.randint(0, 2, size=(50,))
    
    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions_probs = model.predict(x_test, verbose=0)
    predictions = np.argmax(predictions_probs, axis=1)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Training Loss: {history.history['loss'][-1]:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Class 0", "Class 1"]))
    
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:\n{cm}")
    
    print("\n--- Sanity Checks ---")
    
    # Check 1: Predictions are in valid range
    if np.all((predictions >= 0) & (predictions < 2)):
        print("✓ All predictions in valid class range [0-1]")
    else:
        print("✗ WARNING: Some predictions outside valid range!")
    
    # Check 2: Training loss convergence
    final_train_loss = history.history['loss'][-1]
    if final_train_loss < 0.5:
        print(f"✓ Training converged well - Final loss: {final_train_loss:.4f}")
    elif final_train_loss < 1.0:
        print(f"⚠ Moderate convergence - Final loss: {final_train_loss:.4f}")
    else:
        print(f"✗ WARNING: Poor convergence - Final loss: {final_train_loss:.4f}")
    
    # Check 3: Test vs train loss comparison
    if test_loss < final_train_loss * 2:
        print(f"✓ No significant overfitting detected")
    else:
        print(f"⚠ Possible overfitting: Test loss ({test_loss:.4f}) >> Train loss ({final_train_loss:.4f})")
    
    # Check 4: Both classes are predicted
    unique_preds = np.unique(predictions)
    if len(unique_preds) == 2:
        print(f"✓ Model predicts both classes")
    else:
        print(f"⚠ WARNING: Model only predicts {len(unique_preds)} class")
    
    # Check 5: Model is better than random
    random_baseline = 0.5
    if test_accuracy > random_baseline * 1.2:
        print(f"✓ Model performs better than random ({test_accuracy:.4f} vs {random_baseline:.4f})")
    elif test_accuracy > random_baseline:
        print(f"⚠ Model slightly better than random ({test_accuracy:.4f} vs {random_baseline:.4f})")
    else:
        print(f"✗ WARNING: Model not better than random guessing")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all((predictions >= 0) & (predictions < 2)) and
        test_accuracy > random_baseline and
        final_train_loss < 2.0
    )
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nNote: This is a dummy dataset. For real applications, use proper datasets and longer training.")

if __name__ == "__main__":
    train()
