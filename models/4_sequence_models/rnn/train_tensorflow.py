import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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
    
    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    predictions_probs = model.predict(x_test, verbose=0)
    predictions = np.argmax(predictions_probs, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=[str(i) for i in range(10)]))
    
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix shape: {cm.shape}")
    print("Diagonal (correct predictions per class):")
    print(np.diag(cm))
    
    print("\n--- Sanity Checks ---")
    if np.all((predictions >= 0) & (predictions < 10)):
        print("✓ All predictions in valid range [0-9]")
    else:
        print("✗ WARNING: Some predictions outside valid range!")
    
    if accuracy > 0.95:
        print(f"✓ Excellent accuracy: {accuracy:.4f}")
    elif accuracy > 0.90:
        print(f"✓ Good accuracy: {accuracy:.4f}")
    elif accuracy > 0.85:
        print(f"⚠ Moderate accuracy: {accuracy:.4f}")
    else:
        print(f"✗ WARNING: Poor accuracy: {accuracy:.4f}")
    
    unique_preds = np.unique(predictions)
    if len(unique_preds) == 10:
        print("✓ Model predicts all 10 classes")
    else:
        print(f"⚠ WARNING: Only predicts {len(unique_preds)} classes")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = np.all((predictions >= 0) & (predictions < 10)) and accuracy > 0.80 and len(unique_preds) >= 8
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
