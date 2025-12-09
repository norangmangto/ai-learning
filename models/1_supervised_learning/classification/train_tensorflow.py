import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

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
    
    # 6. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")
    predictions_probs = model.predict(test_images, verbose=0)
    predictions = np.argmax(predictions_probs, axis=1)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))
    
    cm = confusion_matrix(test_labels, predictions)
    print(f"\nConfusion Matrix shape: {cm.shape}")
    print("Diagonal (correct predictions per class):")
    print(np.diag(cm))
    
    print("\n--- Sanity Checks ---")
    if np.all((predictions >= 0) & (predictions < 10)):
        print("✓ All predictions in valid range [0-9]")
    else:
        print("✗ WARNING: Some predictions outside valid range!")
    
    if acc > 0.88:
        print(f"✓ Excellent accuracy: {acc:.4f}")
    elif acc > 0.85:
        print(f"✓ Good accuracy: {acc:.4f}")
    elif acc > 0.80:
        print(f"⚠ Moderate accuracy: {acc:.4f}")
    else:
        print(f"✗ WARNING: Poor accuracy: {acc:.4f}")
    
    unique_preds = np.unique(predictions)
    if len(unique_preds) == 10:
        print("✓ Model predicts all 10 classes")
    else:
        print(f"⚠ WARNING: Only predicts {len(unique_preds)} classes")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = np.all((predictions >= 0) & (predictions < 10)) and acc > 0.75 and len(unique_preds) >= 8
    
    if validation_passed:
        print("✓ Model validation PASSED")
    else:
        print("✗ Model validation FAILED")
    
    print("\nDone.")

if __name__ == "__main__":
    train()
