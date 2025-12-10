import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, max_len=100):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, src):
        # src: (batch_size, seq_len) -> Transpose to (seq_len, batch_size)
        src = src.transpose(0, 1)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # Average pooling across sequence
        output = output.mean(dim=0)
        return self.fc(output)

def train():
    print("Training Transformer with PyTorch (Dummy Text Classification)...")

    # Hyperparameters
    VOCAB_SIZE = 1000
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    NUM_CLASSES = 2
    MAX_LEN = 20

    # Dummy Data: Random sequences of integers
    X_train = torch.randint(0, VOCAB_SIZE, (100, MAX_LEN))
    y_train = torch.randint(0, NUM_CLASSES, (100,))

    model = TransformerClassifier(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_CLASSES, MAX_LEN)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("PyTorch Transformer Training Complete.")

    # 5. QA Validation and Results Evaluation
    print("\n=== QA Validation ===")

    # Generate test data
    X_test = torch.randint(0, VOCAB_SIZE, (50, MAX_LEN))
    y_test = torch.randint(0, NUM_CLASSES, (50,))

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = criterion(test_output, y_test).item()
        _, predictions = torch.max(test_output, 1)

    y_pred = predictions.numpy()
    y_true = y_test.numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(NUM_CLASSES)]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    # Sanity checks
    print("\n--- Sanity Checks ---")

    # Check 1: Predictions are in valid range
    if np.all((y_pred >= 0) & (y_pred < NUM_CLASSES)):
        print(f"✓ All predictions are in valid class range [0-{NUM_CLASSES-1}]")
    else:
        print("✗ WARNING: Some predictions are outside valid range!")

    # Check 2: Training loss convergence
    if loss.item() < 0.5:
        print(f"✓ Training converged well - Final loss: {loss.item():.4f}")
    elif loss.item() < 1.0:
        print(f"⚠ Moderate convergence - Final loss: {loss.item():.4f}")
    else:
        print(f"✗ WARNING: Poor convergence - Final loss: {loss.item():.4f}")

    # Check 3: Test vs train loss comparison
    if test_loss < loss.item() * 2:
        print(f"✓ No significant overfitting detected")
    else:
        print(f"⚠ Possible overfitting: Test loss ({test_loss:.4f}) >> Train loss ({loss.item():.4f})")

    # Check 4: Both classes are predicted
    unique_preds = np.unique(y_pred)
    if len(unique_preds) == NUM_CLASSES:
        print(f"✓ Model predicts all {NUM_CLASSES} classes")
    else:
        print(f"⚠ WARNING: Model only predicts {len(unique_preds)} out of {NUM_CLASSES} classes")

    # Check 5: Model is better than random
    random_baseline = 1.0 / NUM_CLASSES
    if accuracy > random_baseline * 1.5:
        print(f"✓ Model performs significantly better than random ({accuracy:.4f} vs {random_baseline:.4f})")
    elif accuracy > random_baseline:
        print(f"⚠ Model slightly better than random ({accuracy:.4f} vs {random_baseline:.4f})")
    else:
        print(f"✗ WARNING: Model not better than random guessing")

    # Overall validation result
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        np.all((y_pred >= 0) & (y_pred < NUM_CLASSES)) and
        accuracy > random_baseline and
        loss.item() < 2.0
    )

    if validation_passed:
        print("✓ Model validation PASSED - Transformer is performing as expected")
    else:
        print("✗ Model validation FAILED - Please review model performance")

    print("\nNote: This is a dummy dataset. For real applications, use proper datasets and longer training.")

if __name__ == "__main__":
    train()
