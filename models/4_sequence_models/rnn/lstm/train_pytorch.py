"""
Long Short-Term Memory (LSTM) Network with PyTorch

LSTM is a type of RNN that can learn long-term dependencies:
- Cell state maintains long-term memory
- Gates (input, forget, output) control information flow
- Solves vanishing gradient problem
- Widely used for sequence modeling

This implementation includes:
- Vanilla LSTM for sequence classification
- Many-to-one and many-to-many architectures
- Bidirectional LSTM
- Stacked LSTM layers
- Comprehensive training and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class LSTMClassifier(nn.Module):
    """
    LSTM for sequence classification (many-to-one).

    Architecture:
    - Embedding layer (optional)
    - LSTM layers
    - Fully connected output layer
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Take the last time step output
        out = out[:, -1, :]

        # Apply dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)

        return out


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for sequence classification.

    Processes sequences in both forward and backward directions,
    capturing context from both past and future.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(BiLSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional
        )

        # *2 because bidirectional concatenates forward and backward
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Initialize states (2 * num_layers for bidirectional)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Take the last time step output
        out = out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)

        return out


class StackedLSTM(nn.Module):
    """
    Stacked LSTM with multiple layers.

    Deeper networks can learn more complex patterns.
    """
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super(StackedLSTM, self).__init__()

        self.lstm_layers = nn.ModuleList()

        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=1,
            batch_first=True
        ))

        # Additional LSTM layers
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(
                input_size=hidden_sizes[i-1],
                hidden_size=hidden_sizes[i],
                num_layers=1,
                batch_first=True
            ))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        # Pass through each LSTM layer
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)

        # Take last time step
        out = x[:, -1, :]
        out = self.fc(out)

        return out


def generate_synthetic_sequences(n_samples=1000, seq_length=50, n_features=10, n_classes=3):
    """
    Generate synthetic sequence data for classification.

    Each class has different patterns:
    - Class 0: Increasing trend
    - Class 1: Decreasing trend
    - Class 2: Oscillating pattern
    """
    print(f"Generating {n_samples} sequences (length={seq_length}, features={n_features})...")

    X = []
    y = []

    np.random.seed(42)

    for i in range(n_samples):
        label = i % n_classes

        if label == 0:
            # Increasing trend
            base = np.linspace(0, 1, seq_length)
            sequence = np.column_stack([base + np.random.randn(seq_length) * 0.1
                                       for _ in range(n_features)])
        elif label == 1:
            # Decreasing trend
            base = np.linspace(1, 0, seq_length)
            sequence = np.column_stack([base + np.random.randn(seq_length) * 0.1
                                       for _ in range(n_features)])
        else:
            # Oscillating
            t = np.linspace(0, 4*np.pi, seq_length)
            base = np.sin(t)
            sequence = np.column_stack([base + np.random.randn(seq_length) * 0.1
                                       for _ in range(n_features)])

        X.append(sequence)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequences."""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_lstm(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train LSTM model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_lstm_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    return history


def evaluate_model(model, test_loader, class_names=['Increasing', 'Decreasing', 'Oscillating']):
    """Evaluate model and show detailed metrics."""
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_architectures(X_train, y_train, X_val, y_val, input_size, num_classes):
    """Compare different LSTM architectures."""
    print("\n" + "="*70)
    print("Comparing LSTM Architectures")
    print("="*70)

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    architectures = {
        'LSTM-1Layer': LSTMClassifier(input_size, 64, 1, num_classes, dropout=0.2),
        'LSTM-2Layers': LSTMClassifier(input_size, 64, 2, num_classes, dropout=0.2),
        'BiLSTM': BiLSTMClassifier(input_size, 64, 1, num_classes, dropout=0.2),
        'StackedLSTM': StackedLSTM(input_size, [128, 64, 32], num_classes, dropout=0.3)
    }

    results = {}

    for name, model in architectures.items():
        print(f"\n{name}:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

        history = train_lstm(model, train_loader, val_loader, epochs=30, lr=0.001)
        results[name] = history

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for name, history in results.items():
        axes[0].plot(history['train_loss'], label=f'{name} Train', alpha=0.7)
        axes[0].plot(history['val_loss'], '--', label=f'{name} Val', alpha=0.7)
        axes[1].plot(history['val_acc'], label=name, linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training/Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lstm_architectures_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Best model summary
    best_arch = max(results.items(), key=lambda x: max(x[1]['val_acc']))
    print(f"\nBest Architecture: {best_arch[0]} (Val Acc: {max(best_arch[1]['val_acc']):.2f}%)")


def visualize_predictions(model, sequences, labels, class_names, n_samples=6):
    """Visualize sequence predictions."""
    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(min(n_samples, len(sequences))):
            seq_tensor = torch.FloatTensor(sequences[i:i+1]).to(device)
            output = model(seq_tensor)
            _, predicted = output.max(1)

            # Plot first feature of sequence
            axes[i].plot(sequences[i, :, 0], linewidth=2)
            axes[i].set_title(f'True: {class_names[labels[i]]}\n'
                            f'Pred: {class_names[predicted.item()]}',
                            color='green' if predicted.item() == labels[i] else 'red')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value (Feature 0)')
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lstm_sequence_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("LSTM for Sequence Classification")
    print("="*70)

    # Generate data
    print("\n1. Generating synthetic sequence data...")
    X, y = generate_synthetic_sequences(n_samples=1200, seq_length=50, n_features=10, n_classes=3)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    input_size = X.shape[2]
    num_classes = len(np.unique(y))
    class_names = ['Increasing', 'Decreasing', 'Oscillating']

    # Compare architectures
    print("\n2. Comparing LSTM architectures...")
    compare_architectures(X_train, y_train, X_val, y_val, input_size, num_classes)

    # Train best model
    print("\n3. Training BiLSTM (typically best for classification)...")
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BiLSTMClassifier(input_size, 128, 2, num_classes, dropout=0.3)
    train_lstm(model, train_loader, val_loader, epochs=50, lr=0.001)

    # Evaluate
    print("\n4. Evaluating on test set...")
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    evaluate_model(model, test_loader, class_names)

    # Visualize
    print("\n5. Visualizing predictions...")
    visualize_predictions(model, X_test, y_test, class_names, n_samples=6)

    print("\n" + "="*70)
    print("LSTM Training Complete!")
    print("="*70)
    print("\nKey Features:")
    print("✓ Cell state maintains long-term dependencies")
    print("✓ Gates control information flow")
    print("✓ Bidirectional captures past and future context")
    print("✓ Stacked layers learn hierarchical features")
    print("\nBest for: Time series, NLP, any sequential data")


if __name__ == "__main__":
    main()
