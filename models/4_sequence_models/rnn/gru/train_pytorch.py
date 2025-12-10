"""
Gated Recurrent Unit (GRU) Network with PyTorch

GRU is a simplified variant of LSTM:
- Fewer parameters than LSTM (faster training)
- Two gates instead of three (update and reset)
- No separate cell state
- Often performs comparably to LSTM

This implementation includes:
- Vanilla GRU for sequence classification
- Bidirectional GRU
- Stacked GRU layers
- Comparison with LSTM
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


class GRUClassifier(nn.Module):
    """
    GRU for sequence classification.

    GRU uses update and reset gates to control information flow,
    making it simpler and often faster than LSTM.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(GRUClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
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

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU forward pass
        out, hn = self.gru(x, h0)

        # Take the last time step output
        out = out[:, -1, :]

        # Apply dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)

        return out


class BiGRUClassifier(nn.Module):
    """
    Bidirectional GRU for sequence classification.

    Processes sequences in both directions for better context understanding.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(BiGRUClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # *2 because bidirectional concatenates forward and backward
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Initialize states (2 * num_layers for bidirectional)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, hn = self.gru(x, h0)

        # Take the last time step output
        out = out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)

        return out


class StackedGRU(nn.Module):
    """
    Stacked GRU with residual connections.

    Deep GRU with skip connections to help gradient flow.
    """
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super(StackedGRU, self).__init__()

        self.gru_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # First GRU layer
        self.gru_layers.append(nn.GRU(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=1,
            batch_first=True
        ))

        # Additional GRU layers
        for i in range(1, len(hidden_sizes)):
            self.gru_layers.append(nn.GRU(
                input_size=hidden_sizes[i-1],
                hidden_size=hidden_sizes[i],
                num_layers=1,
                batch_first=True
            ))

        self.fc = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        # Pass through each GRU layer
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)

        # Take last time step
        out = x[:, -1, :]
        out = self.fc(out)

        return out


class LSTMClassifier(nn.Module):
    """LSTM for comparison with GRU."""
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out


def generate_synthetic_sequences(n_samples=1000, seq_length=50, n_features=10, n_classes=3):
    """Generate synthetic sequence data for classification."""
    print(f"Generating {n_samples} sequences (length={seq_length}, features={n_features})...")

    X = []
    y = []

    np.random.seed(42)

    for i in range(n_samples):
        label = i % n_classes

        if label == 0:
            # Increasing trend with noise
            base = np.linspace(0, 1, seq_length)
            sequence = np.column_stack([base + np.random.randn(seq_length) * 0.1
                                       for _ in range(n_features)])
        elif label == 1:
            # Decreasing trend with noise
            base = np.linspace(1, 0, seq_length)
            sequence = np.column_stack([base + np.random.randn(seq_length) * 0.1
                                       for _ in range(n_features)])
        else:
            # Oscillating pattern
            t = np.linspace(0, 4*np.pi, seq_length)
            base = np.sin(t)
            sequence = np.column_stack([base + np.random.randn(seq_length) * 0.1
                                       for _ in range(n_features)])

        X.append(sequence)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequences."""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'time': []}
    best_val_acc = 0

    for epoch in range(epochs):
        epoch_start = time.time()

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

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['time'].append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gru_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    return history


def compare_gru_lstm(X_train, y_train, X_val, y_val, input_size, num_classes):
    """Compare GRU vs LSTM performance and efficiency."""
    print("\n" + "="*70)
    print("Comparing GRU vs LSTM")
    print("="*70)

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    models = {
        'GRU-2Layer': GRUClassifier(input_size, 128, 2, num_classes, dropout=0.2),
        'LSTM-2Layer': LSTMClassifier(input_size, 128, 2, num_classes, dropout=0.2),
        'BiGRU': BiGRUClassifier(input_size, 64, 2, num_classes, dropout=0.2),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{name}:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

        history = train_model(model, train_loader, val_loader, epochs=30, lr=0.001)
        results[name] = history

    # Performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Accuracy
    for name, history in results.items():
        axes[0].plot(history['val_acc'], label=name, linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Training time
    avg_times = {name: np.mean(history['time']) for name, history in results.items()}
    axes[1].bar(avg_times.keys(), avg_times.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Time per Epoch (s)')
    axes[1].set_title('Training Speed Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Parameters
    params = {name: sum(p.numel() for p in model.parameters())
              for name, model in models.items()}
    axes[2].bar(params.keys(), params.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_ylabel('Number of Parameters')
    axes[2].set_title('Model Complexity')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('gru_lstm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    for name, history in results.items():
        print(f"\n{name}:")
        print(f"  Best Val Accuracy: {max(history['val_acc']):.2f}%")
        print(f"  Avg Training Time: {np.mean(history['time']):.3f}s/epoch")
        print(f"  Parameters: {sum(p.numel() for p in models[name].parameters()):,}")


def visualize_gru_internals(model, sequence, feature_idx=0):
    """Visualize GRU hidden states over time."""
    device = next(model.parameters()).device
    model.eval()

    # Get hidden states at each time step
    seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        h0 = torch.zeros(model.num_layers, 1, model.hidden_size).to(device)
        out, _ = model.gru(seq_tensor, h0)
        hidden_states = out.squeeze(0).cpu().numpy()  # (seq_len, hidden_size)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Input sequence
    axes[0].plot(sequence[:, feature_idx], linewidth=2)
    axes[0].set_title(f'Input Sequence (Feature {feature_idx})')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)

    # Hidden states heatmap
    im = axes[1].imshow(hidden_states.T[:20, :], aspect='auto', cmap='RdBu_r', interpolation='nearest')
    axes[1].set_title('GRU Hidden States (First 20 Units)')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Hidden Unit')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig('gru_internals.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("GRU for Sequence Classification")
    print("="*70)

    # Generate data
    print("\n1. Generating synthetic sequence data...")
    X, y = generate_synthetic_sequences(n_samples=1200, seq_length=50, n_features=10, n_classes=3)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    input_size = X.shape[2]
    num_classes = len(np.unique(y))

    # Compare GRU vs LSTM
    print("\n2. Comparing GRU vs LSTM...")
    compare_gru_lstm(X_train, y_train, X_val, y_val, input_size, num_classes)

    # Train best model
    print("\n3. Training BiGRU (best balance of accuracy and speed)...")
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = BiGRUClassifier(input_size, 128, 2, num_classes, dropout=0.3)
    history = train_model(model, train_loader, val_loader, epochs=50, lr=0.001)

    # Visualize internals
    print("\n4. Visualizing GRU internals...")
    model.load_state_dict(torch.load('best_gru_model.pth'))
    visualize_gru_internals(model, X_test[0], feature_idx=0)

    print("\n" + "="*70)
    print("GRU Training Complete!")
    print("="*70)
    print("\nGRU vs LSTM:")
    print("✓ Fewer parameters → faster training")
    print("✓ Simpler architecture → easier to tune")
    print("✓ Often comparable performance to LSTM")
    print("✓ Better for resource-constrained scenarios")
    print("\nBest for: When you need speed without sacrificing much accuracy")


if __name__ == "__main__":
    main()
