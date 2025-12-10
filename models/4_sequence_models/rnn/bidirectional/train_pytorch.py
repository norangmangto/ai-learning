"""
Bidirectional RNN/LSTM/GRU with PyTorch

Bidirectional RNNs process sequences in both directions:
- Forward pass: left to right (past to future)
- Backward pass: right to left (future to past)
- Concatenate outputs from both directions
- Captures context from entire sequence

This implementation includes:
- Bidirectional LSTM
- Bidirectional GRU
- Comparison with unidirectional variants
- Applications for sequence labeling and classification
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.model_selection import train_test_split


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence classification.

    Processes sequences in both forward and backward directions,
    then combines the representations.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout=0.2, pooling='last'):
        super(BidirectionalLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling = pooling  # 'last', 'mean', 'max', 'attention'

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Output size is hidden_size * 2 due to bidirectionality
        if pooling == 'attention':
            self.attention = nn.Linear(hidden_size * 2, 1)

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # x shape: (batch, seq_len, input_size)

        # Initialize states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out shape: (batch, seq_len, hidden_size * 2)

        # Pooling strategy
        if self.pooling == 'last':
            # Take last time step
            pooled = out[:, -1, :]
            attention_weights = None

        elif self.pooling == 'mean':
            # Average over time
            pooled = torch.mean(out, dim=1)
            attention_weights = None

        elif self.pooling == 'max':
            # Max over time
            pooled, _ = torch.max(out, dim=1)
            attention_weights = None

        elif self.pooling == 'attention':
            # Attention mechanism
            # Compute attention scores
            attention_scores = self.attention(out)  # (batch, seq_len, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)

            # Weighted sum
            pooled = torch.sum(out * attention_weights, dim=1)

        # Final classification
        pooled = self.dropout(pooled)
        output = self.fc(pooled)

        if return_attention and attention_weights is not None:
            return output, attention_weights
        return output


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU for sequence classification."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout=0.2, pooling='last'):
        super(BidirectionalGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling = pooling

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        if pooling == 'attention':
            self.attention = nn.Linear(hidden_size * 2, 1)

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, hn = self.gru(x, h0)

        if self.pooling == 'last':
            pooled = out[:, -1, :]
        elif self.pooling == 'mean':
            pooled = torch.mean(out, dim=1)
        elif self.pooling == 'max':
            pooled, _ = torch.max(out, dim=1)
        elif self.pooling == 'attention':
            attention_scores = self.attention(out)
            attention_weights = torch.softmax(attention_scores, dim=1)
            pooled = torch.sum(out * attention_weights, dim=1)

        pooled = self.dropout(pooled)
        output = self.fc(pooled)

        return output


class UnidirectionalLSTM(nn.Module):
    """Standard unidirectional LSTM for comparison."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(UnidirectionalLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
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
    """Generate synthetic sequence data."""
    print(f"Generating {n_samples} sequences (length={seq_length}, features={n_features})...")

    X = []
    y = []

    np.random.seed(42)

    for i in range(n_samples):
        label = i % n_classes

        if label == 0:
            # Pattern at the beginning
            sequence = np.random.randn(seq_length, n_features) * 0.1
            sequence[:10] = np.linspace(0, 2, 10)[:, None]

        elif label == 1:
            # Pattern at the end
            sequence = np.random.randn(seq_length, n_features) * 0.1
            sequence[-10:] = np.linspace(2, 0, 10)[:, None]

        else:
            # Pattern in the middle
            sequence = np.random.randn(seq_length, n_features) * 0.1
            mid = seq_length // 2
            sequence[mid-5:mid+5] = np.sin(np.linspace(0, 2*np.pi, 10))[:, None]

        X.append(sequence)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, model_name='model'):
    """Train model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining {model_name} on {device}")

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
            torch.save(model.state_dict(), f'best_{model_name}_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    return history


def compare_directions(X_train, y_train, X_val, y_val, input_size, num_classes):
    """Compare unidirectional vs bidirectional models."""
    print("\n" + "="*70)
    print("Comparing Unidirectional vs Bidirectional Models")
    print("="*70)

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    models = {
        'Unidirectional LSTM': UnidirectionalLSTM(input_size, 128, 2, num_classes, dropout=0.2),
        'Bidirectional LSTM': BidirectionalLSTM(input_size, 64, 2, num_classes, dropout=0.2, pooling='last'),
        'BiLSTM + Mean Pooling': BidirectionalLSTM(input_size, 64, 2, num_classes, dropout=0.2, pooling='mean'),
        'BiLSTM + Attention': BidirectionalLSTM(input_size, 64, 2, num_classes, dropout=0.2, pooling='attention'),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{name}:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

        history = train_model(model, train_loader, val_loader, epochs=30, lr=0.001,
                            model_name=name.replace(' ', '_'))
        results[name] = history

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (name, history), color in zip(results.items(), colors):
        axes[0].plot(history['train_loss'], alpha=0.5, color=color)
        axes[0].plot(history['val_loss'], '--', label=name, color=color, linewidth=2)
        axes[1].plot(history['val_acc'], label=name, color=color, linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bidirectional_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    for name, history in results.items():
        print(f"{name}: Best Val Acc = {max(history['val_acc']):.2f}%")


def visualize_attention(model, sequences, labels, class_names, n_samples=3):
    """Visualize attention weights for sequences."""
    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))

    with torch.no_grad():
        for i in range(n_samples):
            seq_tensor = torch.FloatTensor(sequences[i:i+1]).to(device)
            output, attention_weights = model(seq_tensor, return_attention=True)
            _, predicted = output.max(1)

            attention = attention_weights.squeeze().cpu().numpy()

            # Plot sequence
            axes[i, 0].plot(sequences[i, :, 0], linewidth=2)
            axes[i, 0].set_title(f'Sequence (True: {class_names[labels[i]]}, '
                                f'Pred: {class_names[predicted.item()]})')
            axes[i, 0].set_xlabel('Time Step')
            axes[i, 0].set_ylabel('Value')
            axes[i, 0].grid(True, alpha=0.3)

            # Plot attention weights
            axes[i, 1].bar(range(len(attention)), attention, color='steelblue')
            axes[i, 1].set_title('Attention Weights')
            axes[i, 1].set_xlabel('Time Step')
            axes[i, 1].set_ylabel('Attention Weight')
            axes[i, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('bidirectional_attention.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("Bidirectional RNN for Sequence Classification")
    print("="*70)

    # Generate data with patterns at different positions
    print("\n1. Generating sequences with patterns at different positions...")
    X, y = generate_synthetic_sequences(n_samples=1200, seq_length=50, n_features=10, n_classes=3)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    input_size = X.shape[2]
    num_classes = len(np.unique(y))
    class_names = ['Pattern at Start', 'Pattern at End', 'Pattern in Middle']

    # Compare directions and pooling strategies
    print("\n2. Comparing unidirectional vs bidirectional...")
    compare_directions(X_train, y_train, X_val, y_val, input_size, num_classes)

    # Train best model with attention
    print("\n3. Training BiLSTM with attention...")
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = BidirectionalLSTM(input_size, 128, 2, num_classes, dropout=0.3, pooling='attention')
    train_model(model, train_loader, val_loader, epochs=50, lr=0.001,
               model_name='bilstm_attention')

    # Visualize attention
    print("\n4. Visualizing attention weights...")
    model.load_state_dict(torch.load('best_bilstm_attention_model.pth'))
    visualize_attention(model, X_test, y_test, class_names, n_samples=3)

    print("\n" + "="*70)
    print("Bidirectional RNN Complete!")
    print("="*70)
    print("\nKey Advantages:")
    print("✓ Access to full sequence context (past + future)")
    print("✓ Better for tasks where future context matters")
    print("✓ Attention shows which time steps are important")
    print("✓ Significantly better for patterns anywhere in sequence")
    print("\nBest for: NER, POS tagging, sentiment analysis, any task needing full context")


if __name__ == "__main__":
    main()
