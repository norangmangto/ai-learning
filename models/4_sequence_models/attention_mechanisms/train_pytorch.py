"""
Attention Mechanisms for Sequence Models with PyTorch

Attention allows models to focus on relevant parts of the input:
- Additive (Bahdanau) Attention
- Multiplicative (Luong) Attention
- Self-Attention
- Multi-Head Attention
- Applications in sequence-to-sequence tasks

This implementation includes:
- Various attention mechanisms
- Encoder-Decoder with attention
- Visualization of attention weights
- Comparison of attention types
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class AdditiveAttention(nn.Module):
    """
    Bahdanau (Additive) Attention.

    score(h_t, h_s) = v^T * tanh(W_1 * h_t + W_2 * h_s)
    """
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()

        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, values, mask=None):
        """
        Args:
            query: (batch, hidden_size) - decoder hidden state
            keys: (batch, seq_len, hidden_size) - encoder outputs
            values: (batch, seq_len, hidden_size) - encoder outputs
            mask: (batch, seq_len) - padding mask
        """
        # Expand query to match keys dimensions
        query_expanded = query.unsqueeze(1)  # (batch, 1, hidden_size)

        # Calculate attention scores
        scores = self.v(torch.tanh(
            self.W1(query_expanded) + self.W2(keys)
        ))  # (batch, seq_len, 1)

        scores = scores.squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Calculate attention weights
        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Calculate context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            values  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)

        return context, attention_weights


class MultiplicativeAttention(nn.Module):
    """
    Luong (Multiplicative) Attention.

    score(h_t, h_s) = h_t^T * W * h_s
    """
    def __init__(self, hidden_size):
        super(MultiplicativeAttention, self).__init__()

        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query, keys, values, mask=None):
        """Similar to additive attention."""
        # Transform query
        query_transformed = self.W(query).unsqueeze(1)  # (batch, 1, hidden_size)

        # Calculate scores
        scores = torch.bmm(
            query_transformed,  # (batch, 1, hidden_size)
            keys.transpose(1, 2)  # (batch, hidden_size, seq_len)
        ).squeeze(1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=1)

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            values
        ).squeeze(1)

        return context, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (used in Transformers).

    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """
    def __init__(self, hidden_size):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = np.sqrt(hidden_size)

    def forward(self, query, keys, values, mask=None):
        # Calculate scores
        scores = torch.bmm(
            query.unsqueeze(1),  # (batch, 1, hidden_size)
            keys.transpose(1, 2)  # (batch, hidden_size, seq_len)
        ).squeeze(1) / self.scale  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=1)

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            values
        ).squeeze(1)

        return context, attention_weights


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism.

    Allows the model to attend to different positions in the same sequence.
    """
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = np.sqrt(hidden_size)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.size()

        # Generate Q, K, V
        Q = self.query(x)  # (batch, seq_len, hidden_size)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        attended = torch.bmm(attention_weights, V)  # (batch, seq_len, hidden_size)

        return attended, attention_weights


class EncoderRNN(nn.Module):
    """Encoder with LSTM."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class AttentionDecoder(nn.Module):
    """Decoder with attention mechanism."""
    def __init__(self, hidden_size, output_size, attention_type='additive'):
        super(AttentionDecoder, self).__init__()

        self.hidden_size = hidden_size

        # Attention mechanism
        if attention_type == 'additive':
            self.attention = AdditiveAttention(hidden_size * 2)
        elif attention_type == 'multiplicative':
            self.attention = MultiplicativeAttention(hidden_size * 2)
        elif attention_type == 'scaled_dot':
            self.attention = ScaledDotProductAttention(hidden_size * 2)

        # Decoder LSTM
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_size * 3, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, encoder_outputs, hidden, cell):
        """
        Args:
            encoder_outputs: (batch, seq_len, hidden_size * 2)
            hidden: (1, batch, hidden_size)
            cell: (1, batch, hidden_size)
        """
        # Get context using attention
        query = hidden.squeeze(0)  # (batch, hidden_size)
        context, attention_weights = self.attention(
            query, encoder_outputs, encoder_outputs
        )

        # Decoder step
        lstm_input = context.unsqueeze(1)  # (batch, 1, hidden_size * 2)
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # Combine LSTM output with context
        combined = torch.cat([lstm_out.squeeze(1), context], dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)

        return output, hidden, cell, attention_weights


class Seq2SeqWithAttention(nn.Module):
    """Sequence-to-Sequence model with attention."""
    def __init__(self, input_size, hidden_size, output_size, attention_type='additive'):
        super(Seq2SeqWithAttention, self).__init__()

        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttentionDecoder(hidden_size, output_size, attention_type)

    def forward(self, x):
        # Encode
        encoder_outputs, hidden, cell = self.encoder(x)

        # Use last hidden state from encoder
        # Bidirectional: combine forward and backward
        hidden = hidden[-2:].mean(dim=0, keepdim=True)
        cell = cell[-2:].mean(dim=0, keepdim=True)

        # Decode
        output, _, _, attention_weights = self.decoder(encoder_outputs, hidden, cell)

        return output, attention_weights


def generate_synthetic_sequences(n_samples=1000, seq_length=30, n_features=10, n_classes=3):
    """Generate synthetic sequences with important regions."""
    print(f"Generating {n_samples} sequences...")

    X = []
    y = []

    np.random.seed(42)

    for i in range(n_samples):
        label = i % n_classes

        # Create sequence with noise
        sequence = np.random.randn(seq_length, n_features) * 0.1

        # Add important pattern at specific location
        if label == 0:
            # Important pattern at beginning
            sequence[0:5] = np.linspace(1, 2, 5)[:, None]
        elif label == 1:
            # Important pattern in middle
            mid = seq_length // 2
            sequence[mid-2:mid+3] = np.sin(np.linspace(0, np.pi, 5))[:, None]
        else:
            # Important pattern at end
            sequence[-5:] = np.linspace(-1, -2, 5)[:, None]

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


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train attention-based model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs, _ = model(sequences)
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
                outputs, _ = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    return history


def visualize_attention_heatmap(model, sequences, labels, class_names, n_samples=4):
    """Visualize attention weights as heatmaps."""
    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))

    with torch.no_grad():
        for i in range(n_samples):
            seq_tensor = torch.FloatTensor(sequences[i:i+1]).to(device)
            output, attention_weights = model(seq_tensor)
            _, predicted = output.max(1)

            attention = attention_weights.cpu().numpy()[0]

            # Plot sequence (first feature)
            axes[i, 0].plot(sequences[i, :, 0], linewidth=2, color='steelblue')
            axes[i, 0].set_title(f'Input Sequence\nTrue: {class_names[labels[i]]}, '
                                f'Pred: {class_names[predicted.item()]}')
            axes[i, 0].set_xlabel('Time Step')
            axes[i, 0].set_ylabel('Value')
            axes[i, 0].grid(True, alpha=0.3)

            # Plot attention weights
            axes[i, 1].bar(range(len(attention)), attention, color='coral')
            axes[i, 1].set_title('Attention Weights')
            axes[i, 1].set_xlabel('Time Step')
            axes[i, 1].set_ylabel('Attention Weight')
            axes[i, 1].grid(True, alpha=0.3, axis='y')

            # Highlight important regions
            if labels[i] == 0:
                axes[i, 0].axvspan(0, 5, alpha=0.2, color='red', label='Important Region')
            elif labels[i] == 1:
                mid = len(sequences[i]) // 2
                axes[i, 0].axvspan(mid-2, mid+3, alpha=0.2, color='red', label='Important Region')
            else:
                axes[i, 0].axvspan(len(sequences[i])-5, len(sequences[i]), alpha=0.2,
                                  color='red', label='Important Region')
            axes[i, 0].legend()

    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_attention_types(X_train, y_train, X_val, y_val, input_size, num_classes):
    """Compare different attention mechanisms."""
    print("\n" + "="*70)
    print("Comparing Attention Mechanisms")
    print("="*70)

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    attention_types = {
        'Additive (Bahdanau)': 'additive',
        'Multiplicative (Luong)': 'multiplicative',
        'Scaled Dot-Product': 'scaled_dot',
    }

    results = {}

    for name, att_type in attention_types.items():
        print(f"\n{name}:")
        model = Seq2SeqWithAttention(input_size, 64, num_classes, attention_type=att_type)
        history = train_model(model, train_loader, val_loader, epochs=30, lr=0.001)
        results[name] = history

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for name, history in results.items():
        axes[0].plot(history['val_loss'], label=name, linewidth=2)
        axes[1].plot(history['val_acc'], label=name, linewidth=2)

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
    plt.savefig('attention_types_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("Attention Mechanisms for Sequence Models")
    print("="*70)

    # Generate data
    print("\n1. Generating sequences with important regions...")
    X, y = generate_synthetic_sequences(n_samples=1200, seq_length=30, n_features=10, n_classes=3)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    input_size = X.shape[2]
    num_classes = len(np.unique(y))
    class_names = ['Pattern at Start', 'Pattern in Middle', 'Pattern at End']

    # Compare attention types
    print("\n2. Comparing attention mechanisms...")
    compare_attention_types(X_train, y_train, X_val, y_val, input_size, num_classes)

    # Train model with best attention
    print("\n3. Training model with additive attention...")
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Seq2SeqWithAttention(input_size, 128, num_classes, attention_type='additive')
    train_model(model, train_loader, val_loader, epochs=50, lr=0.001)

    # Visualize attention
    print("\n4. Visualizing attention weights...")
    visualize_attention_heatmap(model, X_test, y_test, class_names, n_samples=6)

    print("\n" + "="*70)
    print("Attention Mechanisms Complete!")
    print("="*70)
    print("\nKey Concepts:")
    print("✓ Attention focuses on relevant input parts")
    print("✓ Different mechanisms: additive, multiplicative, scaled dot-product")
    print("✓ Self-attention for intra-sequence dependencies")
    print("✓ Foundation for Transformer architectures")
    print("\nBest for: Machine translation, text summarization, any seq2seq task")


if __name__ == "__main__":
    main()
