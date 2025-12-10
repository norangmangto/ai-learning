"""
Encoder-Only Transformer (BERT-style) with PyTorch

Encoder-only transformers are used for understanding tasks:
- BERT (Bidirectional Encoder Representations from Transformers)
- RoBERTa, ALBERT, DistilBERT variants
- Used for: classification, NER, question answering
- Bidirectional self-attention for full context

This implementation includes:
- Multi-head self-attention
- Position-wise feedforward networks
- Positional encoding
- Layer normalization and residual connections
- Classification head
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class PositionalEncoding(nn.Module):
    """
    Positional encoding adds position information to embeddings.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Allows model to jointly attend to information from different
    representation subspaces at different positions.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)."""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # BERT uses GELU

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer with self-attention and feedforward."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Encoder-only Transformer (BERT-style).

    Used for sequence classification, token classification, etc.
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        num_classes=3,
    ):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.pooler = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len)

        # Embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        attention_weights = []
        for encoder_layer in self.encoder_layers:
            x, attn = encoder_layer(x, mask)
            attention_weights.append(attn)

        # Pooling (use [CLS] token - first token)
        pooled = self.pooler_activation(self.pooler(x[:, 0, :]))

        # Classification
        logits = self.classifier(pooled)

        return logits, attention_weights


def generate_synthetic_sequences(
    n_samples=1000, seq_length=32, vocab_size=100, n_classes=3
):
    """Generate synthetic sequence classification data."""
    print(
        f"Generating {n_samples} sequences (length={seq_length}, vocab={vocab_size})..."
    )

    np.random.seed(42)

    sequences = []
    labels = []

    for i in range(n_samples):
        label = i % n_classes

        # Generate sequence with class-specific patterns
        if label == 0:
            # Pattern: high values at start
            seq = np.random.randint(vocab_size // 2, vocab_size, size=10).tolist()
            seq += np.random.randint(1, vocab_size // 2, size=seq_length - 10).tolist()
        elif label == 1:
            # Pattern: high values at end
            seq = np.random.randint(1, vocab_size // 2, size=seq_length - 10).tolist()
            seq += np.random.randint(vocab_size // 2, vocab_size, size=10).tolist()
        else:
            # Pattern: alternating high/low
            seq = []
            for j in range(seq_length):
                if j % 2 == 0:
                    seq.append(np.random.randint(vocab_size // 2, vocab_size))
                else:
                    seq.append(np.random.randint(1, vocab_size // 2))

        sequences.append(seq)
        labels.append(label)

    return np.array(sequences, dtype=np.int64), np.array(labels, dtype=np.int64)


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_transformer(model, train_loader, val_loader, epochs=30, lr=0.0001):
    """Train transformer encoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < 100:
            return step / 100
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
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

            logits, _ = model(sequences)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                logits, _ = model(sequences)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_encoder_transformer.pth")

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            )

    return history


def visualize_attention(model, sequence, layer_idx=0, head_idx=0):
    """Visualize attention weights."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        seq_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)
        _, attention_weights = model(seq_tensor)

        # Get attention from specific layer and head
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title(f"Self-Attention Weights (Layer {layer_idx}, Head {head_idx})")
    plt.tight_layout()
    plt.savefig("encoder_attention_weights.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_training_curves(history):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(history["val_acc"], label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("encoder_training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Encoder-Only Transformer (BERT-style)")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic sequence data...")
    X, y = generate_synthetic_sequences(
        n_samples=2000, seq_length=32, vocab_size=100, n_classes=3
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create dataloaders
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create model
    print("\n2. Creating Transformer Encoder...")
    model = TransformerEncoder(
        vocab_size=100,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=64,
        dropout=0.1,
        num_classes=3,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print("\n3. Training model...")
    history = train_transformer(model, train_loader, val_loader, epochs=30, lr=0.0001)

    # Plot training curves
    print("\n4. Plotting training curves...")
    plot_training_curves(history)

    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    model.load_state_dict(torch.load("best_encoder_transformer.pth"))
    device = next(model.parameters()).device
    model.eval()

    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits, _ = model(sequences)
            _, predicted = logits.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100.0 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=["Class 0", "Class 1", "Class 2"]
        )
    )

    # Visualize attention
    print("\n6. Visualizing attention weights...")
    visualize_attention(model, X_test[0], layer_idx=0, head_idx=0)

    print("\n" + "=" * 70)
    print("Encoder-Only Transformer Complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("✓ Bidirectional self-attention (sees full context)")
    print("✓ Multi-head attention for different representation subspaces")
    print("✓ Position-wise feedforward networks")
    print("✓ Layer normalization and residual connections")
    print(
        "\nBest for: Classification, NER, question answering, masked language modeling"
    )


if __name__ == "__main__":
    main()
