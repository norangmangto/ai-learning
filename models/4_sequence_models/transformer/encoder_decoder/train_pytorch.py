"""
Encoder-Decoder Transformer (Original Architecture) with PyTorch

The original "Attention Is All You Need" transformer for sequence-to-sequence tasks:
- Machine translation
- Text summarization
- Speech recognition
- Code generation

This implementation includes:
- Multi-head self-attention in encoder
- Masked multi-head self-attention in decoder
- Cross-attention (encoder-decoder attention)
- Position-wise feedforward networks
- Positional encoding
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time


class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine functions."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.W_o(context)

        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attn_weights


class DecoderLayer(nn.Module):
    """Single decoder layer with cross-attention."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross-attention to encoder output
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x, self_attn_weights, cross_attn_weights


class Transformer(nn.Module):
    """
    Original Transformer architecture for sequence-to-sequence tasks.

    Encoder: processes source sequence
    Decoder: generates target sequence with attention to encoder output
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        # Decoder
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def create_causal_mask(self, seq_len, device):
        """Create lower triangular causal mask for decoder."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        # Embedding + positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x, _ = encoder_layer(x, src_mask)

        return self.encoder_norm(x)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence with attention to encoder output."""
        # Embedding + positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through decoder layers
        self_attentions = []
        cross_attentions = []

        for decoder_layer in self.decoder_layers:
            x, self_attn, cross_attn = decoder_layer(
                x, encoder_output, src_mask, tgt_mask
            )
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)

        x = self.decoder_norm(x)

        return x, self_attentions, cross_attentions

    def forward(self, src, tgt):
        """Forward pass through encoder-decoder."""
        # Create causal mask for decoder
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.create_causal_mask(tgt_seq_len, src.device)

        # Encode
        encoder_output = self.encode(src)

        # Decode
        decoder_output, self_attentions, cross_attentions = self.decode(
            tgt, encoder_output, tgt_mask=tgt_mask
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits, self_attentions, cross_attentions


def generate_seq2seq_data(
    n_pairs=1000, src_len=20, tgt_len=25, src_vocab=100, tgt_vocab=80
):
    """
    Generate synthetic sequence-to-sequence data.

    Task: reverse + shift sequence (simple translation-like task)
    """
    print(f"Generating {n_pairs} sequence pairs...")

    np.random.seed(42)

    src_sequences = []
    tgt_sequences = []

    for _ in range(n_pairs):
        # Source: random sequence
        src = np.random.randint(1, src_vocab, size=src_len)

        # Target: reversed + shifted sequence (with padding)
        tgt_data = src[::-1] % tgt_vocab

        # Add BOS token (0) and EOS token (tgt_vocab-1)
        tgt = np.concatenate([[0], tgt_data, [tgt_vocab - 1]])

        # Pad to fixed length
        if len(tgt) < tgt_len:
            tgt = np.concatenate([tgt, np.zeros(tgt_len - len(tgt))])
        else:
            tgt = tgt[:tgt_len]

        src_sequences.append(src)
        tgt_sequences.append(tgt)

    return np.array(src_sequences, dtype=np.int64), np.array(
        tgt_sequences, dtype=np.int64
    )


class Seq2SeqDataset(Dataset):
    """Dataset for sequence-to-sequence tasks."""

    def __init__(self, src_sequences, tgt_sequences):
        self.src = torch.LongTensor(src_sequences)
        self.tgt = torch.LongTensor(tgt_sequences)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # Input: tgt[:-1], Target: tgt[1:]
        return self.src[idx], self.tgt[idx, :-1], self.tgt[idx, 1:]


def train_transformer(model, train_loader, val_loader, epochs=30, lr=0.0003):
    """Train encoder-decoder transformer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Learning rate warmup
    def lr_lambda(step):
        d_model = model.d_model
        step = step + 1
        return (d_model**-0.5) * min(step**-0.5, step * (4000**-1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_val_loss = float("inf")

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for src, tgt_input, tgt_output in train_loader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            logits, _, _ = model(src, tgt_input)

            loss = criterion(logits.view(-1, model.tgt_vocab_size), tgt_output.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # Calculate accuracy (excluding padding)
            predictions = logits.argmax(dim=-1)
            mask = tgt_output != 0
            train_correct += ((predictions == tgt_output) & mask).sum().item()
            train_total += mask.sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for src, tgt_input, tgt_output in val_loader:
                src = src.to(device)
                tgt_input = tgt_input.to(device)
                tgt_output = tgt_output.to(device)

                logits, _, _ = model(src, tgt_input)
                loss = criterion(
                    logits.view(-1, model.tgt_vocab_size), tgt_output.view(-1)
                )

                val_loss += loss.item()

                predictions = logits.argmax(dim=-1)
                mask = tgt_output != 0
                val_correct += ((predictions == tgt_output) & mask).sum().item()
                val_total += mask.sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_transformer.pth")

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}"
            )

    return history


def visualize_cross_attention(model, src, tgt):
    """Visualize encoder-decoder cross-attention."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        src_tensor = torch.LongTensor(src).unsqueeze(0).to(device)
        tgt_tensor = torch.LongTensor(tgt).unsqueeze(0).to(device)

        _, _, cross_attentions = model(src_tensor, tgt_tensor)

        # Get cross-attention from last layer, first head
        cross_attn = cross_attentions[-1][0, 0].cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.imshow(cross_attn, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Source Position")
    plt.ylabel("Target Position")
    plt.title("Encoder-Decoder Cross-Attention (Last Layer, Head 0)")
    plt.tight_layout()
    plt.savefig("transformer_cross_attention.png", dpi=300, bbox_inches="tight")
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

    axes[1].plot(history["train_accuracy"], label="Train", linewidth=2)
    axes[1].plot(history["val_accuracy"], label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("transformer_training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Encoder-Decoder Transformer (Original Architecture)")
    print("=" * 70)

    # Generate data
    print("\n1. Generating sequence-to-sequence data...")
    src_seq, tgt_seq = generate_seq2seq_data(
        n_pairs=2000, src_len=20, tgt_len=25, src_vocab=100, tgt_vocab=80
    )

    # Split data
    n_train = int(0.8 * len(src_seq))
    n_val = int(0.1 * len(src_seq))

    train_src, train_tgt = src_seq[:n_train], tgt_seq[:n_train]
    val_src, val_tgt = (
        src_seq[n_train : n_train + n_val],
        tgt_seq[n_train : n_train + n_val],
    )
    test_src, test_tgt = src_seq[n_train + n_val :], tgt_seq[n_train + n_val :]

    print(f"Train: {len(train_src)}, Val: {len(val_src)}, Test: {len(test_src)}")

    # Create dataloaders
    train_dataset = Seq2SeqDataset(train_src, train_tgt)
    val_dataset = Seq2SeqDataset(val_src, val_tgt)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    print("\n2. Creating Transformer...")
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=80,
        d_model=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=1024,
        max_seq_len=64,
        dropout=0.1,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print("\n3. Training model...")
    history = train_transformer(model, train_loader, val_loader, epochs=30, lr=0.0003)

    # Plot training curves
    print("\n4. Plotting training curves...")
    plot_training_curves(history)

    # Visualize cross-attention
    print("\n5. Visualizing cross-attention...")
    model.load_state_dict(torch.load("best_transformer.pth"))
    visualize_cross_attention(model, test_src[0], test_tgt[0][:-1])

    print("\n" + "=" * 70)
    print("Encoder-Decoder Transformer Complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("✓ Encoder: bidirectional self-attention on source")
    print("✓ Decoder: causal self-attention + cross-attention to encoder")
    print("✓ Cross-attention allows decoder to focus on relevant source parts")
    print("✓ Original 'Attention Is All You Need' architecture")
    print("\nBest for: Translation, summarization, seq2seq tasks")


if __name__ == "__main__":
    main()
