"""
Decoder-Only Transformer (GPT-style) with PyTorch

Decoder-only transformers are used for generation tasks:
- GPT (Generative Pre-trained Transformer)
- GPT-2, GPT-3, GPT-4 variants
- Used for: text generation, completion, few-shot learning
- Causal (autoregressive) attention masks

This implementation includes:
- Masked multi-head self-attention (causal)
- Position-wise feedforward networks
- Positional encoding
- Autoregressive generation
- Temperature and top-k sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

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


class CausalSelfAttention(nn.Module):
    """
    Masked multi-head self-attention with causal mask.

    Ensures that prediction for position i can only depend on
    positions < i (autoregressive property).
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CausalSelfAttention, self).__init__()

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

        # Register causal mask buffer
        self.register_buffer("causal_mask", None)

    def _create_causal_mask(self, seq_len, device):
        """Create lower triangular causal mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Create causal mask if needed
        if self.causal_mask is None or self.causal_mask.size(2) < seq_len:
            self.causal_mask = self._create_causal_mask(seq_len, x.device)

        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0, -1e9
        )

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
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
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class DecoderBlock(nn.Module):
    """Single decoder block with masked self-attention."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Masked self-attention
        attn_output, attn_weights = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attn_weights


class GPTDecoder(nn.Module):
    """
    Decoder-only Transformer (GPT-style).

    Autoregressive model for sequence generation.
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
    ):
        super(GPTDecoder, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # Embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through decoder blocks
        attention_weights = []
        for decoder_block in self.decoder_blocks:
            x, attn = decoder_block(x)
            attention_weights.append(attn)

        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits, attention_weights

    @torch.no_grad()
    def generate(self, start_tokens, max_length=50, temperature=1.0, top_k=None):
        """
        Generate sequence autoregressively.

        Args:
            start_tokens: Initial tokens (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        """
        self.eval()
        generated = start_tokens.clone()

        for _ in range(max_length):
            # Get predictions
            logits, _ = self.forward(generated)

            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated


def generate_language_modeling_data(n_sequences=1000, seq_length=32, vocab_size=100):
    """Generate synthetic data for language modeling."""
    print(f"Generating {n_sequences} sequences for language modeling...")

    np.random.seed(42)

    sequences = []

    for _ in range(n_sequences):
        # Generate sequence with some structure
        seq = []
        for i in range(seq_length):
            if i == 0:
                # Start token
                token = np.random.randint(1, 10)
            else:
                # Token depends on previous token (simple Markov chain)
                prev_token = seq[-1]
                if prev_token < vocab_size // 2:
                    token = np.random.randint(1, vocab_size // 2)
                else:
                    token = np.random.randint(vocab_size // 2, vocab_size)

                # Add some randomness
                if np.random.rand() < 0.2:
                    token = np.random.randint(1, vocab_size)

            seq.append(token)

        sequences.append(seq)

    return np.array(sequences, dtype=np.int64)


class LanguageModelingDataset(Dataset):
    """Dataset for language modeling (predict next token)."""

    def __init__(self, sequences):
        self.sequences = torch.LongTensor(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input: seq[:-1], Target: seq[1:]
        return seq[:-1], seq[1:]


def train_gpt(model, train_loader, val_loader, epochs=30, lr=0.0003):
    """Train GPT decoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_perplexity": [],
        "val_perplexity": [],
    }
    best_val_loss = float("inf")

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_tokens = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits, _ = model(inputs)

            # Reshape for loss calculation
            loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * targets.numel()
            train_tokens += targets.numel()

        train_loss /= train_tokens
        train_perplexity = math.exp(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_tokens = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)

                loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))

                val_loss += loss.item() * targets.numel()
                val_tokens += targets.numel()

        val_loss /= val_tokens
        val_perplexity = math.exp(val_loss)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_perplexity"].append(train_perplexity)
        history["val_perplexity"].append(val_perplexity)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_gpt_decoder.pth")

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, PPL: {train_perplexity:.2f} | "
                f"Val Loss: {val_loss:.4f}, PPL: {val_perplexity:.2f}"
            )

    return history


def visualize_causal_attention(model, sequence):
    """Visualize causal attention pattern."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        seq_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)
        _, attention_weights = model(seq_tensor)

        # Get attention from first layer, first head
        attn = attention_weights[0][0, 0].cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Causal Self-Attention (Lower Triangular)")
    plt.tight_layout()
    plt.savefig("decoder_causal_attention.png", dpi=300, bbox_inches="tight")
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

    axes[1].plot(history["train_perplexity"], label="Train", linewidth=2)
    axes[1].plot(history["val_perplexity"], label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Training and Validation Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("decoder_training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Decoder-Only Transformer (GPT-style)")
    print("=" * 70)

    # Generate data
    print("\n1. Generating language modeling data...")
    sequences = generate_language_modeling_data(
        n_sequences=2000, seq_length=33, vocab_size=100
    )

    # Split data
    n_train = int(0.8 * len(sequences))
    n_val = int(0.1 * len(sequences))

    train_seq = sequences[:n_train]
    val_seq = sequences[n_train : n_train + n_val]
    test_seq = sequences[n_train + n_val :]

    print(f"Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")

    # Create dataloaders
    train_dataset = LanguageModelingDataset(train_seq)
    val_dataset = LanguageModelingDataset(val_seq)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    print("\n2. Creating GPT Decoder...")
    model = GPTDecoder(
        vocab_size=100,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=64,
        dropout=0.1,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print("\n3. Training model...")
    history = train_gpt(model, train_loader, val_loader, epochs=30, lr=0.0003)

    # Plot training curves
    print("\n4. Plotting training curves...")
    plot_training_curves(history)

    # Generate text
    print("\n5. Generating sequences...")
    model.load_state_dict(torch.load("best_gpt_decoder.pth"))
    device = next(model.parameters()).device

    # Start with a few tokens
    start_tokens = torch.LongTensor([[5, 15, 25]]).to(device)

    print("\nGreedy generation:")
    generated = model.generate(start_tokens, max_length=20, temperature=0.1)
    print(f"Generated sequence: {generated[0].cpu().numpy()}")

    print("\nSampling with temperature=1.0:")
    generated = model.generate(start_tokens, max_length=20, temperature=1.0)
    print(f"Generated sequence: {generated[0].cpu().numpy()}")

    print("\nTop-k sampling (k=10):")
    generated = model.generate(start_tokens, max_length=20, temperature=1.0, top_k=10)
    print(f"Generated sequence: {generated[0].cpu().numpy()}")

    # Visualize causal attention
    print("\n6. Visualizing causal attention...")
    visualize_causal_attention(model, test_seq[0][:-1])

    print("\n" + "=" * 70)
    print("Decoder-Only Transformer Complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("✓ Causal (masked) self-attention for autoregressive generation")
    print("✓ Can only attend to previous positions")
    print("✓ Natural for language generation tasks")
    print("✓ Supports various sampling strategies (greedy, temperature, top-k)")
    print("\nBest for: Text generation, completion, dialogue, code generation")


if __name__ == "__main__":
    main()
