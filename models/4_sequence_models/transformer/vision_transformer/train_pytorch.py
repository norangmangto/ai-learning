"""
Vision Transformer (ViT) with PyTorch

Vision Transformer applies transformer architecture to images:
- Splits image into patches (e.g., 16x16)
- Treats patches as tokens/words
- Uses standard transformer encoder
- Classification token (CLS) for predictions

This implementation includes:
- Patch embedding
- Positional encoding for patches
- Transformer encoder
- Classification head
- Attention visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.

    Image shape: (batch, channels, height, width)
    Output shape: (batch, num_patches, embed_dim)
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Convolutional layer to extract patches and embed them
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.projection(x)  # (batch, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.size()

        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch, num_tokens, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, num_tokens, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, num_tokens, embed_dim)

        output = self.projection(context)

        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise feedforward network."""
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_output, attn_weights = self.attention(self.norm1(x))
        x = x + self.dropout1(attn_output)

        # Feed-forward with residual
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)

        return x, attn_weights


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    Architecture:
    1. Split image into patches
    2. Linear embedding of patches
    3. Add learnable positional embeddings
    4. Add learnable [CLS] token
    5. Pass through transformer encoder
    6. Use [CLS] token for classification
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Pass through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attention_weights.append(attn)

        x = self.norm(x)

        # Use [CLS] token for classification
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits, attention_weights


def generate_synthetic_images(n_samples=1000, img_size=64, n_classes=3):
    """Generate synthetic images with simple patterns."""
    print(f"Generating {n_samples} synthetic images...")

    np.random.seed(42)

    images = []
    labels = []

    for _ in range(n_samples):
        label = np.random.randint(0, n_classes)

        # Create image with class-specific pattern
        img = np.zeros((3, img_size, img_size), dtype=np.float32)

        if label == 0:
            # Vertical stripes
            for i in range(0, img_size, 8):
                img[:, :, i:i+4] = 1.0
        elif label == 1:
            # Horizontal stripes
            for i in range(0, img_size, 8):
                img[:, i:i+4, :] = 1.0
        else:
            # Checkerboard
            for i in range(0, img_size, 8):
                for j in range(0, img_size, 8):
                    if (i + j) % 16 == 0:
                        img[:, i:i+8, j:j+8] = 1.0

        # Add noise
        img += np.random.randn(3, img_size, img_size) * 0.1
        img = np.clip(img, 0, 1)

        images.append(img)
        labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


class ImageDataset(Dataset):
    """Dataset for image classification."""
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def train_vit(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train Vision Transformer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    best_val_accuracy = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits, _ = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_vit.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

    return history


def visualize_attention_maps(model, image, label):
    """Visualize attention maps for image patches."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        img_tensor = torch.FloatTensor(image).unsqueeze(0).to(device)
        _, attention_weights = model(img_tensor)

        # Get attention from last layer, first head
        # Shape: (batch, num_heads, num_tokens, num_tokens)
        attn = attention_weights[-1][0, 0].cpu().numpy()

        # Attention from [CLS] token to patches
        cls_attn = attn[0, 1:]  # Skip [CLS] to [CLS]

    # Reshape to grid
    num_patches = int(np.sqrt(len(cls_attn)))
    attn_grid = cls_attn.reshape(num_patches, num_patches)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original image
    img_display = image.transpose(1, 2, 0)
    axes[0].imshow(img_display)
    axes[0].set_title(f'Original Image (Class {label})')
    axes[0].axis('off')

    # Attention map
    im = axes[1].imshow(attn_grid, cmap='viridis')
    axes[1].set_title('[CLS] Token Attention to Patches')
    axes[1].set_xlabel('Patch Column')
    axes[1].set_ylabel('Patch Row')
    plt.colorbar(im, ax=axes[1])

    # Overlay attention on image
    from scipy.ndimage import zoom
    img_size = image.shape[1]
    scale = img_size / num_patches
    attn_upsampled = zoom(attn_grid, scale, order=1)

    axes[2].imshow(img_display)
    axes[2].imshow(attn_upsampled, alpha=0.6, cmap='jet')
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('vit_attention_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(history):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_accuracy'], label='Train', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vit_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("Vision Transformer (ViT)")
    print("="*70)

    # Generate data
    print("\n1. Generating synthetic images...")
    images, labels = generate_synthetic_images(n_samples=1500, img_size=64, n_classes=3)

    # Split data
    n_train = int(0.7 * len(images))
    n_val = int(0.15 * len(images))

    train_images, train_labels = images[:n_train], labels[:n_train]
    val_images, val_labels = images[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_images, test_labels = images[n_train+n_val:], labels[n_train+n_val:]

    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    # Create dataloaders
    train_dataset = ImageDataset(train_images, train_labels)
    val_dataset = ImageDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    print("\n2. Creating Vision Transformer...")
    model = VisionTransformer(
        img_size=64,
        patch_size=8,
        in_channels=3,
        num_classes=3,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Number of patches: {model.patch_embed.num_patches}")

    # Train
    print("\n3. Training model...")
    history = train_vit(model, train_loader, val_loader, epochs=50, lr=0.001)

    # Plot training curves
    print("\n4. Plotting training curves...")
    plot_training_curves(history)

    # Visualize attention
    print("\n5. Visualizing attention maps...")
    model.load_state_dict(torch.load('best_vit.pth'))
    visualize_attention_maps(model, test_images[0], test_labels[0])

    print("\n" + "="*70)
    print("Vision Transformer Complete!")
    print("="*70)
    print("\nKey Features:")
    print("✓ Splits images into patches (tokens)")
    print("✓ Treats image classification as sequence modeling")
    print("✓ Uses standard transformer encoder")
    print("✓ [CLS] token for global image representation")
    print("✓ No convolutional layers (except patch embedding)")
    print("\nBest for: Image classification, especially with large datasets")


if __name__ == "__main__":
    main()
