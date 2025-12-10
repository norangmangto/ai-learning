"""
CLIP-style Text-Image Matching with PyTorch

Contrastive Language-Image Pre-training (CLIP):
- Joint embedding space for images and text
- Contrastive learning to match corresponding pairs
- Zero-shot classification capability
- Applications: image search, caption generation, multimodal understanding

This implementation includes:
- Dual encoder architecture (image + text)
- Contrastive loss (InfoNCE)
- Cross-modal retrieval
- Zero-shot classification
- Similarity visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time


class ImageEncoder(nn.Module):
    """CNN-based image encoder."""

    def __init__(self, embed_dim=512):
        super(ImageEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
            # 3x64x64 -> 64x32x32
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64x32x32 -> 128x16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128x16x16 -> 256x8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256x8x8 -> 512x4x4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(512, embed_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x


class TextEncoder(nn.Module):
    """LSTM-based text encoder."""

    def __init__(self, vocab_size, embed_dim=512, hidden_dim=256):
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)

        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)

        # Use final hidden states (concatenate forward and backward)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # Project to common embedding space
        x = self.projection(hidden)

        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x


class CLIPModel(nn.Module):
    """
    CLIP-style dual encoder for text-image matching.

    Uses contrastive learning to align image and text embeddings.
    """

    def __init__(self, vocab_size, embed_dim=512):
        super(CLIPModel, self).__init__()

        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        return image_features, text_features

    def get_similarity(self, images, texts):
        """Compute cosine similarity between images and texts."""
        image_features, text_features = self.forward(images, texts)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def contrastive_loss(logits_per_image, logits_per_text):
    """
    Contrastive loss (InfoNCE) for CLIP.

    Each image should match its corresponding text (diagonal).
    """
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)

    return (loss_image + loss_text) / 2


def generate_synthetic_multimodal_data(n_samples=1000, img_size=64, vocab_size=100):
    """
    Generate synthetic image-text pairs.

    Images have patterns (vertical, horizontal, checkerboard).
    Texts describe the patterns.
    """
    print(f"Generating {n_samples} image-text pairs...")

    np.random.seed(42)

    images = []
    texts = []
    labels = []

    # Text patterns for each class
    # Class 0: vertical lines -> tokens [1, 2, 3, ...]
    # Class 1: horizontal lines -> tokens [10, 11, 12, ...]
    # Class 2: checkerboard -> tokens [20, 21, 22, ...]

    for _ in range(n_samples):
        label = np.random.randint(0, 3)

        # Create image
        img = np.zeros((3, img_size, img_size), dtype=np.float32)

        if label == 0:
            # Vertical stripes
            for i in range(0, img_size, 8):
                img[:, :, i : i + 4] = 1.0
        elif label == 1:
            # Horizontal stripes
            for i in range(0, img_size, 8):
                img[:, i : i + 4, :] = 1.0
        else:
            # Checkerboard
            for i in range(0, img_size, 8):
                for j in range(0, img_size, 8):
                    if (i + j) % 16 == 0:
                        img[:, i : i + 8, j : j + 8] = 1.0

        # Add noise
        img += np.random.randn(3, img_size, img_size) * 0.1
        img = np.clip(img, 0, 1)

        # Create text description (tokens representing pattern)
        if label == 0:
            # "vertical lines pattern"
            text_tokens = [1, 2, 3, 4, 5]
        elif label == 1:
            # "horizontal lines pattern"
            text_tokens = [10, 11, 12, 13, 14]
        else:
            # "checkerboard grid pattern"
            text_tokens = [20, 21, 22, 23, 24]

        # Add some variation
        text_tokens = text_tokens + list(np.random.randint(30, 40, size=3))

        # Pad to fixed length
        text_tokens = text_tokens[:10] + [0] * max(0, 10 - len(text_tokens))

        images.append(img)
        texts.append(text_tokens)
        labels.append(label)

    return (
        np.array(images, dtype=np.float32),
        np.array(texts, dtype=np.int64),
        np.array(labels, dtype=np.int64),
    )


class MultimodalDataset(Dataset):
    """Dataset for image-text pairs."""

    def __init__(self, images, texts, labels):
        self.images = torch.FloatTensor(images)
        self.texts = torch.LongTensor(texts)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx], self.labels[idx]


def train_clip(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train CLIP model with contrastive learning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_val_accuracy = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, texts, labels in train_loader:
            images, texts = images.to(device), texts.to(device)

            # Get similarity logits
            logits_per_image, logits_per_text = model.get_similarity(images, texts)

            # Contrastive loss
            loss = contrastive_loss(logits_per_image, logits_per_text)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy: check if highest similarity is on diagonal
            predictions = logits_per_image.argmax(dim=1)
            targets = torch.arange(len(images), device=device)
            train_correct += (predictions == targets).sum().item()
            train_total += len(images)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, texts, labels in val_loader:
                images, texts = images.to(device), texts.to(device)

                logits_per_image, logits_per_text = model.get_similarity(images, texts)
                loss = contrastive_loss(logits_per_image, logits_per_text)

                val_loss += loss.item()

                predictions = logits_per_image.argmax(dim=1)
                targets = torch.arange(len(images), device=device)
                val_correct += (predictions == targets).sum().item()
                val_total += len(images)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_clip.pth")

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}"
            )

    return history


def cross_modal_retrieval(model, images, texts, labels, k=5):
    """
    Perform cross-modal retrieval.

    Given a query (image or text), retrieve most similar items from other modality.
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        images_tensor = torch.FloatTensor(images).to(device)
        texts_tensor = torch.LongTensor(texts).to(device)

        # Get embeddings
        image_features = model.image_encoder(images_tensor)
        text_features = model.text_encoder(texts_tensor)

        # Compute similarity matrix
        similarity = image_features @ text_features.t()

    similarity = similarity.cpu().numpy()

    print("\n" + "=" * 70)
    print("Cross-Modal Retrieval Results")
    print("=" * 70)

    # Image-to-Text retrieval
    print("\nImage-to-Text Retrieval (top-5 for first 3 images):")
    for i in range(min(3, len(images))):
        top_k_indices = np.argsort(similarity[i])[-k:][::-1]
        print(f"\nQuery Image {i} (Class {labels[i]}):")
        for rank, idx in enumerate(top_k_indices, 1):
            print(
                f"  {rank}. Text {idx} (Class {labels[idx]}) - Similarity: {similarity[i, idx]:.3f}"
            )

    # Text-to-Image retrieval
    print("\nText-to-Image Retrieval (top-5 for first 3 texts):")
    for i in range(min(3, len(texts))):
        top_k_indices = np.argsort(similarity[:, i])[-k:][::-1]
        print(f"\nQuery Text {i} (Class {labels[i]}):")
        for rank, idx in enumerate(top_k_indices, 1):
            print(
                f"  {rank}. Image {idx} (Class {labels[idx]}) - Similarity: {similarity[idx, i]:.3f}"
            )

    return similarity


def visualize_similarity_matrix(similarity, labels, n_samples=20):
    """Visualize image-text similarity matrix."""

    # Use subset for visualization
    similarity_subset = similarity[:n_samples, :n_samples]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Similarity matrix
    im = axes[0].imshow(similarity_subset, cmap="viridis", aspect="auto")
    axes[0].set_xlabel("Text Index")
    axes[0].set_ylabel("Image Index")
    axes[0].set_title("Image-Text Similarity Matrix")
    plt.colorbar(im, ax=axes[0])

    # Mark diagonal (correct matches)
    for i in range(n_samples):
        axes[0].plot(i, i, "r*", markersize=10)

    # Accuracy by class
    unique_labels = np.unique(labels[:n_samples])
    accuracies = []

    for label in unique_labels:
        mask = labels[:n_samples] == label
        class_similarity = similarity_subset[mask][:, mask]

        # Check if diagonal has highest similarity
        correct = 0
        for i in range(len(class_similarity)):
            if np.argmax(class_similarity[i]) == i:
                correct += 1

        accuracy = correct / len(class_similarity)
        accuracies.append(accuracy)

    axes[1].bar(unique_labels, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Retrieval Accuracy")
    axes[1].set_title("Per-Class Retrieval Accuracy")
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("clip_similarity_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_training_curves(history):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Contrastive Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_accuracy"], label="Train", linewidth=2)
    axes[1].plot(history["val_accuracy"], label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Matching Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("clip_training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def zero_shot_classification(model, images, text_prompts, true_labels):
    """
    Zero-shot classification using text prompts.

    Compare image with multiple text descriptions.
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        images_tensor = torch.FloatTensor(images).to(device)
        text_tensors = torch.LongTensor(text_prompts).to(device)

        # Get embeddings
        image_features = model.image_encoder(images_tensor)
        text_features = model.text_encoder(text_tensors)

        # Compute similarity
        logit_scale = model.logit_scale.exp()
        similarity = logit_scale * image_features @ text_features.t()

        # Softmax to get probabilities
        probs = F.softmax(similarity, dim=1)

    predictions = similarity.argmax(dim=1).cpu().numpy()
    probs = probs.cpu().numpy()

    accuracy = (predictions == true_labels).mean()

    print("\n" + "=" * 70)
    print("Zero-Shot Classification Results")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nPer-class probabilities (first 5 images):")
    for i in range(min(5, len(images))):
        print(f"\nImage {i} (True: {true_labels[i]}, Pred: {predictions[i]}):")
        for j, prob in enumerate(probs[i]):
            print(f"  Class {j}: {prob:.4f}")

    return predictions, probs


def main():
    """Main execution function."""
    print("=" * 70)
    print("CLIP-style Text-Image Matching")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic multimodal data...")
    images, texts, labels = generate_synthetic_multimodal_data(
        n_samples=1500, img_size=64, vocab_size=100
    )

    # Split data
    n_train = int(0.7 * len(images))
    n_val = int(0.15 * len(images))

    train_images, train_texts, train_labels = (
        images[:n_train],
        texts[:n_train],
        labels[:n_train],
    )
    val_images, val_texts, val_labels = (
        images[n_train : n_train + n_val],
        texts[n_train : n_train + n_val],
        labels[n_train : n_train + n_val],
    )
    test_images, test_texts, test_labels = (
        images[n_train + n_val :],
        texts[n_train + n_val :],
        labels[n_train + n_val :],
    )

    print(
        f"Train: {
        len(train_images)}, Val: {
            len(val_images)}, Test: {
                len(test_images)}"
    )

    # Create dataloaders
    train_dataset = MultimodalDataset(train_images, train_texts, train_labels)
    val_dataset = MultimodalDataset(val_images, val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    print("\n2. Creating CLIP model...")
    model = CLIPModel(vocab_size=100, embed_dim=512)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print("\n3. Training model...")
    history = train_clip(model, train_loader, val_loader, epochs=50, lr=0.001)

    # Plot training curves
    print("\n4. Plotting training curves...")
    plot_training_curves(history)

    # Load best model
    model.load_state_dict(torch.load("best_clip.pth"))

    # Cross-modal retrieval
    print("\n5. Testing cross-modal retrieval...")
    similarity = cross_modal_retrieval(model, test_images, test_texts, test_labels, k=5)

    # Visualize similarity
    print("\n6. Visualizing similarity matrix...")
    visualize_similarity_matrix(similarity, test_labels, n_samples=20)

    # Zero-shot classification
    print("\n7. Testing zero-shot classification...")

    # Create text prompts for each class
    text_prompts = np.array(
        [
            [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],  # Class 0: vertical
            [10, 11, 12, 13, 14, 0, 0, 0, 0, 0],  # Class 1: horizontal
            [20, 21, 22, 23, 24, 0, 0, 0, 0, 0],  # Class 2: checkerboard
        ],
        dtype=np.int64,
    )

    predictions, probs = zero_shot_classification(
        model, test_images[:50], text_prompts, test_labels[:50]
    )

    print("\n" + "=" * 70)
    print("CLIP Text-Image Matching Complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("✓ Joint embedding space for images and text")
    print("✓ Contrastive learning (InfoNCE loss)")
    print("✓ Cross-modal retrieval (image↔text)")
    print("✓ Zero-shot classification capability")
    print("✓ Scalable to large datasets")
    print("\nApplications: Image search, caption generation, multimodal understanding")


if __name__ == "__main__":
    main()
