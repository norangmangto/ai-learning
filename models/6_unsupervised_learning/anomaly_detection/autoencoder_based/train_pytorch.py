"""
Autoencoder-based Anomaly Detection with PyTorch

Autoencoders for anomaly detection:
- Train on normal data to learn compressed representation
- Normal data reconstructs well (low reconstruction error)
- Anomalies reconstruct poorly (high reconstruction error)
- Deep learning approach, can capture complex patterns

This implementation includes:
- Standard Autoencoder architecture
- Reconstruction error analysis
- Threshold tuning for anomaly detection
- Comparison with traditional methods
- Visualization of reconstructions
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)
from sklearn.model_selection import train_test_split
import seaborn as sns


class Autoencoder(nn.Module):
    """
    Standard Autoencoder architecture for anomaly detection.
    """

    def __init__(self, input_dim, encoding_dim=8):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


def generate_anomaly_data(n_samples=2000, contamination=0.1, n_features=20):
    """
    Generate dataset with anomalies.

    Args:
        n_samples: Total number of samples
        contamination: Fraction of anomalies
        n_features: Number of features

    Returns:
        X: Feature matrix
        y_true: True labels (1 = normal, 0 = anomaly)
    """
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies

    # Normal data
    X_normal, _ = make_classification(
        n_samples=n_normal,
        n_features=n_features,
        n_informative=n_features - 4,
        n_redundant=4,
        n_clusters_per_class=2,
        random_state=42,
    )

    # Anomalies: extreme values
    np.random.seed(42)
    X_anomalies = np.random.uniform(low=-5, high=5, size=(n_anomalies, n_features))

    # Combine and shuffle
    X = np.vstack([X_normal, X_anomalies])
    y_true = np.hstack([np.ones(n_normal), np.zeros(n_anomalies)])

    shuffle_idx = np.random.RandomState(42).permutation(len(X))
    X = X[shuffle_idx]
    y_true = y_true[shuffle_idx]

    print(f"Generated dataset:")
    print(f"  Total samples: {len(X)}")
    print(f"  Normal: {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
    print(f"  Anomalies: {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true)*100:.1f}%)")
    print(f"  Features: {X.shape[1]}")

    return X, y_true


def train_autoencoder(
    X_train, input_dim, encoding_dim=8, epochs=50, batch_size=32, lr=0.001
):
    """
    Train autoencoder on normal data.

    Args:
        X_train: Training data (normal samples only)
        input_dim: Number of input features
        encoding_dim: Dimension of encoded representation
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate

    Returns:
        model: Trained autoencoder
        train_losses: Training loss history
    """
    print(f"\nTraining Autoencoder...")
    print(f"  Input dim: {input_dim}")
    print(f"  Encoding dim: {encoding_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = Autoencoder(input_dim, encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare data
    X_tensor = torch.FloatTensor(X_train).to(device)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, _ in dataloader:
            # Forward pass
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    print(f"\nTraining complete! Final loss: {train_losses[-1]:.6f}")

    return model, train_losses


def calculate_reconstruction_errors(model, X):
    """
    Calculate reconstruction errors for data.

    Args:
        model: Trained autoencoder
        X: Data to evaluate

    Returns:
        errors: Reconstruction errors (MSE per sample)
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        reconstructed = model(X_tensor)

        # Calculate MSE per sample
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        errors = errors.cpu().numpy()

    return errors


def find_optimal_threshold(errors, y_true, method="f1"):
    """
    Find optimal threshold for anomaly detection.

    Args:
        errors: Reconstruction errors
        y_true: True labels (1 = normal, 0 = anomaly)
        method: 'f1' or 'percentile'

    Returns:
        threshold: Optimal threshold
    """
    if method == "f1":
        # Find threshold that maximizes F1 score
        thresholds = np.percentile(errors, np.arange(80, 100, 0.5))
        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            y_pred = (errors > threshold).astype(int)
            y_pred = 1 - y_pred  # Convert to (1=normal, 0=anomaly)
            f1 = f1_score(y_true, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"\nOptimal threshold (F1 maximization): {best_threshold:.6f}")
        print(f"F1 Score: {best_f1:.4f}")

    else:  # percentile
        # Use percentile of training errors
        best_threshold = np.percentile(errors, 95)
        print(f"\nThreshold (95th percentile): {best_threshold:.6f}")

    return best_threshold


def evaluate_autoencoder(model, X_test, y_test, threshold):
    """
    Evaluate autoencoder anomaly detection.

    Args:
        model: Trained autoencoder
        X_test: Test data
        y_test: True labels
        threshold: Anomaly threshold

    Returns:
        y_pred: Predictions
        errors: Reconstruction errors
    """
    # Calculate reconstruction errors
    errors = calculate_reconstruction_errors(model, X_test)

    # Predict (1 = normal, 0 = anomaly)
    y_pred = (errors <= threshold).astype(int)

    print("\n" + "=" * 60)
    print("Performance Evaluation")
    print("=" * 60)

    print(f"\nPredicted normal: {sum(y_pred == 1)}")
    print(f"Predicted anomalies: {sum(y_pred == 0)}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Anomaly", "Normal"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Anomaly", "Normal"],
        yticklabels=["Anomaly", "Normal"],
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("autoencoder_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("=" * 60)

    return y_pred, errors


def plot_training_loss(train_losses):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2, color="darkblue")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Autoencoder Training Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("autoencoder_training_loss.png", dpi=300, bbox_inches="tight")
    plt.show()


def analyze_reconstruction_errors(errors, y_true):
    """
    Analyze distribution of reconstruction errors.

    Args:
        errors: Reconstruction errors
        y_true: True labels (1 = normal, 0 = anomaly)
    """
    normal_errors = errors[y_true == 1]
    anomaly_errors = errors[y_true == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(
        normal_errors,
        bins=50,
        alpha=0.7,
        label="Normal",
        color="blue",
        edgecolor="black",
    )
    axes[0].hist(
        anomaly_errors,
        bins=50,
        alpha=0.7,
        label="Anomaly",
        color="red",
        edgecolor="black",
    )
    axes[0].set_xlabel("Reconstruction Error", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Distribution of Reconstruction Errors", fontsize=13)
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Box plot
    data_to_plot = [normal_errors, anomaly_errors]
    bp = axes[1].boxplot(data_to_plot, labels=["Normal", "Anomaly"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")
    axes[1].set_ylabel("Reconstruction Error", fontsize=12)
    axes[1].set_title("Reconstruction Error by Class", fontsize=13)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("autoencoder_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nReconstruction Error Statistics:")
    print(f"Normal - Mean: {normal_errors.mean():.6f}, Std: {normal_errors.std():.6f}")
    print(
        f"Anomaly - Mean: {anomaly_errors.mean():.6f}, Std: {anomaly_errors.std():.6f}"
    )
    print(
        f"Separation ratio: {
        anomaly_errors.mean() /
         normal_errors.mean():.2f}x"
    )


def plot_roc_pr_curves(y_true, errors):
    """
    Plot ROC and Precision-Recall curves.

    Args:
        y_true: True labels (1 = normal, 0 = anomaly)
        errors: Reconstruction errors (higher = more anomalous)
    """
    # For ROC curve: lower error = normal (higher score)
    scores = -errors

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Calculate PR curve
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    axes[0].plot(
        fpr,
        tpr,
        linewidth=2,
        color="darkorange",
        label=f"ROC Curve (AUC = {roc_auc:.3f})",
    )
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title("ROC Curve", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    axes[1].plot(
        recall,
        precision,
        linewidth=2,
        color="green",
        label=f"PR Curve (AUC = {pr_auc:.3f})",
    )
    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall Curve", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("autoencoder_roc_pr.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")


def visualize_reconstructions(model, X, y_true, n_samples=5):
    """
    Visualize original vs reconstructed samples.

    Args:
        model: Trained autoencoder
        X: Data samples
        y_true: True labels
        n_samples: Number of samples to visualize
    """
    device = next(model.parameters()).device
    model.eval()

    # Get some normal and anomalous samples
    normal_idx = np.where(y_true == 1)[0][:n_samples]
    anomaly_idx = np.where(y_true == 0)[0][:n_samples]

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    with torch.no_grad():
        # Normal samples
        for i, idx in enumerate(normal_idx):
            X_sample = torch.FloatTensor(X[idx : idx + 1]).to(device)
            reconstructed = model(X_sample).cpu().numpy()[0]
            original = X[idx]

            axes[0, i].plot(original, "b-", label="Original", linewidth=2)
            axes[0, i].plot(reconstructed, "r--", label="Reconstructed", linewidth=2)
            error = np.mean((original - reconstructed) ** 2)
            axes[0, i].set_title(f"Normal\nError: {error:.4f}", fontsize=10)
            if i == 0:
                axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

        # Anomalous samples
        for i, idx in enumerate(anomaly_idx):
            X_sample = torch.FloatTensor(X[idx : idx + 1]).to(device)
            reconstructed = model(X_sample).cpu().numpy()[0]
            original = X[idx]

            axes[1, i].plot(original, "b-", label="Original", linewidth=2)
            axes[1, i].plot(reconstructed, "r--", label="Reconstructed", linewidth=2)
            error = np.mean((original - reconstructed) ** 2)
            axes[1, i].set_title(f"Anomaly\nError: {error:.4f}", fontsize=10)
            if i == 0:
                axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("autoencoder_reconstructions.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Autoencoder-based Anomaly Detection")
    print("=" * 70)

    # 1. Generate data
    print("\n1. Generating dataset with anomalies...")
    X, y_true = generate_anomaly_data(n_samples=2000, contamination=0.1, n_features=20)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data: train on normal samples only
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_true, test_size=0.3, random_state=42, stratify=y_true
    )

    # Use only normal samples for training
    X_train_normal = X_train[y_train == 1]
    print(f"\nTraining set (normal only): {len(X_train_normal)} samples")
    print(
        f"Test set: {len(X_test)} samples ({sum(y_test==1)} normal, {sum(y_test==0)} anomalies)"
    )

    # 2. Train autoencoder
    print("\n2. Training autoencoder on normal data...")
    model, train_losses = train_autoencoder(
        X_train_normal,
        input_dim=X_scaled.shape[1],
        encoding_dim=8,
        epochs=50,
        batch_size=32,
        lr=0.001,
    )

    # 3. Plot training loss
    print("\n3. Plotting training loss...")
    plot_training_loss(train_losses)

    # 4. Calculate reconstruction errors on training set
    print("\n4. Calculating reconstruction errors...")
    train_errors = calculate_reconstruction_errors(model, X_train)

    # 5. Find optimal threshold
    print("\n5. Finding optimal anomaly threshold...")
    threshold = find_optimal_threshold(train_errors, y_train, method="f1")

    # 6. Evaluate on test set
    print("\n6. Evaluating on test set...")
    y_pred, test_errors = evaluate_autoencoder(model, X_test, y_test, threshold)

    # 7. Analyze reconstruction errors
    print("\n7. Analyzing reconstruction error distribution...")
    analyze_reconstruction_errors(test_errors, y_test)

    # 8. Plot ROC and PR curves
    print("\n8. Plotting ROC and Precision-Recall curves...")
    plot_roc_pr_curves(y_test, test_errors)

    # 9. Visualize reconstructions
    print("\n9. Visualizing sample reconstructions...")
    visualize_reconstructions(model, X_test, y_test, n_samples=5)

    print("\n" + "=" * 70)
    print("Autoencoder Analysis Complete!")
    print("=" * 70)

    print("\nKey Advantages:")
    print("✓ Can learn complex patterns in data")
    print("✓ Unsupervised (only needs normal data for training)")
    print("✓ Flexible architecture (can adapt to different data types)")
    print("✓ Provides interpretable reconstruction errors")
    print("✓ Can handle high-dimensional data")

    print("\nWhen to use Autoencoder:")
    print("✓ Complex, high-dimensional data")
    print("✓ Have enough normal samples for training")
    print("✓ Need to capture non-linear patterns")
    print("✓ When interpretability is important (can visualize reconstructions)")

    print("\nLimitations:")
    print("✗ Requires more normal training data")
    print("✗ Slower to train than traditional methods")
    print("✗ Requires GPU for large datasets")
    print("✗ Hyperparameter tuning needed (architecture, epochs, etc.)")
    print("✗ May not work well if anomalies are subtle")

    print("\nBest Practices:")
    print("1. Train only on normal data")
    print("2. Use validation set to tune architecture and threshold")
    print("3. Scale features before training")
    print("4. Monitor training loss to avoid overfitting")
    print("5. Try different encoding dimensions (4, 8, 16)")
    print("6. Consider variational autoencoder (VAE) for better anomaly detection")


if __name__ == "__main__":
    main()
