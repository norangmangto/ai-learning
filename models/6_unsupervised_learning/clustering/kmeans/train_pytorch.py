"""
K-Means Clustering - PyTorch Implementation
GPU-accelerated K-Means using PyTorch
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import time


class KMeansPyTorch:
    """K-Means clustering using PyTorch (GPU-accelerated)"""

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, device="cuda"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def initialize_centroids(self, X, method="k-means++"):
        """
        Initialize centroids using K-Means++ algorithm
        """
        n_samples = X.shape[0]

        if method == "k-means++":
            # K-Means++ initialization
            centroids = torch.zeros((self.n_clusters, X.shape[1]), device=self.device)

            # Choose first centroid randomly
            idx = torch.randint(0, n_samples, (1,)).item()
            centroids[0] = X[idx]

            # Choose remaining centroids
            for i in range(1, self.n_clusters):
                # Compute distances to nearest centroid
                distances = torch.cdist(X, centroids[:i])
                min_distances = distances.min(dim=1)[0]

                # Choose next centroid with probability proportional to squared
                # distance
                probs = min_distances**2
                probs = probs / probs.sum()
                idx = torch.multinomial(probs, 1).item()
                centroids[i] = X[idx]

            return centroids
        else:
            # Random initialization
            indices = torch.randperm(n_samples)[: self.n_clusters]
            return X[indices].clone()

    def assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid"""
        # Compute distances to all centroids: [n_samples, n_clusters]
        distances = torch.cdist(X, centroids)

        # Assign to nearest centroid
        labels = torch.argmin(distances, dim=1)

        return labels, distances

    def update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        new_centroids = torch.zeros_like(self.centroids)

        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                # If cluster is empty, reinitialize randomly
                idx = torch.randint(0, X.shape[0], (1,)).item()
                new_centroids[k] = X[idx]

        return new_centroids

    def compute_inertia(self, X, labels, centroids):
        """Compute within-cluster sum of squares"""
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                cluster_points = X[mask]
                centroid = centroids[k]
                inertia += ((cluster_points - centroid) ** 2).sum().item()
        return inertia

    def fit(self, X):
        """
        Fit K-Means model

        Parameters:
        -----------
        X : torch.Tensor or numpy.ndarray
            Training data [n_samples, n_features]
        """
        # Convert to torch tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        # Initialize centroids
        self.centroids = self.initialize_centroids(X, method="k-means++")

        prev_inertia = float("inf")

        for iteration in range(self.max_iter):
            # Assign points to clusters
            labels, distances = self.assign_clusters(X, self.centroids)

            # Update centroids
            new_centroids = self.update_centroids(X, labels)

            # Check convergence
            self.centroids = new_centroids  # Compute inertia
            inertia = self.compute_inertia(X, labels, self.centroids)

            self.n_iter_ = iteration + 1

            if abs(prev_inertia - inertia) < self.tol:
                break

            prev_inertia = inertia

        self.labels_ = labels.cpu().numpy()
        self.inertia_ = inertia

        return self

    def predict(self, X):
        """Predict cluster labels for new data"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        labels, _ = self.assign_clusters(X, self.centroids)
        return labels.cpu().numpy()

    def fit_predict(self, X):
        """Fit model and return cluster labels"""
        self.fit(X)
        return self.labels_

    def transform(self, X):
        """Transform X to cluster-distance space"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        distances = torch.cdist(X, self.centroids)
        return distances.cpu().numpy()


def generate_data(n_samples=10000, n_features=10, n_clusters=5):
    """Generate synthetic clustering data"""
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=2.0,
        random_state=42,
    )
    return X, y_true


def benchmark_comparison(X, n_clusters=5):
    """Compare PyTorch vs scikit-learn performance"""
    from sklearn.cluster import KMeans as SKLearnKMeans

    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    # PyTorch (GPU)
    print("\n⚡ PyTorch K-Means (GPU):")
    kmeans_torch = KMeansPyTorch(n_clusters=n_clusters, device="cuda")

    start_time = time.time()
    kmeans_torch.fit(X)
    torch_time = time.time() - start_time

    print(f"  Time: {torch_time:.4f} seconds")
    print(f"  Inertia: {kmeans_torch.inertia_:.2f}")
    print(f"  Iterations: {kmeans_torch.n_iter_}")

    # Scikit-learn (CPU)
    print("\n🐍 Scikit-Learn K-Means (CPU):")
    kmeans_sklearn = SKLearnKMeans(
        n_clusters=n_clusters, n_init=1, max_iter=300, random_state=42
    )

    start_time = time.time()
    kmeans_sklearn.fit(X)
    sklearn_time = time.time() - start_time

    print(f"  Time: {sklearn_time:.4f} seconds")
    print(f"  Inertia: {kmeans_sklearn.inertia_:.2f}")
    print(f"  Iterations: {kmeans_sklearn.n_iter_}")

    # Speedup
    speedup = sklearn_time / torch_time
    print(f"\n🚀 Speedup: {speedup:.2f}x")

    print("=" * 60 + "\n")

    return kmeans_torch, kmeans_sklearn


def visualize_clusters_2d(X, labels, centers, title="PyTorch K-Means"):
    """Visualize clustering results for 2D data"""
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6, s=30, edgecolors="k"
    )

    if centers is not None:
        centers_np = centers.cpu().numpy() if torch.is_tensor(centers) else centers
        plt.scatter(
            centers_np[:, 0],
            centers_np[:, 1],
            c="red",
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=2,
            label="Centroids",
        )

    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("kmeans_pytorch_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("K-MEANS CLUSTERING - PYTORCH (GPU-ACCELERATED)")
    print("=" * 60)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 1. Generate data
    print("\n📊 Generating data...")
    n_samples = 10000
    n_features = 10
    n_clusters = 5

    X, y_true = generate_data(n_samples, n_features, n_clusters)
    print(f"Dataset shape: {X.shape}")

    # 2. Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train PyTorch K-Means
    print(f"\n🚀 Training PyTorch K-Means (K={n_clusters})...")
    kmeans = KMeansPyTorch(n_clusters=n_clusters, max_iter=300, device=device)

    start_time = time.time()
    labels = kmeans.fit_predict(X_scaled)
    train_time = time.time() - start_time

    print(f"\n✅ Training complete!")
    print(f"   Time: {train_time:.4f} seconds")
    print(f"   Inertia: {kmeans.inertia_:.2f}")
    print(f"   Iterations: {kmeans.n_iter_}")

    # 4. Evaluate
    print("\n📊 Evaluation Metrics:")
    silhouette = silhouette_score(X_scaled, labels)
    print(f"   Silhouette Score: {silhouette:.4f}")

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n   Cluster Sizes:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        print(f"     Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")

    # 5. Benchmark comparison
    if n_features <= 100:  # Only for reasonable sizes
        torch_model, sklearn_model = benchmark_comparison(X_scaled, n_clusters)

    # 6. Visualize if 2D
    if n_features == 2:
        print("\n📈 Visualizing clusters...")
        visualize_clusters_2d(
            X_scaled,
            labels,
            kmeans.centroids,
            title=f"PyTorch K-Means (K={n_clusters})",
        )
        print("   Saved: kmeans_pytorch_clusters.png")

    # 7. Predict new samples
    print("\n🔮 Testing prediction on new samples...")
    X_new = np.random.randn(5, n_features)
    X_new_scaled = scaler.transform(X_new)
    new_labels = kmeans.predict(X_new_scaled)
    print(f"   New sample labels: {new_labels}")

    # 8. Get distances to centroids
    distances = kmeans.transform(X_new_scaled)
    print(f"\n   Distances to centroids (first sample):")
    print(f"   {distances[0]}")

    return kmeans, labels


if __name__ == "__main__":
    model, labels = main()
