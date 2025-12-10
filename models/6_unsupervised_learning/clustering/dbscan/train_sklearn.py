"""
DBSCAN Clustering - Scikit-Learn Implementation
Density-Based Spatial Clustering of Applications with Noise
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors


def generate_sample_data(dataset_type="moons", n_samples=1000):
    """
    Generate synthetic clustering data
    DBSCAN works well with non-convex shapes
    """
    if dataset_type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    elif dataset_type == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=0.05, factor=0.5, random_state=42
        )
    elif dataset_type == "blobs":
        X, y = make_blobs(
            n_samples=n_samples, centers=4, cluster_std=0.6, random_state=42
        )
    else:
        # Mixed dataset with noise
        X1, _ = make_blobs(
            n_samples=n_samples // 2, centers=3, cluster_std=0.4, random_state=42
        )
        X2 = np.random.uniform(low=-3, high=3, size=(n_samples // 2, 2))
        X = np.vstack([X1, X2])
        y = None

    return X, y


def find_optimal_eps(X, min_samples=5, k=4):
    """
    Find optimal eps parameter using k-distance graph

    The elbow in the k-distance plot suggests a good eps value
    """
    # Compute k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)

    # Sort distances
    distances = np.sort(distances[:, k - 1], axis=0)

    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.ylabel(f"{k}-NN Distance")
    plt.xlabel("Points sorted by distance")
    plt.title(f'K-Distance Graph (k={k}) - Find the "Elbow"')
    plt.grid(True, alpha=0.3)

    # Add horizontal lines at potential eps values
    percentiles = [90, 95, 98]
    for p in percentiles:
        eps_candidate = np.percentile(distances, p)
        plt.axhline(
            y=eps_candidate,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"{p}th percentile: {eps_candidate:.3f}",
        )

    plt.legend()
    plt.savefig("dbscan_eps_selection.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Suggest eps as 95th percentile
    suggested_eps = np.percentile(distances, 95)
    print(f"\n📊 K-Distance Analysis (k={k}):")
    print(f"   Suggested eps (95th percentile): {suggested_eps:.4f}")
    print(f"   Range: [{distances.min():.4f}, {distances.max():.4f}]")

    return suggested_eps


def train_dbscan(X, eps=0.5, min_samples=5, metric="euclidean"):
    """
    Train DBSCAN clustering model

    Parameters:
    -----------
    X : array-like
        Training data
    eps : float
        Maximum distance between two samples to be considered neighbors
        Smaller eps = more clusters, more noise points
    min_samples : int
        Minimum number of samples in neighborhood for core point
        Larger min_samples = denser clusters, more noise points
    metric : str
        Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
    """
    dbscan = DBSCAN(
        eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1  # Use all CPU cores
    )

    # Fit and get labels
    labels = dbscan.fit_predict(X)

    return dbscan, labels


def evaluate_clustering(X, labels):
    """
    Evaluate DBSCAN clustering quality
    Note: Noise points (label=-1) are excluded from some metrics
    """
    print("\n" + "=" * 60)
    print("DBSCAN CLUSTERING EVALUATION")
    print("=" * 60)

    # Count clusters and noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\n✓ Number of Clusters: {n_clusters}")
    print(
        f"✓ Number of Noise Points: {n_noise} ({
        n_noise/
        len(labels)*
        100:.1f}%)"
    )

    # Cluster sizes
    unique_labels = set(labels)
    print(f"\n✓ Cluster Sizes:")
    for label in sorted(unique_labels):
        if label == -1:
            print(f"  Noise: {list(labels).count(label)} points")
        else:
            count = list(labels).count(label)
            percentage = (count / len(labels)) * 100
            print(f"  Cluster {label}: {count} points ({percentage:.1f}%)")

    # Silhouette score (excluding noise points)
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 0:
            silhouette = silhouette_score(X[mask], labels[mask])
            print(f"\n✓ Silhouette Score (excluding noise): {silhouette:.4f}")
            print(f"  (Range: -1 to 1, higher is better)")

            # Davies-Bouldin Index
            if n_clusters > 1:
                db_index = davies_bouldin_score(X[mask], labels[mask])
                print(f"\n✓ Davies-Bouldin Index: {db_index:.4f}")
                print(f"  (Lower is better)")

    print("=" * 60 + "\n")

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels),
    }


def visualize_clusters(X, labels, title="DBSCAN Clustering"):
    """
    Visualize DBSCAN clustering results
    """
    plt.figure(figsize=(12, 8))

    # Unique labels
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in black
            class_member_mask = labels == label
            xy = X[class_member_mask]
            plt.scatter(
                xy[:, 0],
                xy[:, 1],
                c="black",
                marker="x",
                s=50,
                alpha=0.5,
                label="Noise",
            )
        else:
            # Cluster points
            class_member_mask = labels == label
            xy = X[class_member_mask]
            plt.scatter(
                xy[:, 0],
                xy[:, 1],
                c=[color],
                s=50,
                alpha=0.8,
                edgecolors="k",
                linewidth=0.5,
                label=f"Cluster {label}",
            )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("dbscan_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()


def parameter_sensitivity_analysis(X, eps_range, min_samples_range):
    """
    Analyze how parameters affect clustering results
    """
    results = []

    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)

            results.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "noise_ratio": noise_ratio,
                }
            )

            print(f"\neps={eps:.3f}, min_samples={min_samples}:")
            print(f"  Clusters: {n_clusters}, Noise: {noise_ratio:.1%}")

    print("=" * 60 + "\n")

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Number of clusters
    for min_samp in min_samples_range:
        data = [r for r in results if r["min_samples"] == min_samp]
        eps_vals = [r["eps"] for r in data]
        n_clust = [r["n_clusters"] for r in data]
        ax1.plot(eps_vals, n_clust, marker="o", label=f"min_samples={min_samp}")

    ax1.set_xlabel("eps")
    ax1.set_ylabel("Number of Clusters")
    ax1.set_title("Effect of eps on Cluster Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Noise ratio
    for min_samp in min_samples_range:
        data = [r for r in results if r["min_samples"] == min_samp]
        eps_vals = [r["eps"] for r in data]
        noise_ratios = [r["noise_ratio"] for r in data]
        ax2.plot(eps_vals, noise_ratios, marker="o", label=f"min_samples={min_samp}")

    ax2.set_xlabel("eps")
    ax2.set_ylabel("Noise Ratio")
    ax2.set_title("Effect of eps on Noise Points")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dbscan_parameter_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    return results


def compare_with_kmeans(X, n_clusters=None):
    """
    Compare DBSCAN with K-Means on same data
    """
    from sklearn.cluster import KMeans

    print("\n" + "=" * 60)
    print("DBSCAN vs K-MEANS COMPARISON")
    print("=" * 60)

    # DBSCAN
    eps = find_optimal_eps(X, k=4)
    dbscan, dbscan_labels = train_dbscan(X, eps=eps, min_samples=5)
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    # K-Means
    if n_clusters is None:
        n_clusters = max(n_dbscan_clusters, 2)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # DBSCAN
    unique_labels = set(dbscan_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            mask = dbscan_labels == label
            ax1.scatter(X[mask, 0], X[mask, 1], c="black", marker="x", s=50, alpha=0.5)
        else:
            mask = dbscan_labels == label
            ax1.scatter(
                X[mask, 0], X[mask, 1], c=[color], s=50, alpha=0.8, edgecolors="k"
            )
    ax1.set_title(
        f"DBSCAN ({n_dbscan_clusters} clusters, {list(dbscan_labels).count(-1)} noise points)"
    )
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.grid(True, alpha=0.3)

    # K-Means
    ax2.scatter(
        X[:, 0],
        X[:, 1],
        c=kmeans_labels,
        cmap="viridis",
        s=50,
        alpha=0.8,
        edgecolors="k",
    )
    ax2.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=2,
    )
    ax2.set_title(f"K-Means ({n_clusters} clusters)")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dbscan_vs_kmeans.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"\n✓ DBSCAN: {n_dbscan_clusters} clusters, {
        list(dbscan_labels).count(
            -1)} noise points"
    )
    print(f"✓ K-Means: {n_clusters} clusters, 0 noise points")
    print("✓ Saved comparison: dbscan_vs_kmeans.png")
    print("=" * 60 + "\n")


def main():
    print("=" * 60)
    print("DBSCAN CLUSTERING - SCIKIT-LEARN")
    print("=" * 60)

    # 1. Generate data
    print("\n📊 Generating data...")
    dataset_type = "moons"  # Try: 'moons', 'circles', 'blobs', 'mixed'
    X, y_true = generate_sample_data(dataset_type=dataset_type, n_samples=1000)
    print(f"Dataset: {dataset_type}")
    print(f"Shape: {X.shape}")

    # 2. Standardize
    print("\n🔧 Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Find optimal eps
    print("\n🔍 Finding optimal eps parameter...")
    eps = find_optimal_eps(X_scaled, min_samples=5, k=4)

    # 4. Train DBSCAN
    print(f"\n🚀 Training DBSCAN (eps={eps:.4f}, min_samples=5)...")
    dbscan, labels = train_dbscan(X_scaled, eps=eps, min_samples=5)

    # 5. Evaluate
    metrics = evaluate_clustering(X_scaled, labels)

    # 6. Visualize
    print("📈 Visualizing clusters...")
    visualize_clusters(X_scaled, labels, title=f"DBSCAN (eps={eps:.3f}, min_samples=5)")

    # 7. Parameter sensitivity
    print("🔬 Analyzing parameter sensitivity...")
    eps_range = np.linspace(0.1, 1.0, 5)
    min_samples_range = [3, 5, 10]
    parameter_sensitivity_analysis(X_scaled, eps_range, min_samples_range)

    # 8. Compare with K-Means
    print("⚖️  Comparing with K-Means...")
    compare_with_kmeans(X_scaled, n_clusters=2)

    print("\n✅ Analysis complete!")
    print("📁 Saved visualizations:")
    print("   - dbscan_eps_selection.png")
    print("   - dbscan_clusters.png")
    print("   - dbscan_parameter_analysis.png")
    print("   - dbscan_vs_kmeans.png")

    return dbscan, labels, metrics


if __name__ == "__main__":
    model, labels, metrics = main()
