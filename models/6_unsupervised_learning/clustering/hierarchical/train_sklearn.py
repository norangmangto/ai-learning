"""
Hierarchical Clustering with Scikit-learn

Hierarchical clustering builds a tree of clusters (dendrogram) through either:
- Agglomerative (bottom-up): Each point starts as a cluster, then merges
- Divisive (top-down): All points start in one cluster, then splits

This implementation focuses on Agglomerative Clustering with:
- Multiple linkage methods (ward, complete, average, single)
- Dendrogram visualization
- Optimal cluster number selection
- Comparison with K-Means
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage


def generate_sample_data(n_samples=500, dataset_type="blobs"):
    """
    Generate sample datasets for clustering.

    Args:
        n_samples: Number of samples to generate
        dataset_type: 'blobs', 'moons', or 'circles'

    Returns:
        X: Feature matrix
        y_true: True labels (for evaluation only)
    """
    if dataset_type == "blobs":
        X, y_true = make_blobs(
            n_samples=n_samples, centers=4, cluster_std=1.0, random_state=42
        )
    elif dataset_type == "moons":
        X, y_true = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    else:
        from sklearn.datasets import make_circles

        X, y_true = make_circles(
            n_samples=n_samples, noise=0.05, factor=0.5, random_state=42
        )

    return X, y_true


def plot_dendrogram(X, linkage_method="ward", max_display=30):
    """
    Create and plot a dendrogram.

    Args:
        X: Feature matrix (scaled)
        linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
        max_display: Maximum number of samples to display

    Returns:
        Z: Linkage matrix
    """
    # Compute linkage matrix
    Z = linkage(X, method=linkage_method)

    # Plot dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=max_display)
    plt.title(f"Hierarchical Clustering Dendrogram ({linkage_method} linkage)")
    plt.xlabel("Sample Index or (Cluster Size)")
    plt.ylabel("Distance")
    plt.axhline(
        y=np.median(Z[-10:, 2]), color="r", linestyle="--", label="Suggested Cut Height"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("hierarchical_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    return Z


def find_optimal_clusters(X, max_clusters=10, linkage_method="ward"):
    """
    Find optimal number of clusters using multiple metrics.

    Args:
        X: Feature matrix (scaled)
        max_clusters: Maximum number of clusters to try
        linkage_method: Linkage method to use

    Returns:
        optimal_k: Suggested number of clusters
    """
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_scores = []

    k_range = range(2, max_clusters + 1)

    for k in k_range:
        # Fit hierarchical clustering
        clusterer = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = clusterer.fit_predict(X)

        # Calculate metrics
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        calinski_scores.append(calinski_harabasz_score(X, labels))

    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Silhouette Score (higher is better)
    axes[0].plot(k_range, silhouette_scores, "bo-")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Silhouette Score vs K")
    axes[0].grid(True, alpha=0.3)

    # Davies-Bouldin Index (lower is better)
    axes[1].plot(k_range, davies_bouldin_scores, "ro-")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Davies-Bouldin Index")
    axes[1].set_title("Davies-Bouldin Index vs K")
    axes[1].grid(True, alpha=0.3)

    # Calinski-Harabasz Index (higher is better)
    axes[2].plot(k_range, calinski_scores, "go-")
    axes[2].set_xlabel("Number of Clusters")
    axes[2].set_ylabel("Calinski-Harabasz Index")
    axes[2].set_title("Calinski-Harabasz Index vs K")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hierarchical_optimal_k.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Suggest optimal k based on silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters (based on Silhouette Score): {optimal_k}")
    print(f"Silhouette Score: {max(silhouette_scores):.4f}")

    return optimal_k


def compare_linkage_methods(X, n_clusters=4):
    """
    Compare different linkage methods.

    Args:
        X: Feature matrix (scaled)
        n_clusters: Number of clusters
    """
    linkage_methods = ["ward", "complete", "average", "single"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    results = []

    for idx, method in enumerate(linkage_methods):
        # Fit clustering
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = clusterer.fit_predict(X)

        # Calculate metrics
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)

        results.append(
            {
                "method": method,
                "silhouette": sil_score,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
            }
        )

        # Visualize
        axes[idx].scatter(
            X[:, 0],
            X[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )
        axes[idx].set_title(
            f"{method.capitalize()} Linkage\n" f"Silhouette: {sil_score:.3f}"
        )
        axes[idx].set_xlabel("Feature 1")
        axes[idx].set_ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig("hierarchical_linkage_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print comparison table
    print("\n" + "=" * 70)
    print("Linkage Method Comparison")
    print("=" * 70)
    print(
        f"{'Method':<15} {'Silhouette':<15} {'Davies-Bouldin':<15} {'Calinski-H':<15}"
    )
    print("-" * 70)
    for result in results:
        print(
            f"{result['method']:<15} "
            f"{result['silhouette']:<15.4f} "
            f"{result['davies_bouldin']:<15.4f} "
            f"{result['calinski_harabasz']:<15.2f}"
        )
    print("=" * 70)


def train_hierarchical_clustering(
    X, n_clusters=4, linkage="ward", distance_threshold=None
):
    """
    Train hierarchical clustering model.

    Args:
        X: Feature matrix (scaled)
        n_clusters: Number of clusters (ignored if distance_threshold is set)
        linkage: Linkage method ('ward', 'complete', 'average', 'single')
        distance_threshold: Distance threshold for forming flat clusters

    Returns:
        model: Trained clustering model
        labels: Cluster labels
    """
    # Create and fit model
    if distance_threshold is not None:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage=linkage
        )
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    labels = model.fit_predict(X)

    print(f"\nHierarchical Clustering Results:")
    print(f"Number of clusters formed: {len(np.unique(labels))}")
    print(f"Linkage method: {linkage}")

    return model, labels


def evaluate_clustering(X, labels):
    """
    Evaluate clustering performance using multiple metrics.

    Args:
        X: Feature matrix
        labels: Cluster labels

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if len(np.unique(labels)) < 2:
        print("Cannot evaluate: Less than 2 clusters formed")
        return None

    # Calculate metrics
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    metrics = {
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "calinski_harabasz_index": calinski_harabasz,
        "n_clusters": len(np.unique(labels)),
    }

    print("\n" + "=" * 50)
    print("Clustering Evaluation Metrics")
    print("=" * 50)
    print(f"Number of Clusters: {metrics['n_clusters']}")
    print(
        f"Silhouette Score: {
        silhouette:.4f} (higher is better, range: [-1, 1])"
    )
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f} (higher is better)")
    print("=" * 50)

    return metrics


def visualize_clusters(X, labels, title="Hierarchical Clustering Results"):
    """
    Visualize clustering results.

    Args:
        X: Feature matrix
        labels: Cluster labels
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    # Plot clusters
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )

    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hierarchical_clusters.png", dpi=300, bbox_inches="tight")
    plt.show()


def analyze_cluster_sizes(labels):
    """
    Analyze and visualize cluster size distribution.

    Args:
        labels: Cluster labels
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Plot distribution
    plt.figure(figsize=(10, 5))
    plt.bar(unique_labels, counts, color="steelblue", edgecolor="black")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Points")
    plt.title("Cluster Size Distribution")
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        plt.text(label, count, str(count), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("hierarchical_cluster_sizes.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nCluster Size Distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} points ({count/len(labels)*100:.1f}%)")


def compare_with_distance_threshold(X, thresholds=[5, 10, 15, 20]):
    """
    Compare clustering with different distance thresholds.

    Args:
        X: Feature matrix (scaled)
        thresholds: List of distance thresholds to try
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, threshold in enumerate(thresholds):
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, linkage="ward"
        )
        labels = model.fit_predict(X)
        n_clusters = len(np.unique(labels))

        axes[idx].scatter(
            X[:, 0],
            X[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )
        axes[idx].set_title(f"Threshold: {threshold}, Clusters: {n_clusters}")
        axes[idx].set_xlabel("Feature 1")
        axes[idx].set_ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig("hierarchical_threshold_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Hierarchical Clustering with Scikit-learn")
    print("=" * 70)

    # 1. Generate sample data
    print("\n1. Generating sample data...")
    X, y_true = generate_sample_data(n_samples=400, dataset_type="blobs")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset shape: {X_scaled.shape}")
    print(f"Number of true clusters: {len(np.unique(y_true))}")

    # 2. Plot dendrogram
    print("\n2. Creating dendrogram...")
    plot_dendrogram(X_scaled, linkage_method="ward")

    # 3. Find optimal number of clusters
    print("\n3. Finding optimal number of clusters...")
    optimal_k = find_optimal_clusters(X_scaled, max_clusters=10, linkage_method="ward")

    # 4. Compare linkage methods
    print("\n4. Comparing linkage methods...")
    compare_linkage_methods(X_scaled, n_clusters=optimal_k)

    # 5. Train with optimal parameters
    print("\n5. Training hierarchical clustering with optimal parameters...")
    model, labels = train_hierarchical_clustering(
        X_scaled, n_clusters=optimal_k, linkage="ward"
    )

    # 6. Evaluate clustering
    print("\n6. Evaluating clustering performance...")
    evaluate_clustering(X_scaled, labels)

    # 7. Visualize results
    print("\n7. Visualizing clustering results...")
    visualize_clusters(
        X_scaled, labels, title=f"Hierarchical Clustering (Ward, k={optimal_k})"
    )

    # 8. Analyze cluster sizes
    print("\n8. Analyzing cluster size distribution...")
    analyze_cluster_sizes(labels)

    # 9. Compare with distance threshold approach
    print("\n9. Comparing with distance threshold approach...")
    compare_with_distance_threshold(X_scaled, thresholds=[3, 5, 7, 10])

    # 10. Try on different dataset type (non-spherical)
    print("\n10. Testing on non-spherical data (moons)...")
    X_moons, _ = generate_sample_data(n_samples=400, dataset_type="moons")
    X_moons_scaled = scaler.fit_transform(X_moons)

    compare_linkage_methods(X_moons_scaled, n_clusters=2)

    print("\n" + "=" * 70)
    print("Hierarchical Clustering Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- Ward linkage: Best for spherical clusters (minimizes variance)")
    print("- Complete linkage: Better for elongated clusters")
    print("- Average linkage: Compromise between ward and complete")
    print("- Single linkage: Can form long chains (often not recommended)")
    print("\nWhen to use Hierarchical Clustering:")
    print("✓ Need to understand cluster hierarchy")
    print("✓ Don't know number of clusters in advance")
    print("✓ Want to visualize relationships (dendrogram)")
    print("✓ Small to medium datasets (computational cost: O(n²))")


if __name__ == "__main__":
    main()
