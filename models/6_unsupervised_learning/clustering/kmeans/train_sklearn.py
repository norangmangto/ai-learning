"""
K-Means Clustering - Scikit-Learn Implementation
Unsupervised learning algorithm for partitioning data into K clusters
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def generate_sample_data(n_samples=1000, n_features=2, n_clusters=4):
    """Generate synthetic clustering data"""
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )
    return X, y_true


def load_real_data():
    """Load Iris dataset for clustering"""
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    return X, y_true, iris.feature_names


def find_optimal_k(X, k_range=(2, 11)):
    """
    Find optimal number of clusters using:
    - Elbow method (Within-Cluster Sum of Squares)
    - Silhouette score
    """
    inertias = []
    silhouette_scores = []

    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    ax1.plot(range(k_range[0], k_range[1]), inertias, "bo-")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)")
    ax1.set_title("Elbow Method")
    ax1.grid(True)

    # Silhouette plot
    ax2.plot(range(k_range[0], k_range[1]), silhouette_scores, "ro-")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("kmeans_optimal_k.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Find best K by silhouette score
    best_k = np.argmax(silhouette_scores) + k_range[0]
    print(f"\n📊 Optimal K by Silhouette Score: {best_k}")
    print(f"   Silhouette Score: {max(silhouette_scores):.4f}")

    return best_k


def train_kmeans(X, n_clusters=3, algorithm="lloyd"):
    """
    Train K-Means clustering model

    Parameters:
    -----------
    X : array-like
        Training data
    n_clusters : int
        Number of clusters
    algorithm : str
        'lloyd' (standard), 'elkan' (faster for dense data)
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",  # Smart initialization
        n_init=10,  # Run algorithm 10 times, pick best
        max_iter=300,
        algorithm=algorithm,
        random_state=42,
    )

    # Fit and predict
    labels = kmeans.fit_predict(X)

    return kmeans, labels


def evaluate_clustering(X, labels, model=None):
    """
    Evaluate clustering quality using multiple metrics
    """
    print("\n" + "=" * 60)
    print("CLUSTERING EVALUATION METRICS")
    print("=" * 60)

    # Silhouette Score: measures how similar an object is to its own cluster
    # Range: [-1, 1], higher is better
    silhouette = silhouette_score(X, labels)
    print(f"✓ Silhouette Score: {silhouette:.4f}")
    print(f"  (Range: -1 to 1, higher is better)")

    # Davies-Bouldin Index: average similarity between clusters
    # Lower is better
    db_index = davies_bouldin_score(X, labels)
    print(f"\n✓ Davies-Bouldin Index: {db_index:.4f}")
    print(f"  (Lower is better, 0 is perfect)")

    # Calinski-Harabasz Score: ratio of between-cluster to within-cluster dispersion
    # Higher is better
    ch_score = calinski_harabasz_score(X, labels)
    print(f"\n✓ Calinski-Harabasz Score: {ch_score:.2f}")
    print(f"  (Higher is better)")

    if model is not None:
        # Inertia: sum of squared distances to nearest cluster center
        print(f"\n✓ Inertia (WCSS): {model.inertia_:.2f}")
        print(f"  (Within-Cluster Sum of Squares, lower is better)")

        # Number of iterations to converge
        print(f"\n✓ Iterations to Converge: {model.n_iter_}")

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n✓ Cluster Sizes:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        print(f"  Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")

    print("=" * 60 + "\n")

    return {
        "silhouette": silhouette,
        "davies_bouldin": db_index,
        "calinski_harabasz": ch_score,
    }


def visualize_clusters(X, labels, centers=None, title="K-Means Clustering"):
    """
    Visualize clustering results (2D)
    """
    plt.figure(figsize=(10, 8))

    # Plot points
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6, s=50, edgecolors="k"
    )

    # Plot cluster centers if provided
    if centers is not None:
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
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
    plt.savefig("kmeans_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()


def predict_new_samples(model, X_new):
    """
    Predict cluster assignments for new data
    """
    labels = model.predict(X_new)
    distances = model.transform(X_new)  # Distance to each centroid

    print("\n" + "=" * 60)
    print("PREDICTIONS FOR NEW SAMPLES")
    print("=" * 60)
    for i, (label, dists) in enumerate(zip(labels, distances)):
        print(f"\nSample {i+1}:")
        print(f"  Assigned Cluster: {label}")
        print(f"  Distance to Centroid: {dists[label]:.4f}")
        print(f"  Distances to all centroids: {dists}")
    print("=" * 60 + "\n")

    return labels


def advanced_initialization_comparison(X, n_clusters=3):
    """
    Compare different initialization methods
    """
    init_methods = {
        "k-means++": "Smart initialization (default)",
        "random": "Random initialization",
    }

    results = {}

    print("\n" + "=" * 60)
    print("INITIALIZATION METHOD COMPARISON")
    print("=" * 60)

    for init_name, description in init_methods.items():
        kmeans = KMeans(
            n_clusters=n_clusters, init=init_name, n_init=10, random_state=42
        )
        kmeans.fit(X)

        silhouette = silhouette_score(X, kmeans.labels_)
        results[init_name] = {
            "silhouette": silhouette,
            "inertia": kmeans.inertia_,
            "iterations": kmeans.n_iter_,
        }

        print(f"\n{init_name} ({description}):")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        print(f"  Iterations: {kmeans.n_iter_}")

    print("=" * 60 + "\n")
    return results


def main():
    print("=" * 60)
    print("K-MEANS CLUSTERING - SCIKIT-LEARN")
    print("=" * 60)

    # 1. Generate or load data
    print("\n📊 Loading data...")
    # Option 1: Synthetic data (2D for easy visualization)
    X, y_true = generate_sample_data(n_samples=500, n_features=2, n_clusters=4)

    # Option 2: Real data (uncomment to use Iris dataset)
    # X, y_true, feature_names = load_real_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    # 2. Standardize features (important for K-Means!)
    print("\n🔧 Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Find optimal number of clusters
    print("\n🔍 Finding optimal number of clusters...")
    optimal_k = find_optimal_k(X_scaled, k_range=(2, 11))

    # 4. Train K-Means with optimal K
    print(f"\n🚀 Training K-Means with K={optimal_k}...")
    kmeans, labels = train_kmeans(X_scaled, n_clusters=optimal_k)

    # 5. Evaluate clustering
    metrics = evaluate_clustering(X_scaled, labels, kmeans)

    # 6. Visualize results (if 2D data)
    if X.shape[1] == 2:
        print("📈 Visualizing clusters...")
        visualize_clusters(
            X_scaled,
            labels,
            kmeans.cluster_centers_,
            title=f"K-Means Clustering (K={optimal_k})",
        )

    # 7. Compare initialization methods
    advanced_initialization_comparison(X_scaled, n_clusters=optimal_k)

    # 8. Predict on new samples
    print("🔮 Predicting new samples...")
    X_new = np.array([[0, 0], [2, 2], [-2, -2]])
    X_new_scaled = scaler.transform(X_new)
    predict_new_samples(kmeans, X_new_scaled)

    # 9. Get cluster centers in original space
    print("📍 Cluster Centers (original space):")
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, center in enumerate(centers_original):
        print(f"  Cluster {i}: {center}")

    print("\n✅ Training complete!")
    print("📁 Saved visualizations:")
    print("   - kmeans_optimal_k.png")
    print("   - kmeans_clusters.png")

    return kmeans, labels, metrics


if __name__ == "__main__":
    model, labels, metrics = main()
