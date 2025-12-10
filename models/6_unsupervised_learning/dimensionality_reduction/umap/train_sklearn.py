"""
UMAP (Uniform Manifold Approximation and Projection) with umap-learn

UMAP is a modern dimensionality reduction technique that:
- Faster than t-SNE (especially on large datasets)
- Preserves both local AND global structure better
- Can embed new data (has transform method)
- More stable and deterministic results

This implementation includes:
- Parameter optimization (n_neighbors, min_dist)
- Comparison with PCA and t-SNE
- Supervised UMAP variant
- Best practices and hyperparameter guidance

Installation required: pip install umap-learn
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import time

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed. Install with: pip install umap-learn")


def load_sample_data(dataset='digits'):
    """
    Load sample high-dimensional dataset.

    Args:
        dataset: 'digits' or 'synthetic'

    Returns:
        X: Feature matrix
        y: Labels
    """
    if dataset == 'digits':
        digits = load_digits()
        X, y = digits.data, digits.target
        print(f"Loaded digits dataset: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        X, y = make_classification(n_samples=2000, n_features=50,
                                   n_informative=30, n_redundant=10,
                                   n_classes=8, random_state=42)
        print(f"Generated synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y


def compare_n_neighbors(X, y, n_neighbors_list=[5, 15, 30, 50, 100]):
    """
    Compare UMAP results with different n_neighbors values.

    n_neighbors: Controls local vs global structure
    - Low values (5-15): Focus on local structure, tight clusters
    - Medium values (15-30): Balanced (recommended)
    - High values (30-100): Focus on global structure

    Args:
        X: Feature matrix (scaled)
        y: Labels
        n_neighbors_list: List of n_neighbors values to test
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping this analysis.")
        return

     # Use subset for faster visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    print("\n" + "="*70)
    print("Comparing Different n_neighbors Values")
    print("="*70)

    for idx, n_neighbors in enumerate(n_neighbors_list):
        if idx >= len(axes):
            break

        print(f"\nTesting n_neighbors = {n_neighbors}...")
        start_time = time.time()

        # Apply UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42, verbose=False)
        X_umap = reducer.fit_transform(X)

        elapsed_time = time.time() - start_time

        # Calculate silhouette score
        sil_score = silhouette_score(X_umap, y)

         # Visualize
        axes[idx].scatter(X_umap[:, 0], X_umap[:, 1],
                         c=y, cmap='tab10', alpha=0.7,
                         edgecolors='black', linewidth=0.3, s=20)
        axes[idx].set_title(f'n_neighbors = {n_neighbors}\n'
                          f'Silhouette: {sil_score:.3f}, Time: {elapsed_time:.1f}s',
                          fontsize=11)
        axes[idx].set_xlabel('UMAP 1', fontsize=10)
        axes[idx].set_ylabel('UMAP 2', fontsize=10)

        print(f"  Silhouette Score: {sil_score:.4f}")
        print(f"  Time: {elapsed_time:.2f} seconds")

    # Hide unused subplot
    if len(n_neighbors_list) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('umap_neighbors_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("="*70)
    print("Recommendation: Start with n_neighbors=15-30")


def compare_min_dist(X, y, min_dist_list=[0.0, 0.1, 0.25, 0.5, 0.8]):
    """
    Compare UMAP results with different min_dist values.

    min_dist: Controls how tightly points are packed
    - 0.0: Tight clusters, focus on structure
    - 0.1-0.3: Balanced (recommended)
    - 0.5-1.0: Spread out, preserves topology

    Args:
        X: Feature matrix (scaled)
        y: Labels
        min_dist_list: List of min_dist values to test
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping this analysis.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    print("\n" + "="*70)
    print("Comparing Different min_dist Values")
    print("="*70)

    for idx, min_dist in enumerate(min_dist_list):
        if idx >= len(axes):
            break

        print(f"\nTesting min_dist = {min_dist}...")
        start_time = time.time()

        # Apply UMAP
        reducer = umap.UMAP(min_dist=min_dist, random_state=42, verbose=False)
        X_umap = reducer.fit_transform(X)

        elapsed_time = time.time() - start_time

        # Calculate silhouette score
        sil_score = silhouette_score(X_umap, y)

         # Visualize
        axes[idx].scatter(X_umap[:, 0], X_umap[:, 1],
                         c=y, cmap='tab10', alpha=0.7,
                         edgecolors='black', linewidth=0.3, s=20)
        axes[idx].set_title(f'min_dist = {min_dist}\n'
                          f'Silhouette: {sil_score:.3f}, Time: {elapsed_time:.1f}s',
                          fontsize=11)
        axes[idx].set_xlabel('UMAP 1', fontsize=10)
        axes[idx].set_ylabel('UMAP 2', fontsize=10)

        print(f"  Silhouette Score: {sil_score:.4f}")
        print(f"  Time: {elapsed_time:.2f} seconds")

    # Hide unused subplot
    if len(min_dist_list) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('umap_mindist_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("="*70)
    print("Recommendation: Use min_dist=0.1 for tight clusters, 0.3 for spread")


def train_umap(X, y=None, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    Train UMAP with specified parameters.

    Args:
        X: Feature matrix (scaled)
        y: Labels (optional, for supervised UMAP)
        n_neighbors: Number of neighbors
        min_dist: Minimum distance between points
        n_components: Number of components (default: 2)

    Returns:
        X_umap: Reduced representation
        reducer: Fitted UMAP model
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Please install: pip install umap-learn")
        return None, None

    print(f"\nTraining UMAP...")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  min_dist: {min_dist}")
    print(f"  n_components: {n_components}")

    start_time = time.time()

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        verbose=False
    )

    X_umap = reducer.fit_transform(X)

    elapsed_time = time.time() - start_time

    print(f"\nUMAP Training Complete!")
    print(f"  Time taken: {elapsed_time:.2f} seconds")

    if y is not None:
        sil_score = silhouette_score(X_umap, y)
        print(f"  Silhouette Score: {sil_score:.4f}")

    return X_umap, reducer


def compare_with_pca_tsne(X, y):
    """
    Compare UMAP with PCA and t-SNE.

    Args:
        X: Feature matrix (scaled)
        y: Labels
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping comparison.")
        return

    print("\n" + "="*70)
    print("Comparing UMAP, PCA, and t-SNE")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # PCA
    print("\n1. Computing PCA...")
    start_time = time.time()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    pca_time = time.time() - start_time
    pca_sil = silhouette_score(X_pca, y)

    axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                   c=y, cmap='tab10', alpha=0.7,
                   edgecolors='black', linewidth=0.3, s=20)
    axes[0].set_title(f'PCA\nSilhouette: {pca_sil:.3f}, '
                     f'Variance: {pca.explained_variance_ratio_.sum():.2%}\n'
                     f'Time: {pca_time:.2f}s',
                     fontsize=11)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # t-SNE
    print("2. Computing t-SNE...")
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=30, init='pca',
               random_state=42, verbose=0)
    X_tsne = tsne.fit_transform(X)
    tsne_time = time.time() - start_time
    tsne_sil = silhouette_score(X_tsne, y)

    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                   c=y, cmap='tab10', alpha=0.7,
                   edgecolors='black', linewidth=0.3, s=20)
    axes[1].set_title(f't-SNE\nSilhouette: {tsne_sil:.3f}\n'
                     f'Time: {tsne_time:.2f}s',
                     fontsize=11)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    # UMAP
    print("3. Computing UMAP...")
    start_time = time.time()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, verbose=False)
    X_umap = reducer.fit_transform(X)
    umap_time = time.time() - start_time
    umap_sil = silhouette_score(X_umap, y)

    axes[2].scatter(X_umap[:, 0], X_umap[:, 1],
                   c=y, cmap='tab10', alpha=0.7,
                   edgecolors='black', linewidth=0.3, s=20)
    axes[2].set_title(f'UMAP\nSilhouette: {umap_sil:.3f}\n'
                     f'Time: {umap_time:.2f}s',
                     fontsize=11)
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.savefig('umap_vs_pca_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print comparison table
    print("\n" + "="*70)
    print("Method Comparison")
    print("="*70)
    print(f"{'Method':<10} {'Silhouette':<15} {'Time (s)':<12} {'Notes'}")
    print("-"*70)
    print(f"{'PCA':<10} {pca_sil:<15.4f} {pca_time:<12.2f} {'Fast, linear'}")
    print(f"{'t-SNE':<10} {tsne_sil:<15.4f} {tsne_time:<12.2f} {'Slow, local focus'}")
    print(f"{'UMAP':<10} {umap_sil:<15.4f} {umap_time:<12.2f} {'Fast, balanced'}")
    print("="*70)


def supervised_umap_comparison(X, y):
    """
    Compare unsupervised vs supervised UMAP.

    Supervised UMAP uses labels to guide the embedding.

    Args:
        X: Feature matrix (scaled)
        y: Labels
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping supervised UMAP.")
        return

    print("\n" + "="*70)
    print("Unsupervised vs Supervised UMAP")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Unsupervised UMAP
    print("\n1. Training unsupervised UMAP...")
    start_time = time.time()
    unsupervised = umap.UMAP(n_neighbors=15, min_dist=0.1,
                            random_state=42, verbose=False)
    X_unsup = unsupervised.fit_transform(X)
    unsup_time = time.time() - start_time
    unsup_sil = silhouette_score(X_unsup, y)

    axes[0].scatter(X_unsup[:, 0], X_unsup[:, 1],
                   c=y, cmap='tab10', alpha=0.7,
                   edgecolors='black', linewidth=0.3, s=20)
    axes[0].set_title(f'Unsupervised UMAP\n'
                     f'Silhouette: {unsup_sil:.3f}, Time: {unsup_time:.2f}s',
                     fontsize=12)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')

    # Supervised UMAP
    print("2. Training supervised UMAP (using labels)...")
    start_time = time.time()
    supervised = umap.UMAP(n_neighbors=15, min_dist=0.1,
                          random_state=42, verbose=False)
    X_sup = supervised.fit_transform(X, y=y)
    sup_time = time.time() - start_time
    sup_sil = silhouette_score(X_sup, y)

    axes[1].scatter(X_sup[:, 0], X_sup[:, 1],
                   c=y, cmap='tab10', alpha=0.7,
                   edgecolors='black', linewidth=0.3, s=20)
    axes[1].set_title(f'Supervised UMAP\n'
                     f'Silhouette: {sup_sil:.3f}, Time: {sup_time:.2f}s',
                     fontsize=12)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.savefig('umap_supervised_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*70)
    print(f"Unsupervised Silhouette: {unsup_sil:.4f}")
    print(f"Supervised Silhouette: {sup_sil:.4f}")
    print(f"Improvement: {(sup_sil - unsup_sil):.4f}")
    print("\nSupervised UMAP:")
    print("✓ Better class separation")
    print("✓ Use when labels are available")
    print("✗ Can overfit to labels")
    print("="*70)


def test_transform_capability(X, y):
    """
    Demonstrate UMAP's ability to transform new data.

    Unlike t-SNE, UMAP can embed new points without retraining.

    Args:
        X: Feature matrix (scaled)
        y: Labels
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping transform test.")
        return

    print("\n" + "="*70)
    print("Testing Transform Capability (New Data Embedding)")
    print("="*70)

    # Split data
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Fit on training data
    print("\n1. Fitting UMAP on training data...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, verbose=False)
    X_train_umap = reducer.fit_transform(X_train)

    # Transform test data
    print("2. Transforming test data (without retraining)...")
    X_test_umap = reducer.transform(X_test)

    # Visualize
    plt.figure(figsize=(12, 8))
    plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1],
               c=y_train, cmap='tab10', alpha=0.5,
               edgecolors='black', linewidth=0.3, s=30, label='Training data')
    plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1],
               c=y_test, cmap='tab10', alpha=0.9, marker='^',
               edgecolors='red', linewidth=1, s=100, label='Test data (transformed)')

    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.title('UMAP Transform Capability: Embedding New Data', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('umap_transform_capability.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nKey Advantage:")
    print("✓ UMAP can transform new data without retraining")
    print("✓ Useful for production ML pipelines")
    print("✗ t-SNE cannot do this (must retrain on all data)")
    print("="*70)


def main():
    """Main execution function."""
    print("="*70)
    print("UMAP (Uniform Manifold Approximation and Projection)")
    print("="*70)

    if not UMAP_AVAILABLE:
        print("\nERROR: umap-learn is not installed!")
        print("Please install it with: pip install umap-learn")
        print("\nOr add it to your requirements.txt:")
        print("umap-learn>=0.5.0")
        return

    # 1. Load data
    print("\n1. Loading dataset...")
    X, y = load_sample_data(dataset='digits')

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Compare n_neighbors values
    print("\n2. Comparing different n_neighbors values...")
    compare_n_neighbors(X_scaled, y, n_neighbors_list=[5, 15, 30, 50, 100])

    # 3. Compare min_dist values
    print("\n3. Comparing different min_dist values...")
    compare_min_dist(X_scaled, y, min_dist_list=[0.0, 0.1, 0.25, 0.5, 0.8])

    # 4. Train UMAP with optimal parameters
    print("\n4. Training UMAP with optimal parameters...")
    X_umap, reducer = train_umap(X_scaled, y, n_neighbors=15, min_dist=0.1)

    # 5. Compare with PCA and t-SNE
    print("\n5. Comparing UMAP with PCA and t-SNE...")
    compare_with_pca_tsne(X_scaled, y)

    # 6. Supervised vs unsupervised UMAP
    print("\n6. Comparing supervised vs unsupervised UMAP...")
    supervised_umap_comparison(X_scaled, y)

    # 7. Test transform capability
    print("\n7. Testing transform capability on new data...")
    test_transform_capability(X_scaled, y)

    print("\n" + "="*70)
    print("UMAP Analysis Complete!")
    print("="*70)
    print("\nKey Advantages of UMAP:")
    print("✓ Faster than t-SNE (especially on large datasets)")
    print("✓ Preserves both local AND global structure")
    print("✓ Can transform new data (has .transform() method)")
    print("✓ More stable and reproducible")
    print("✓ Better mathematical foundation")

    print("\nWhen to use UMAP:")
    print("✓ Large datasets (> 10,000 samples)")
    print("✓ Need to embed new data points")
    print("✓ Want both local and global structure")
    print("✓ Production ML pipelines")

    print("\nRecommended Parameters:")
    print("- n_neighbors: 15-30 (balance local/global)")
    print("- min_dist: 0.1 (tight clusters) to 0.3 (spread out)")
    print("- metric: 'euclidean' (default), 'cosine' for text data")

    print("\nUMAP vs t-SNE vs PCA:")
    print("- PCA: Fastest, linear, global structure")
    print("- t-SNE: Slowest, best local structure, no transform")
    print("- UMAP: Fast, balanced, can transform new data ⭐")


if __name__ == "__main__":
    main()
