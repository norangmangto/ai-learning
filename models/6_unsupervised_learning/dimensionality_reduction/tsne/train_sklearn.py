"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) with Scikit-learn

t-SNE is a non-linear dimensionality reduction technique optimized for:
- Visualization of high-dimensional data
- Preserving local structure and clusters
- Revealing patterns not visible with linear methods like PCA

This implementation includes:
- Perplexity tuning
- Multiple initialization methods
- Comparison with PCA
- Best practices for parameter selection
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import time


def load_sample_data(dataset='digits'):
    """
    Load sample high-dimensional dataset.

    Args:
        dataset: 'digits' or 'mnist_sample'

    Returns:
        X: Feature matrix
        y: Labels
    """
    if dataset == 'digits':
        # Load digits dataset (64 features, 1797 samples)
        digits = load_digits()
        X, y = digits.data, digits.target
        print(f"Loaded digits dataset: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        # Load a sample of MNIST
        print("Loading MNIST sample (this may take a moment)...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        # Take a smaller sample for faster processing
        indices = np.random.RandomState(42).choice(len(mnist.data), 5000, replace=False)
        X = np.array(mnist.data.iloc[indices] if hasattr(mnist.data, 'iloc') else mnist.data[indices])
        y = np.array(mnist.target.iloc[indices] if hasattr(mnist.target, 'iloc') else mnist.target[indices])
        y = y.astype(int)
        print(f"Loaded MNIST sample: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y


def compare_perplexity(X, y, perplexity_values=[5, 30, 50, 100]):
    """
    Compare t-SNE results with different perplexity values.

    Perplexity: Balance between local and global aspects of data
    - Low perplexity (5-15): Focus on very local structure
    - Medium perplexity (30-50): Recommended default
    - High perplexity (50-100): Focus more on global structure

    Args:
        X: Feature matrix (scaled)
        y: Labels
        perplexity_values: List of perplexity values to test
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    print("\n" + "="*70)
    print("Comparing Different Perplexity Values")
    print("="*70)

    for idx, perplexity in enumerate(perplexity_values):
        print(f"\nTesting perplexity = {perplexity}...")
        start_time = time.time()

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity,
                   random_state=42, n_iter=1000, verbose=0)
        X_tsne = tsne.fit_transform(X)

        elapsed_time = time.time() - start_time

        # Calculate silhouette score
        sil_score = silhouette_score(X_tsne, y)

         # Visualize
        axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=y, cmap='tab10', alpha=0.7,
                         edgecolors='black', linewidth=0.3, s=20)
        axes[idx].set_title(f'Perplexity = {perplexity}\n'
                          f'Silhouette: {sil_score:.3f}, Time: {elapsed_time:.1f}s',
                          fontsize=12)
        axes[idx].set_xlabel('t-SNE 1', fontsize=11)
        axes[idx].set_ylabel('t-SNE 2', fontsize=11)

        print(f"  Silhouette Score: {sil_score:.4f}")
        print(f"  Time: {elapsed_time:.2f} seconds")

    plt.tight_layout()
    plt.savefig('tsne_perplexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("="*70)


def compare_initialization_methods(X, y, perplexity=30):
    """
    Compare different initialization methods for t-SNE.

    Args:
        X: Feature matrix (scaled)
        y: Labels
        perplexity: Perplexity value
    """
    init_methods = ['random', 'pca']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    print("\n" + "="*70)
    print("Comparing Initialization Methods")
    print("="*70)

    for idx, init_method in enumerate(init_methods):
        print(f"\nTesting initialization: {init_method}...")
        start_time = time.time()

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, init=init_method,
                   random_state=42, n_iter=1000, verbose=0)
        X_tsne = tsne.fit_transform(X)

        elapsed_time = time.time() - start_time

        # Calculate silhouette score
        sil_score = silhouette_score(X_tsne, y)

         # Visualize
        axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=y, cmap='tab10', alpha=0.7,
                         edgecolors='black', linewidth=0.3, s=20)
        axes[idx].set_title(f'Initialization: {init_method.upper()}\n'
                          f'Silhouette: {sil_score:.3f}, Time: {elapsed_time:.1f}s',
                          fontsize=12)
        axes[idx].set_xlabel('t-SNE 1', fontsize=11)
        axes[idx].set_ylabel('t-SNE 2', fontsize=11)

        print(f"  Silhouette Score: {sil_score:.4f}")
        print(f"  Time: {elapsed_time:.2f} seconds")

    plt.tight_layout()
    plt.savefig('tsne_initialization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("="*70)
    print("\nRecommendation: Use 'pca' initialization for faster convergence")


def train_tsne(X, y=None, perplexity=30, n_iter=1000):
    """
    Train t-SNE with optimal parameters.

    Args:
        X: Feature matrix (scaled)
        y: Labels (optional, for visualization)
        perplexity: Perplexity value (default: 30)
        n_iter: Number of iterations (default: 1000)

    Returns:
        X_tsne: Reduced representation
        tsne: Fitted t-SNE model
    """
    print(f"\nTraining t-SNE...")
    print(f"  Perplexity: {perplexity}")
    print(f"  Iterations: {n_iter}")
    print(f"  Initialization: PCA")

    start_time = time.time()

    tsne = TSNE(n_components=2,
               perplexity=perplexity,
               n_iter=n_iter,
               init='pca',
               random_state=42,
               verbose=0)

    X_tsne = tsne.fit_transform(X)

    elapsed_time = time.time() - start_time

    print(f"\nt-SNE Training Complete!")
    print(f"  Time taken: {elapsed_time:.2f} seconds")
    print(f"  Final KL divergence: {tsne.kl_divergence_:.4f}")

    if y is not None:
        sil_score = silhouette_score(X_tsne, y)
        print(f"  Silhouette Score: {sil_score:.4f}")

    return X_tsne, tsne


def compare_with_pca(X, y, perplexity=30):
    """
    Compare t-SNE with PCA visualization.

    Args:
        X: Feature matrix (scaled)
        y: Labels
        perplexity: Perplexity for t-SNE
    """
    print("\nComparing t-SNE with PCA...")

    # PCA
    print("  Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # t-SNE
    print("  Computing t-SNE...")
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity,
               init='pca', random_state=42, verbose=0)
    X_tsne = tsne.fit_transform(X)
    tsne_time = time.time() - start_time

    # Calculate metrics
    pca_sil = silhouette_score(X_pca, y)
    tsne_sil = silhouette_score(X_tsne, y)

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=y, cmap='tab10', alpha=0.7,
                              edgecolors='black', linewidth=0.3, s=20)
    axes[0].set_title(f'PCA (2 components)\n'
                     f'Silhouette: {pca_sil:.3f}, '
                     f'Variance: {pca.explained_variance_ratio_.sum():.2%}',
                     fontsize=12)
    axes[0].set_xlabel('PC1', fontsize=11)
    axes[0].set_ylabel('PC2', fontsize=11)
    plt.colorbar(scatter1, ax=axes[0], label='Class')

    # t-SNE
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                              c=y, cmap='tab10', alpha=0.7,
                              edgecolors='black', linewidth=0.3, s=20)
    axes[1].set_title(f't-SNE (perplexity={perplexity})\n'
                     f'Silhouette: {tsne_sil:.3f}, Time: {tsne_time:.1f}s',
                     fontsize=12)
    axes[1].set_xlabel('t-SNE 1', fontsize=11)
    axes[1].set_ylabel('t-SNE 2', fontsize=11)
    plt.colorbar(scatter2, ax=axes[1], label='Class')

    plt.tight_layout()
    plt.savefig('tsne_vs_pca.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*70)
    print("PCA vs t-SNE Comparison")
    print("="*70)
    print(f"PCA Silhouette Score: {pca_sil:.4f}")
    print(f"t-SNE Silhouette Score: {tsne_sil:.4f}")
    print(f"\nt-SNE took {tsne_time:.2f}s (PCA is much faster)")
    print("\nWhen to use each:")
    print("PCA: Fast, linear, preserves global structure, interpretable")
    print("t-SNE: Slow, non-linear, preserves local structure, better clusters")
    print("="*70)


def visualize_with_labels(X_tsne, y, class_names=None):
    """
    Create detailed visualization with class labels.

    Args:
        X_tsne: t-SNE reduced features
        y: Labels
        class_names: Optional class names for legend
    """
    plt.figure(figsize=(14, 10))

    unique_labels = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = y == label
        label_name = class_names[label] if class_names else f'Class {label}'
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[color], label=label_name, alpha=0.7,
                   edgecolors='black', linewidth=0.3, s=30)

    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('t-SNE Visualization with Class Labels', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_labeled_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_convergence(X, perplexity=30, n_iter_list=[250, 500, 1000, 2000]):
    """
    Analyze convergence behavior with different iteration counts.

    Args:
        X: Feature matrix (scaled)
        perplexity: Perplexity value
        n_iter_list: List of iteration counts to test
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    print("\n" + "="*70)
    print("Analyzing Convergence (Different Iteration Counts)")
    print("="*70)

    for idx, n_iter in enumerate(n_iter_list):
        print(f"\nTesting n_iter = {n_iter}...")
        start_time = time.time()

        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                   init='pca', random_state=42, verbose=0)
        X_tsne = tsne.fit_transform(X)

        elapsed_time = time.time() - start_time

        axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=15)
        axes[idx].set_title(f'Iterations = {n_iter}\n'
                          f'KL Divergence: {tsne.kl_divergence_:.3f}, '
                          f'Time: {elapsed_time:.1f}s',
                          fontsize=12)
        axes[idx].set_xlabel('t-SNE 1', fontsize=11)
        axes[idx].set_ylabel('t-SNE 2', fontsize=11)

        print(f"  KL Divergence: {tsne.kl_divergence_:.4f}")
        print(f"  Time: {elapsed_time:.2f} seconds")

    plt.tight_layout()
    plt.savefig('tsne_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("="*70)
    print("Recommendation: Use at least 1000 iterations for good results")


def main():
    """Main execution function."""
    print("="*70)
    print("t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    print("="*70)

    # 1. Load data
    print("\n1. Loading dataset...")
    X, y = load_sample_data(dataset='digits')

    # Scale features (important for t-SNE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # For large datasets, pre-reduce dimensions with PCA
    if X_scaled.shape[1] > 50:
        print("  Pre-reducing dimensions with PCA (recommended for > 50 features)...")
        pca = PCA(n_components=50, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"  Reduced to {X_scaled.shape[1]} dimensions")

    # 2. Compare perplexity values
    print("\n2. Comparing different perplexity values...")
    compare_perplexity(X_scaled, y, perplexity_values=[5, 30, 50, 100])

    # 3. Compare initialization methods
    print("\n3. Comparing initialization methods...")
    compare_initialization_methods(X_scaled, y, perplexity=30)

    # 4. Train t-SNE with optimal parameters
    print("\n4. Training t-SNE with optimal parameters...")
    X_tsne, tsne_model = train_tsne(X_scaled, y, perplexity=30, n_iter=1000)

    # 5. Compare with PCA
    print("\n5. Comparing t-SNE with PCA...")
    compare_with_pca(X_scaled, y, perplexity=30)

    # 6. Create detailed visualization
    print("\n6. Creating detailed visualization with labels...")
    class_names = [str(i) for i in range(10)]  # Digit names
    visualize_with_labels(X_tsne, y, class_names=class_names)

    # 7. Analyze convergence
    print("\n7. Analyzing convergence behavior...")
    analyze_convergence(X_scaled, perplexity=30, n_iter_list=[250, 500, 1000, 2000])

    print("\n" + "="*70)
    print("t-SNE Analysis Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("✓ t-SNE excels at revealing cluster structure")
    print("✓ Better than PCA for visualizing complex patterns")
    print("✓ Perplexity controls local vs global structure")
    print("✓ Results can vary between runs (use random_state)")

    print("\nBest Practices:")
    print("1. Pre-reduce dimensions with PCA if > 50 features")
    print("2. Use perplexity between 5-50 (30 is good default)")
    print("3. Use PCA initialization for faster convergence")
    print("4. Run at least 1000 iterations")
    print("5. Always scale your data first")

    print("\nWhen to use t-SNE:")
    print("✓ Visualizing high-dimensional data")
    print("✓ Exploring cluster structure")
    print("✓ Presentations and publications")
    print("✓ Anomaly detection visualization")

    print("\nLimitations:")
    print("✗ Slow on large datasets (O(n²))")
    print("✗ Non-deterministic (different runs give different results)")
    print("✗ Cannot embed new data (no transform method)")
    print("✗ Distances in t-SNE space don't have meaning")
    print("✗ Cluster sizes don't reflect actual sizes")


if __name__ == "__main__":
    main()
