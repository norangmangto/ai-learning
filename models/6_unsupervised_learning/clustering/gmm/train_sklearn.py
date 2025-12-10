"""
Gaussian Mixture Models (GMM) with Scikit-learn

GMM is a probabilistic model that assumes data is generated from a mixture of
Gaussian distributions. Unlike K-Means (hard clustering), GMM provides:
- Soft clustering (probability of belonging to each cluster)
- Can model elliptical clusters (not just spherical)
- Provides uncertainty estimates

This implementation includes:
- EM algorithm for parameter estimation
- BIC/AIC for model selection
- Comparison with K-Means
- Visualization of cluster probabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse


def generate_sample_data(n_samples=500, dataset_type='blobs'):
    """
    Generate sample datasets for clustering.

    Args:
        n_samples: Number of samples to generate
        dataset_type: 'blobs', 'moons', or 'elongated'

    Returns:
        X: Feature matrix
        y_true: True labels (for evaluation only)
    """
    if dataset_type == 'blobs':
        X, y_true = make_blobs(n_samples=n_samples, centers=4,
                               cluster_std=1.0, random_state=42)
    elif dataset_type == 'moons':
        X, y_true = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    else:  # elongated clusters
        # Create elongated clusters
        np.random.seed(42)
        X1 = np.random.randn(n_samples//3, 2) @ np.array([[3, 0], [0, 0.5]])
        X2 = np.random.randn(n_samples//3, 2) @ np.array([[0.5, 0], [0, 3]]) + [8, 8]
        X3 = np.random.randn(n_samples//3, 2) + [8, 0]
        X = np.vstack([X1, X2, X3])
        y_true = np.hstack([np.zeros(n_samples//3),
                           np.ones(n_samples//3),
                           np.ones(n_samples//3)*2])

    return X, y_true


def find_optimal_components(X, max_components=10, covariance_type='full'):
    """
    Find optimal number of components using BIC and AIC.

    Args:
        X: Feature matrix (scaled)
        max_components: Maximum number of components to try
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')

    Returns:
        optimal_n: Suggested number of components
    """
    n_components_range = range(2, max_components + 1)
    bic_scores = []
    aic_scores = []

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, covariance_type=covariance_type,
                             random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    # Plot BIC and AIC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BIC (lower is better)
    axes[0].plot(n_components_range, bic_scores, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Components', fontsize=12)
    axes[0].set_ylabel('BIC Score', fontsize=12)
    axes[0].set_title('Bayesian Information Criterion (Lower is Better)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    optimal_bic = n_components_range[np.argmin(bic_scores)]
    axes[0].axvline(x=optimal_bic, color='r', linestyle='--',
                    label=f'Optimal: {optimal_bic}')
    axes[0].legend()

    # AIC (lower is better)
    axes[1].plot(n_components_range, aic_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('AIC Score', fontsize=12)
    axes[1].set_title('Akaike Information Criterion (Lower is Better)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    optimal_aic = n_components_range[np.argmin(aic_scores)]
    axes[1].axvline(x=optimal_aic, color='r', linestyle='--',
                    label=f'Optimal: {optimal_aic}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('gmm_model_selection.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nOptimal number of components:")
    print(f"  Based on BIC: {optimal_bic}")
    print(f"  Based on AIC: {optimal_aic}")

    # Use BIC as primary criterion (more conservative)
    return optimal_bic


def compare_covariance_types(X, n_components=4):
    """
    Compare different covariance types.

    Args:
        X: Feature matrix (scaled)
        n_components: Number of components
    """
    covariance_types = ['full', 'tied', 'diag', 'spherical']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    results = []

    for idx, cov_type in enumerate(covariance_types):
        # Fit GMM
        gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type,
                             random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)

        # Calculate metrics
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        sil_score = silhouette_score(X, labels)

        results.append({
            'type': cov_type,
            'bic': bic,
            'aic': aic,
            'silhouette': sil_score
        })

        # Visualize
        axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                         alpha=0.6, edgecolors='black', linewidth=0.5)

        # Draw ellipses for each component
        draw_ellipses(gmm, axes[idx])

        axes[idx].set_title(f'{cov_type.capitalize()} Covariance\n'
                           f'BIC: {bic:.0f}, Silhouette: {sil_score:.3f}',
                           fontsize=11)
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')

    plt.tight_layout()
    plt.savefig('gmm_covariance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print comparison table
    print("\n" + "="*70)
    print("Covariance Type Comparison")
    print("="*70)
    print(f"{'Type':<15} {'BIC':<15} {'AIC':<15} {'Silhouette':<15}")
    print("-"*70)
    for result in results:
        print(f"{result['type']:<15} "
              f"{result['bic']:<15.2f} "
              f"{result['aic']:<15.2f} "
              f"{result['silhouette']:<15.4f}")
    print("="*70)
    print("\nCovariance Types Explained:")
    print("- full: Each component has its own covariance matrix (most flexible)")
    print("- tied: All components share the same covariance matrix")
    print("- diag: Diagonal covariance (features are independent)")
    print("- spherical: Circular/spherical clusters (like K-Means)")


def draw_ellipses(gmm, ax):
    """
    Draw confidence ellipses for GMM components.

    Args:
        gmm: Fitted GaussianMixture model
        ax: Matplotlib axis
    """
    for i in range(gmm.n_components):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[i][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[i][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(2) * gmm.covariances_[i]

        # Calculate eigenvalues and eigenvectors
        v, w = np.linalg.eigh(covariances)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])

        # Calculate angle
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi

        # Draw ellipse
        ell = Ellipse(gmm.means_[i, :2], v[0], v[1], angle=180.0 + angle,
                     edgecolor='red', facecolor='none', linewidth=2, alpha=0.7)
        ax.add_patch(ell)


def train_gmm(X, n_components=4, covariance_type='full'):
    """
    Train Gaussian Mixture Model.

    Args:
        X: Feature matrix (scaled)
        n_components: Number of components
        covariance_type: Type of covariance

    Returns:
        gmm: Fitted GMM model
        labels: Hard cluster assignments
        probs: Soft cluster probabilities
    """
    # Create and fit model
    gmm = GaussianMixture(n_components=n_components,
                         covariance_type=covariance_type,
                         random_state=42,
                         max_iter=200,
                         n_init=10)

    gmm.fit(X)

    # Get predictions
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    # Print model information
    print(f"\nGMM Training Results:")
    print(f"Number of components: {n_components}")
    print(f"Covariance type: {covariance_type}")
    print(f"Converged: {gmm.converged_}")
    print(f"Number of iterations: {gmm.n_iter_}")
    print(f"Log-likelihood: {gmm.score(X) * len(X):.2f}")
    print(f"BIC: {gmm.bic(X):.2f}")
    print(f"AIC: {gmm.aic(X):.2f}")

    return gmm, labels, probs


def visualize_soft_clustering(X, probs, title='GMM Soft Clustering'):
    """
    Visualize soft clustering probabilities.

    Args:
        X: Feature matrix
        probs: Cluster probabilities
        title: Plot title
    """
    n_components = probs.shape[1]

    fig, axes = plt.subplots(1, n_components + 1, figsize=(5 * (n_components + 1), 4))

    # Plot hard clustering
    hard_labels = np.argmax(probs, axis=1)
    axes[0].scatter(X[:, 0], X[:, 1], c=hard_labels, cmap='viridis',
                   alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_title('Hard Clustering')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # Plot probability for each component
    for i in range(n_components):
        scatter = axes[i + 1].scatter(X[:, 0], X[:, 1], c=probs[:, i],
                                     cmap='RdYlGn', vmin=0, vmax=1,
                                     alpha=0.7, edgecolors='black', linewidth=0.5)
        axes[i + 1].set_title(f'Component {i} Probability')
        axes[i + 1].set_xlabel('Feature 1')
        axes[i + 1].set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=axes[i + 1], label='Probability')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('gmm_soft_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_kmeans(X, n_clusters=4):
    """
    Compare GMM with K-Means clustering.

    Args:
        X: Feature matrix (scaled)
        n_clusters: Number of clusters
    """
    from sklearn.cluster import KMeans

    # Fit models
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)

    # Calculate metrics
    gmm_sil = silhouette_score(X, gmm_labels)
    kmeans_sil = silhouette_score(X, kmeans_labels)

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # GMM
    axes[0].scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis',
                   alpha=0.6, edgecolors='black', linewidth=0.5)
    draw_ellipses(gmm, axes[0])
    axes[0].set_title(f'GMM (Silhouette: {gmm_sil:.3f})')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # K-Means
    axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis',
                   alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                   label='Centroids')
    axes[1].set_title(f'K-Means (Silhouette: {kmeans_sil:.3f})')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('gmm_vs_kmeans.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*50)
    print("GMM vs K-Means Comparison")
    print("="*50)
    print(f"GMM Silhouette Score: {gmm_sil:.4f}")
    print(f"K-Means Silhouette Score: {kmeans_sil:.4f}")
    print("\nKey Differences:")
    print("- GMM: Soft clustering, probabilistic, elliptical clusters")
    print("- K-Means: Hard clustering, deterministic, spherical clusters")
    print("="*50)


def analyze_uncertainty(probs):
    """
    Analyze clustering uncertainty.

    Args:
        probs: Cluster probabilities
    """
    # Calculate entropy (uncertainty measure)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

    # Calculate max probability
    max_probs = np.max(probs, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Entropy distribution
    axes[0].hist(entropy, bins=30, edgecolor='black', color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Entropy (Uncertainty)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Clustering Uncertainty')
    axes[0].axvline(x=np.median(entropy), color='r', linestyle='--',
                   label=f'Median: {np.median(entropy):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Max probability distribution
    axes[1].hist(max_probs, bins=30, edgecolor='black', color='green', alpha=0.7)
    axes[1].set_xlabel('Maximum Probability')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Maximum Cluster Probability')
    axes[1].axvline(x=np.median(max_probs), color='r', linestyle='--',
                   label=f'Median: {np.median(max_probs):.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gmm_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Identify uncertain samples
    uncertain_threshold = np.percentile(entropy, 90)
    uncertain_samples = np.where(entropy > uncertain_threshold)[0]

    print(f"\nUncertainty Analysis:")
    print(f"Mean entropy: {np.mean(entropy):.3f}")
    print(f"Median entropy: {np.median(entropy):.3f}")
    print(f"Mean max probability: {np.mean(max_probs):.3f}")
    print(f"Samples with high uncertainty (top 10%): {len(uncertain_samples)}")


def main():
    """Main execution function."""
    print("="*70)
    print("Gaussian Mixture Models (GMM) with Scikit-learn")
    print("="*70)

    # 1. Generate sample data
    print("\n1. Generating sample data with elongated clusters...")
    X, y_true = generate_sample_data(n_samples=600, dataset_type='elongated')

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset shape: {X_scaled.shape}")
    print(f"Number of true clusters: {len(np.unique(y_true))}")

    # 2. Find optimal number of components
    print("\n2. Finding optimal number of components using BIC/AIC...")
    optimal_n = find_optimal_components(X_scaled, max_components=8)

    # 3. Compare covariance types
    print("\n3. Comparing covariance types...")
    compare_covariance_types(X_scaled, n_components=optimal_n)

    # 4. Train GMM with optimal parameters
    print("\n4. Training GMM with optimal parameters...")
    gmm, labels, probs = train_gmm(X_scaled, n_components=optimal_n,
                                   covariance_type='full')

    # 5. Visualize soft clustering
    print("\n5. Visualizing soft clustering probabilities...")
    visualize_soft_clustering(X_scaled, probs,
                             title='GMM Soft Clustering Probabilities')

    # 6. Compare with K-Means
    print("\n6. Comparing GMM with K-Means...")
    compare_with_kmeans(X_scaled, n_clusters=optimal_n)

    # 7. Analyze uncertainty
    print("\n7. Analyzing clustering uncertainty...")
    analyze_uncertainty(probs)

    # 8. Test on spherical data
    print("\n8. Testing on spherical data (comparing with K-Means)...")
    X_blobs, _ = generate_sample_data(n_samples=500, dataset_type='blobs')
    X_blobs_scaled = scaler.fit_transform(X_blobs)
    compare_with_kmeans(X_blobs_scaled, n_clusters=4)

    print("\n" + "="*70)
    print("GMM Clustering Complete!")
    print("="*70)
    print("\nKey Advantages of GMM:")
    print("✓ Soft clustering (probability of membership)")
    print("✓ Can model elliptical/elongated clusters")
    print("✓ Provides uncertainty estimates")
    print("✓ Probabilistic framework (generative model)")
    print("\nWhen to use GMM over K-Means:")
    print("✓ Clusters are not spherical")
    print("✓ Need probability/confidence scores")
    print("✓ Want to detect overlapping clusters")
    print("✓ Need density estimation")
    print("\nLimitations:")
    print("✗ Slower than K-Means (EM algorithm)")
    print("✗ Can converge to local optima")
    print("✗ Sensitive to initialization")
    print("✗ Assumes Gaussian distribution")


if __name__ == "__main__":
    main()
