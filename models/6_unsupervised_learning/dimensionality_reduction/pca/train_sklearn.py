"""
Principal Component Analysis (PCA) with Scikit-learn

PCA is a linear dimensionality reduction technique that:
- Projects data onto orthogonal principal components
- Maximizes variance explained
- Reduces computational cost and noise
- Enables visualization of high-dimensional data

This implementation includes:
- Standard PCA, Incremental PCA, Kernel PCA
- Explained variance analysis
- 2D/3D visualization
- Reconstruction quality analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.datasets import load_digits, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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
        # Load digits dataset (64 features)
        digits = load_digits()
        X, y = digits.data, digits.target
        print(f"Loaded digits dataset: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        # Create synthetic high-dimensional data
        X, y = make_classification(n_samples=1000, n_features=50,
                                   n_informative=30, n_redundant=10,
                                   n_classes=5, random_state=42)
        print(f"Generated synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y


def analyze_explained_variance(X, max_components=None):
    """
    Analyze explained variance for different numbers of components.

    Args:
        X: Feature matrix (scaled)
        max_components: Maximum components to analyze

    Returns:
        optimal_n: Suggested number of components
    """
    if max_components is None:
        max_components = min(X.shape[0], X.shape[1])

    # Fit PCA with all components
    pca_full = PCA(n_components=max_components, random_state=42)
    pca_full.fit(X)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Individual explained variance
    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                pca_full.explained_variance_ratio_,
                color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('Individual Explained Variance per Component', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Cumulative explained variance
    axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                'o-', color='darkgreen', linewidth=2, markersize=6)
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Explained Variance', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Find optimal number of components (95% variance)
    optimal_95 = np.argmax(cumulative_variance >= 0.95) + 1
    optimal_90 = np.argmax(cumulative_variance >= 0.90) + 1
    axes[1].axvline(x=optimal_95, color='r', linestyle=':', alpha=0.5)
    axes[1].axvline(x=optimal_90, color='orange', linestyle=':', alpha=0.5)

    # 3. Eigenvalues (Scree plot)
    axes[2].plot(range(1, len(pca_full.explained_variance_) + 1),
                pca_full.explained_variance_,
                'o-', color='purple', linewidth=2, markersize=6)
    axes[2].set_xlabel('Principal Component', fontsize=12)
    axes[2].set_ylabel('Eigenvalue', fontsize=12)
    axes[2].set_title('Scree Plot (Eigenvalues)', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    # Find elbow using rate of change
    variance_diff = np.diff(pca_full.explained_variance_)
    elbow = np.argmax(variance_diff[1:] - variance_diff[:-1]) + 2
    axes[2].axvline(x=elbow, color='r', linestyle='--',
                   label=f'Elbow: {elbow}', alpha=0.7)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('pca_explained_variance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*60)
    print("Explained Variance Analysis")
    print("="*60)
    print(f"Components for 90% variance: {optimal_90}")
    print(f"Components for 95% variance: {optimal_95}")
    print(f"Elbow point: {elbow}")
    print(f"\nTop 5 components explain: {cumulative_variance[4]:.2%} of variance")
    print(f"Top 10 components explain: {cumulative_variance[9]:.2%} of variance")
    print("="*60)

    return optimal_95


def train_pca(X, n_components=2):
    """
    Train standard PCA.

    Args:
        X: Feature matrix (scaled)
        n_components: Number of components to keep

    Returns:
        pca: Fitted PCA model
        X_pca: Transformed data
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    print(f"\nPCA Results:")
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_pca.shape}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.4f}")

    return pca, X_pca


def visualize_2d(X_reduced, y, title='PCA Projection (2D)'):
    """
    Visualize 2D projection.

    Args:
        X_reduced: Reduced feature matrix (2D)
        y: Labels
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                         c=y, cmap='tab10', alpha=0.7,
                         edgecolors='black', linewidth=0.5, s=50)
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title(title, fontsize=14)
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_2d_projection.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_3d(X_reduced, y, title='PCA Projection (3D)'):
    """
    Visualize 3D projection.

    Args:
        X_reduced: Reduced feature matrix (3D)
        y: Labels
        title: Plot title
    """

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                        c=y, cmap='tab10', alpha=0.7,
                        edgecolors='black', linewidth=0.5, s=50)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.colorbar(scatter, label='Class', pad=0.1)
    plt.tight_layout()
    plt.savefig('pca_3d_projection.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_reconstruction_error(X, n_components_list=[2, 5, 10, 20]):
    """
    Analyze reconstruction error for different numbers of components.

    Args:
        X: Original feature matrix (scaled)
        n_components_list: List of component numbers to test
    """
    reconstruction_errors = []

    for n_comp in n_components_list:
        pca = PCA(n_components=n_comp, random_state=42)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)

        # Calculate reconstruction error (MSE)
        error = np.mean((X - X_reconstructed) ** 2)
        reconstruction_errors.append(error)

    # Plot reconstruction error
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_list, reconstruction_errors, 'o-',
            linewidth=2, markersize=8, color='darkred')
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
    plt.title('PCA Reconstruction Error', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add annotations
    for n_comp, error in zip(n_components_list, reconstruction_errors):
        plt.annotate(f'{error:.4f}', (n_comp, error),
                    textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('pca_reconstruction_error.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nReconstruction Error Analysis:")
    for n_comp, error in zip(n_components_list, reconstruction_errors):
        print(f"{n_comp} components: MSE = {error:.6f}")


def compare_pca_variants(X, n_components=2):
    """
    Compare standard PCA, Incremental PCA, and Kernel PCA.

    Args:
        X: Feature matrix (scaled)
        n_components: Number of components
    """
    # Standard PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # Incremental PCA (for large datasets)
    ipca = IncrementalPCA(n_components=n_components, batch_size=100)
    X_ipca = ipca.fit_transform(X)

    # Kernel PCA (for non-linear patterns)
    kpca_rbf = KernelPCA(n_components=n_components, kernel='rbf',
                        gamma=0.1, random_state=42)
    X_kpca_rbf = kpca_rbf.fit_transform(X)

    kpca_poly = KernelPCA(n_components=n_components, kernel='poly',
                         degree=3, random_state=42)
    X_kpca_poly = kpca_poly.fit_transform(X)

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    methods = [
        (X_pca, 'Standard PCA', axes[0, 0]),
        (X_ipca, 'Incremental PCA', axes[0, 1]),
        (X_kpca_rbf, 'Kernel PCA (RBF)', axes[1, 0]),
        (X_kpca_poly, 'Kernel PCA (Polynomial)', axes[1, 1])
    ]

    for X_reduced, title, ax in methods:
        # Use dummy labels for visualization
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                  alpha=0.6, edgecolors='black', linewidth=0.5, s=30)
        ax.set_xlabel('Component 1', fontsize=11)
        ax.set_ylabel('Component 2', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*60)
    print("PCA Variants Comparison")
    print("="*60)
    print("Standard PCA:")
    print("  - Fast, linear dimensionality reduction")
    print("  - Best for: Linear relationships, large datasets")
    print("\nIncremental PCA:")
    print("  - Memory-efficient (processes data in batches)")
    print("  - Best for: Very large datasets that don't fit in memory")
    print("\nKernel PCA:")
    print("  - Can capture non-linear relationships")
    print("  - Best for: Non-linear data patterns")
    print("  - Slower and more complex than standard PCA")
    print("="*60)


def evaluate_classification_performance(X, y, n_components_list=[2, 5, 10, 20]):
    """
    Evaluate how dimensionality reduction affects classification performance.

    Args:
        X: Feature matrix (scaled)
        y: Labels
        n_components_list: List of component numbers to test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Baseline (no PCA)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, clf.predict(X_test))

    # With PCA
    accuracies = [baseline_acc]
    component_counts = [X.shape[1]]  # Original dimensionality

    for n_comp in n_components_list:
        if n_comp >= X.shape[1]:
            continue

        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_pca, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_pca))

        accuracies.append(acc)
        component_counts.append(n_comp)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(component_counts, accuracies, 'o-', linewidth=2, markersize=8,
            color='darkgreen')
    plt.axhline(y=baseline_acc, color='r', linestyle='--',
               label=f'Baseline (no PCA): {baseline_acc:.3f}')
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title('Classification Performance vs Dimensionality', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_classification_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nClassification Performance Analysis:")
    print(f"Baseline (no PCA, {X.shape[1]} features): {baseline_acc:.4f}")
    for n_comp, acc in zip(component_counts[1:], accuracies[1:]):
        print(f"{n_comp} components: {acc:.4f} "
              f"({'↑' if acc > baseline_acc else '↓'} {abs(acc - baseline_acc):.4f})")


def main():
    """Main execution function."""
    print("="*70)
    print("Principal Component Analysis (PCA) with Scikit-learn")
    print("="*70)

    # 1. Load data
    print("\n1. Loading high-dimensional dataset...")
    X, y = load_sample_data(dataset='digits')

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

     # 2. Analyze explained variance
    print("\n2. Analyzing explained variance...")
    analyze_explained_variance(X_scaled, max_components=30)

    # 3. Analyze reconstruction error
    print("\n3. Analyzing reconstruction error...")
    analyze_reconstruction_error(X_scaled, n_components_list=[2, 5, 10, 20, 30])

    # 4. Train PCA with 2 components
    print("\n4. Training PCA with 2 components (for visualization)...")
    pca_2d, X_pca_2d = train_pca(X_scaled, n_components=2)

    # 5. Visualize 2D projection
    print("\n5. Visualizing 2D projection...")
    visualize_2d(X_pca_2d, y, title='PCA Projection - Digits Dataset (2D)')

    # 6. Train PCA with 3 components
    print("\n6. Training PCA with 3 components...")
    pca_3d, X_pca_3d = train_pca(X_scaled, n_components=3)

    # 7. Visualize 3D projection
    print("\n7. Visualizing 3D projection...")
    visualize_3d(X_pca_3d, y, title='PCA Projection - Digits Dataset (3D)')

    # 8. Compare PCA variants
    print("\n8. Comparing PCA variants...")
    compare_pca_variants(X_scaled, n_components=2)

    # 9. Evaluate classification performance
    print("\n9. Evaluating classification performance with PCA...")
    evaluate_classification_performance(X_scaled, y,
                                      n_components_list=[2, 5, 10, 20, 30, 40])

    print("\n" + "="*70)
    print("PCA Analysis Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("✓ PCA reduces dimensionality while preserving variance")
    print("✓ Use explained variance to choose number of components")
    print("✓ Typically 90-95% variance is sufficient")
    print("✓ Can improve model performance and reduce overfitting")
    print("\nWhen to use PCA:")
    print("✓ High-dimensional data (curse of dimensionality)")
    print("✓ Visualization of high-dimensional data")
    print("✓ Noise reduction and feature extraction")
    print("✓ Speeding up training time")
    print("\nLimitations:")
    print("✗ Assumes linear relationships")
    print("✗ Loses interpretability (PCs are combinations of features)")
    print("✗ Sensitive to feature scaling")
    print("✗ May not preserve local structure")


if __name__ == "__main__":
    main()
