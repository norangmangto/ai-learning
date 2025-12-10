"""
One-Class SVM for Anomaly Detection with Scikit-learn

One-Class SVM learns a decision boundary around normal data in feature space.
- Based on Support Vector Machines
- Learns a hyperplane that separates normal data from the origin
- Effective for novelty detection
- Works well in high-dimensional spaces

This implementation includes:
- Kernel selection (RBF, polynomial, linear)
- Nu parameter tuning (upper bound on anomaly fraction)
- Comparison with Isolation Forest
- Decision boundary visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
)
from sklearn.ensemble import IsolationForest
import seaborn as sns


def generate_anomaly_data(n_samples=1000, contamination=0.1, n_features=2):
    """
    Generate dataset with anomalies.

    Args:
        n_samples: Total number of samples
        contamination: Fraction of anomalies
        n_features: Number of features

    Returns:
        X: Feature matrix
        y_true: True labels (1 = normal, -1 = anomaly)
    """
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies

    if n_features == 2:
        # Normal data: tight Gaussian
        np.random.seed(42)
        X_normal = np.random.randn(n_normal, 2) * 0.8

        # Anomalies: scattered outliers
        angles = np.linspace(0, 2 * np.pi, n_anomalies)
        radius = np.random.uniform(3, 5, n_anomalies)
        X_anomalies = np.column_stack(
            [radius * np.cos(angles), radius * np.sin(angles)]
        )
    else:
        # High-dimensional data
        X_normal, _ = make_classification(
            n_samples=n_normal,
            n_features=n_features,
            n_informative=n_features - 2,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42,
        )

        # Anomalies: extreme values
        np.random.seed(42)
        X_anomalies = np.random.uniform(low=-6, high=6, size=(n_anomalies, n_features))

    # Combine and shuffle
    X = np.vstack([X_normal, X_anomalies])
    y_true = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])

    shuffle_idx = np.random.RandomState(42).permutation(len(X))
    X = X[shuffle_idx]
    y_true = y_true[shuffle_idx]

    print(f"Generated dataset:")
    print(f"  Total samples: {len(X)}")
    print(f"  Normal: {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
    print(
        f"  Anomalies: {sum(y_true == -1)} ({sum(y_true == -1)/len(y_true)*100:.1f}%)"
    )
    print(f"  Features: {X.shape[1]}")

    return X, y_true


def train_one_class_svm(X, nu=0.1, kernel="rbf", gamma="scale"):
    """
    Train One-Class SVM model.

    Args:
        X: Feature matrix
        nu: Upper bound on fraction of anomalies (and lower bound on support vectors)
        kernel: Kernel type ('rbf', 'poly', 'linear', 'sigmoid')
        gamma: Kernel coefficient

    Returns:
        model: Trained One-Class SVM
        y_pred: Predictions (1 = normal, -1 = anomaly)
        scores: Decision function scores
    """
    print(f"\nTraining One-Class SVM...")
    print(f"  nu: {nu}")
    print(f"  kernel: {kernel}")
    print(f"  gamma: {gamma}")

    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    # Fit and predict
    model.fit(X)
    y_pred = model.predict(X)

    # Get decision function scores (distance to hyperplane)
    scores = model.decision_function(X)

    print(f"\nModel trained!")
    print(f"  Predicted normal: {sum(y_pred == 1)}")
    print(f"  Predicted anomalies: {sum(y_pred == -1)}")
    print(f"  Number of support vectors: {len(model.support_vectors_)}")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    return model, y_pred, scores


def compare_kernels(X, y_true, nu=0.1):
    """
    Compare different kernel types.

    Args:
        X: Feature matrix
        y_true: True labels
        nu: Nu parameter
    """
    if X.shape[1] != 2:
        print("Kernel comparison visualization only available for 2D data")
        return

    kernels = ["rbf", "poly", "linear", "sigmoid"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    print("\n" + "=" * 70)
    print("Comparing Different Kernels")
    print("=" * 70)

    results = []

    for idx, kernel in enumerate(kernels):
        print(f"\nTesting kernel: {kernel}...")

        # Train model
        model = OneClassSVM(nu=nu, kernel=kernel, gamma="auto")
        model.fit(X)
        y_pred = model.predict(X)
        model.decision_function(X)

        # Calculate metrics
        f1 = f1_score(y_true, y_pred, pos_label=1)

        results.append(
            {
                "kernel": kernel,
                "f1_score": f1,
                "n_support_vectors": len(model.support_vectors_),
            }
        )

        # Create mesh for decision boundary
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict on mesh
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        axes[idx].contourf(
            xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap="Blues_r", alpha=0.6
        )
        axes[idx].contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkblue")

        # Plot data
        colors = ["red" if label == -1 else "green" for label in y_true]
        axes[idx].scatter(
            X[:, 0],
            X[:, 1],
            c=colors,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
            s=40,
        )

        # Plot support vectors
        axes[idx].scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100,
            facecolors="none",
            edgecolors="orange",
            linewidth=2,
            label="Support Vectors",
        )

        axes[idx].set_title(
            f"{kernel.upper()} Kernel\nF1: {f1:.3f}, "
            f"SVs: {len(model.support_vectors_)}",
            fontsize=11,
        )
        axes[idx].set_xlabel("Feature 1")
        axes[idx].set_ylabel("Feature 2")
        axes[idx].legend(loc="upper right")

        print(f"  F1 Score: {f1:.4f}")
        print(f"  Support Vectors: {len(model.support_vectors_)}")

    plt.tight_layout()
    plt.savefig("ocsvm_kernel_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Kernel':<15} {'F1 Score':<15} {'Support Vectors':<20}")
    print("-" * 70)
    for result in results:
        print(
            f"{result['kernel']:<15} "
            f"{result['f1_score']:<15.4f} "
            f"{result['n_support_vectors']:<20}"
        )
    print("=" * 70)


def tune_nu_parameter(X, y_true, nu_values=[0.01, 0.05, 0.1, 0.15, 0.2]):
    """
    Tune the nu parameter.

    Nu: Upper bound on fraction of training errors and lower bound on support vectors.

    Args:
        X: Feature matrix
        y_true: True labels
        nu_values: List of nu values to test
    """
    print("\n" + "=" * 70)
    print("Tuning Nu Parameter")
    print("=" * 70)

    results = []

    for nu in nu_values:
        model = OneClassSVM(nu=nu, kernel="rbf", gamma="auto")
        model.fit(X)
        y_pred = model.predict(X)
        scores = model.decision_function(X)

        # Calculate metrics
        f1 = f1_score(y_true, y_pred, pos_label=1)

        # Calculate ROC AUC
        y_true_binary = (y_true == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, scores)
        roc_auc = auc(fpr, tpr)

        results.append(
            {
                "nu": nu,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "predicted_anomalies": sum(y_pred == -1),
                "n_support_vectors": len(model.support_vectors_),
            }
        )

        print(f"\nNu = {nu}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Predicted anomalies: {sum(y_pred == -1)}")
        print(f"  Support vectors: {len(model.support_vectors_)}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    nu_vals = [r["nu"] for r in results]

    # F1 Score
    axes[0].plot(
        nu_vals,
        [r["f1_score"] for r in results],
        "o-",
        linewidth=2,
        markersize=8,
        color="darkgreen",
    )
    axes[0].set_xlabel("Nu", fontsize=12)
    axes[0].set_ylabel("F1 Score", fontsize=12)
    axes[0].set_title("F1 Score vs Nu", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # ROC AUC
    axes[1].plot(
        nu_vals,
        [r["roc_auc"] for r in results],
        "o-",
        linewidth=2,
        markersize=8,
        color="darkblue",
    )
    axes[1].set_xlabel("Nu", fontsize=12)
    axes[1].set_ylabel("ROC AUC", fontsize=12)
    axes[1].set_title("ROC AUC vs Nu", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # Number of support vectors
    axes[2].plot(
        nu_vals,
        [r["n_support_vectors"] for r in results],
        "o-",
        linewidth=2,
        markersize=8,
        color="darkorange",
    )
    axes[2].set_xlabel("Nu", fontsize=12)
    axes[2].set_ylabel("Number of Support Vectors", fontsize=12)
    axes[2].set_title("Support Vectors vs Nu", fontsize=13)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ocsvm_nu_tuning.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Find best nu
    best_idx = np.argmax([r["f1_score"] for r in results])
    best_nu = results[best_idx]["nu"]

    print(f"\n{'='*70}")
    print(f"Best nu: {best_nu}")
    print(f"F1 Score: {results[best_idx]['f1_score']:.4f}")
    print(f"ROC AUC: {results[best_idx]['roc_auc']:.4f}")
    print(f"{'='*70}")

    return best_nu


def compare_with_isolation_forest(X, y_true):
    """
    Compare One-Class SVM with Isolation Forest.

    Args:
        X: Feature matrix
        y_true: True labels
    """
    print("\n" + "=" * 70)
    print("Comparing One-Class SVM with Isolation Forest")
    print("=" * 70)

    # One-Class SVM
    print("\n1. Training One-Class SVM...")
    ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="auto")
    ocsvm.fit(X)
    y_pred_svm = ocsvm.predict(X)
    scores_svm = ocsvm.decision_function(X)

    # Isolation Forest
    print("2. Training Isolation Forest...")
    iforest = IsolationForest(contamination=0.1, random_state=42)
    y_pred_if = iforest.fit_predict(X)
    scores_if = iforest.score_samples(X)

    # Calculate metrics
    f1_svm = f1_score(y_true, y_pred_svm, pos_label=1)
    f1_if = f1_score(y_true, y_pred_if, pos_label=1)

    y_true_binary = (y_true == 1).astype(int)

    fpr_svm, tpr_svm, _ = roc_curve(y_true_binary, scores_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)

    fpr_if, tpr_if, _ = roc_curve(y_true_binary, scores_if)
    roc_auc_if = auc(fpr_if, tpr_if)

    # Visualize comparison
    if X.shape[1] == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # One-Class SVM
        colors_svm = ["red" if label == -1 else "blue" for label in y_pred_svm]
        axes[0].scatter(
            X[:, 0],
            X[:, 1],
            c=colors_svm,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
            s=40,
        )
        axes[0].set_title(
            f"One-Class SVM\nF1: {f1_svm:.3f}, ROC AUC: {roc_auc_svm:.3f}", fontsize=12
        )
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")

        # Isolation Forest
        colors_if = ["red" if label == -1 else "blue" for label in y_pred_if]
        axes[1].scatter(
            X[:, 0],
            X[:, 1],
            c=colors_if,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
            s=40,
        )
        axes[1].set_title(
            f"Isolation Forest\nF1: {
        f1_if:.3f}, ROC AUC: {
            roc_auc_if:.3f}",
            fontsize=12,
        )
        axes[1].set_xlabel("Feature 1")
        axes[1].set_ylabel("Feature 2")

        plt.tight_layout()
        plt.savefig("ocsvm_vs_iforest_visual.png", dpi=300, bbox_inches="tight")
        plt.show()

    # ROC curves comparison
    plt.figure(figsize=(10, 6))
    plt.plot(
        fpr_svm, tpr_svm, linewidth=2, label=f"One-Class SVM (AUC = {roc_auc_svm:.3f})"
    )
    plt.plot(
        fpr_if, tpr_if, linewidth=2, label=f"Isolation Forest (AUC = {roc_auc_if:.3f})"
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ocsvm_vs_iforest_roc.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print comparison
    print("\n" + "=" * 70)
    print(f"{'Method':<25} {'F1 Score':<15} {'ROC AUC':<15}")
    print("-" * 70)
    print(f"{'One-Class SVM':<25} {f1_svm:<15.4f} {roc_auc_svm:<15.4f}")
    print(f"{'Isolation Forest':<25} {f1_if:<15.4f} {roc_auc_if:<15.4f}")
    print("=" * 70)

    print("\nWhen to use each:")
    print("One-Class SVM:")
    print("  ✓ Small to medium datasets")
    print("  ✓ When decision boundary is important")
    print("  ✓ Low-dimensional data")
    print("  ✓ When using kernel tricks is beneficial")
    print("\nIsolation Forest:")
    print("  ✓ Large datasets")
    print("  ✓ High-dimensional data")
    print("  ✓ When speed is critical")
    print("  ✓ When interpretability is not needed")


def evaluate_performance(y_true, y_pred, scores):
    """
    Evaluate model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        scores: Decision scores
    """
    print("\n" + "=" * 60)
    print("Performance Evaluation")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Anomaly", "Normal"]))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

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
    plt.savefig("ocsvm_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("=" * 60)


def main():
    """Main execution function."""
    print("=" * 70)
    print("One-Class SVM for Anomaly Detection")
    print("=" * 70)

    # 1. Generate data
    print("\n1. Generating dataset with anomalies...")
    X, y_true = generate_anomaly_data(n_samples=1000, contamination=0.1, n_features=2)

    # Scale features (crucial for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Compare different kernels
    print("\n2. Comparing different kernel types...")
    compare_kernels(X_scaled, y_true, nu=0.1)

    # 3. Tune nu parameter
    print("\n3. Tuning nu parameter...")
    best_nu = tune_nu_parameter(
        X_scaled, y_true, nu_values=[0.01, 0.05, 0.1, 0.15, 0.2]
    )

    # 4. Train with optimal parameters
    print(f"\n4. Training One-Class SVM with optimal parameters (nu={best_nu})...")
    model, y_pred, scores = train_one_class_svm(
        X_scaled, nu=best_nu, kernel="rbf", gamma="auto"
    )

    # 5. Evaluate performance
    print("\n5. Evaluating performance...")
    evaluate_performance(y_true, y_pred, scores)

    # 6. Compare with Isolation Forest
    print("\n6. Comparing with Isolation Forest...")
    compare_with_isolation_forest(X_scaled, y_true)

    print("\n" + "=" * 70)
    print("One-Class SVM Analysis Complete!")
    print("=" * 70)

    print("\nKey Advantages:")
    print("✓ Effective in high-dimensional spaces")
    print("✓ Uses kernel trick for non-linear boundaries")
    print("✓ Theoretically well-founded (SVM theory)")
    print("✓ Good for small to medium datasets")

    print("\nWhen to use One-Class SVM:")
    print("✓ Need clear decision boundary")
    print("✓ Small to medium sized datasets (< 10,000)")
    print("✓ When kernel methods are beneficial")
    print("✓ Novelty detection problems")

    print("\nLimitations:")
    print("✗ Slow on large datasets (quadratic complexity)")
    print("✗ Sensitive to kernel and parameter choice")
    print("✗ Requires feature scaling")
    print("✗ Memory intensive for large datasets")

    print("\nBest Practices:")
    print("1. Always scale features (StandardScaler)")
    print("2. Start with RBF kernel and nu=0.1")
    print("3. Tune nu based on expected anomaly rate")
    print("4. Try different kernels if RBF doesn't work")
    print("5. Consider Isolation Forest for large datasets")


if __name__ == "__main__":
    main()
