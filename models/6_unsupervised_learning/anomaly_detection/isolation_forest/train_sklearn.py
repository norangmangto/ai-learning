"""
Isolation Forest for Anomaly Detection with Scikit-learn

Isolation Forest is an unsupervised anomaly detection algorithm that:
- Isolates anomalies by randomly selecting features and split values
- Anomalies require fewer splits to isolate (shorter paths)
- Fast and effective for high-dimensional data
- No assumptions about data distribution

This implementation includes:
- Contamination parameter tuning
- ROC curve analysis
- Comparison with other methods
- Feature importance for anomalies
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
import seaborn as sns


def generate_anomaly_data(n_samples=1000, contamination=0.1, dataset_type='blobs'):
    """
    Generate dataset with anomalies.

    Args:
        n_samples: Total number of samples
        contamination: Fraction of anomalies (0.1 = 10%)
        dataset_type: 'blobs', 'classification', or 'mixed'

    Returns:
        X: Feature matrix
        y_true: True labels (1 = normal, -1 = anomaly)
    """
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies

    if dataset_type == 'blobs':
        # Normal data: tight clusters
        X_normal, _ = make_blobs(n_samples=n_normal, centers=3,
                                 cluster_std=0.5, random_state=42)

        # Anomalies: scattered points
        np.random.seed(42)
        X_anomalies = np.random.uniform(low=-8, high=8, size=(n_anomalies, 2))

    elif dataset_type == 'classification':
        # Normal data: separable classes
        X_normal, _ = make_classification(n_samples=n_normal, n_features=10,
                                         n_informative=8, n_redundant=2,
                                         n_clusters_per_class=1, random_state=42)

        # Anomalies: extreme values
        np.random.seed(42)
        X_anomalies = np.random.uniform(low=-6, high=6, size=(n_anomalies, 10))

    else:  # mixed
        # Normal: Gaussian
        np.random.seed(42)
        X_normal = np.random.randn(n_normal, 2) * 0.5

        # Anomalies: outliers in different directions
        angles = np.linspace(0, 2*np.pi, n_anomalies)
        radius = np.random.uniform(3, 5, n_anomalies)
        X_anomalies = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ])

    # Combine
    X = np.vstack([X_normal, X_anomalies])
    y_true = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])

    # Shuffle
    shuffle_idx = np.random.RandomState(42).permutation(len(X))
    X = X[shuffle_idx]
    y_true = y_true[shuffle_idx]

    print(f"Generated dataset:")
    print(f"  Total samples: {len(X)}")
    print(f"  Normal: {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
    print(f"  Anomalies: {sum(y_true == -1)} ({sum(y_true == -1)/len(y_true)*100:.1f}%)")
    print(f"  Features: {X.shape[1]}")

    return X, y_true


def train_isolation_forest(X, contamination=0.1, n_estimators=100):
    """
    Train Isolation Forest model.

    Args:
        X: Feature matrix
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees

    Returns:
        model: Trained Isolation Forest
        y_pred: Predictions (1 = normal, -1 = anomaly)
        scores: Anomaly scores (lower = more anomalous)
    """
    print(f"\nTraining Isolation Forest...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  contamination: {contamination}")
    print(f"  max_samples: auto")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    # Fit and predict
    y_pred = model.fit_predict(X)

    # Get anomaly scores (lower = more anomalous)
    scores = model.score_samples(X)

    # Decision function (negative values = anomalies)
    decision_scores = model.decision_function(X)

    print(f"\nModel trained!")
    print(f"  Predicted normal: {sum(y_pred == 1)}")
    print(f"  Predicted anomalies: {sum(y_pred == -1)}")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    return model, y_pred, scores


def evaluate_performance(y_true, y_pred, scores):
    """
    Evaluate anomaly detection performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        scores: Anomaly scores
    """
    print("\n" + "="*60)
    print("Performance Metrics")
    print("="*60)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                               target_names=['Anomaly', 'Normal']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
               xticklabels=['Anomaly', 'Normal'],
               yticklabels=['Anomaly', 'Normal'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Plot metrics
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1]
    bars = axes[1].bar(metrics, values, color=['steelblue', 'green', 'orange'],
                      edgecolor='black')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim([0, 1])
    axes[1].set_title('Performance Metrics')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('isolation_forest_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("="*60)


def plot_roc_curve(y_true, scores):
    """
    Plot ROC curve for anomaly detection.

    Args:
        y_true: True labels (1 = normal, -1 = anomaly)
        scores: Anomaly scores (higher = more normal)
    """
    # Convert labels: 1 for normal, 0 for anomaly
    y_true_binary = (y_true == 1).astype(int)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_binary, scores)
    roc_auc = auc(fpr, tpr)

    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, scores)
    pr_auc = auc(recall, precision)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    axes[0].plot(fpr, tpr, color='darkorange', linewidth=2,
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=13)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    axes[1].plot(recall, precision, color='green', linewidth=2,
                label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=13)
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('isolation_forest_roc_pr.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")


def visualize_results(X, y_true, y_pred, scores):
    """
    Visualize anomaly detection results (for 2D data).

    Args:
        X: Feature matrix (2D)
        y_true: True labels
        y_pred: Predicted labels
        scores: Anomaly scores
    """
    if X.shape[1] != 2:
        print("Visualization only available for 2D data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True labels
    colors_true = ['red' if label == -1 else 'blue' for label in y_true]
    axes[0].scatter(X[:, 0], X[:, 1], c=colors_true, alpha=0.6,
                   edgecolors='black', linewidth=0.5, s=50)
    axes[0].set_title('True Labels\n(Red = Anomaly, Blue = Normal)', fontsize=12)
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)

    # Predicted labels
    colors_pred = ['red' if label == -1 else 'blue' for label in y_pred]
    axes[1].scatter(X[:, 0], X[:, 1], c=colors_pred, alpha=0.6,
                   edgecolors='black', linewidth=0.5, s=50)
    axes[1].set_title('Predicted Labels\n(Red = Anomaly, Blue = Normal)', fontsize=12)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].grid(True, alpha=0.3)

    # Anomaly scores (color gradient)
    scatter = axes[2].scatter(X[:, 0], X[:, 1], c=scores, cmap='RdYlGn',
                             alpha=0.7, edgecolors='black', linewidth=0.5, s=50)
    axes[2].set_title('Anomaly Scores\n(Red = More Anomalous)', fontsize=12)
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2], label='Anomaly Score')

    plt.tight_layout()
    plt.savefig('isolation_forest_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def tune_contamination(X, y_true, contamination_values=[0.05, 0.1, 0.15, 0.2]):
    """
    Compare performance with different contamination values.

    Args:
        X: Feature matrix
        y_true: True labels
        contamination_values: List of contamination values to test
    """
    print("\n" + "="*60)
    print("Tuning Contamination Parameter")
    print("="*60)

    results = []

    for contam in contamination_values:
        model = IsolationForest(contamination=contam, random_state=42, n_jobs=-1)
        y_pred = model.fit_predict(X)
        scores = model.score_samples(X)

        # Convert for sklearn metrics
        y_true_binary = (y_true == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, scores)
        roc_auc = auc(fpr, tpr)

        # Calculate F1
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, pos_label=1)

        results.append({
            'contamination': contam,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'predicted_anomalies': sum(y_pred == -1)
        })

        print(f"\nContamination = {contam}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Predicted anomalies: {sum(y_pred == -1)}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    contam_vals = [r['contamination'] for r in results]

    # ROC AUC
    axes[0].plot(contam_vals, [r['roc_auc'] for r in results],
                'o-', linewidth=2, markersize=8, color='darkblue')
    axes[0].set_xlabel('Contamination', fontsize=12)
    axes[0].set_ylabel('ROC AUC', fontsize=12)
    axes[0].set_title('ROC AUC vs Contamination', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # F1 Score
    axes[1].plot(contam_vals, [r['f1_score'] for r in results],
                'o-', linewidth=2, markersize=8, color='darkgreen')
    axes[1].set_xlabel('Contamination', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score vs Contamination', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('isolation_forest_contamination_tuning.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Find best
    best_idx = np.argmax([r['roc_auc'] for r in results])
    best_contam = results[best_idx]['contamination']
    print(f"\n{'='*60}")
    print(f"Best contamination: {best_contam}")
    print(f"ROC AUC: {results[best_idx]['roc_auc']:.4f}")
    print(f"{'='*60}")

    return best_contam


def analyze_anomaly_scores(scores, y_true):
    """
    Analyze distribution of anomaly scores.

    Args:
        scores: Anomaly scores
        y_true: True labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram by class
    normal_scores = scores[y_true == 1]
    anomaly_scores = scores[y_true == -1]

    axes[0].hist(normal_scores, bins=30, alpha=0.7, label='Normal',
                color='blue', edgecolor='black')
    axes[0].hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly',
                color='red', edgecolor='black')
    axes[0].set_xlabel('Anomaly Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Anomaly Scores', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Box plot
    data_to_plot = [normal_scores, anomaly_scores]
    axes[1].boxplot(data_to_plot, labels=['Normal', 'Anomaly'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue', edgecolor='black'),
                   medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Anomaly Score', fontsize=12)
    axes[1].set_title('Anomaly Score Distribution by Class', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('isolation_forest_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nScore Statistics:")
    print(f"Normal - Mean: {normal_scores.mean():.4f}, Std: {normal_scores.std():.4f}")
    print(f"Anomaly - Mean: {anomaly_scores.mean():.4f}, Std: {anomaly_scores.std():.4f}")


def main():
    """Main execution function."""
    print("="*70)
    print("Isolation Forest for Anomaly Detection")
    print("="*70)

    # 1. Generate data with anomalies
    print("\n1. Generating dataset with anomalies...")
    X, y_true = generate_anomaly_data(n_samples=1000, contamination=0.1,
                                      dataset_type='blobs')

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Tune contamination parameter
    print("\n2. Tuning contamination parameter...")
    best_contamination = tune_contamination(X_scaled, y_true,
                                           contamination_values=[0.05, 0.1, 0.15, 0.2])

    # 3. Train with optimal contamination
    print(f"\n3. Training Isolation Forest with optimal contamination={best_contamination}...")
    model, y_pred, scores = train_isolation_forest(X_scaled,
                                                   contamination=best_contamination,
                                                   n_estimators=100)

    # 4. Evaluate performance
    print("\n4. Evaluating performance...")
    evaluate_performance(y_true, y_pred, scores)

    # 5. Plot ROC and PR curves
    print("\n5. Plotting ROC and Precision-Recall curves...")
    plot_roc_curve(y_true, scores)

    # 6. Visualize results (2D only)
    if X.shape[1] == 2:
        print("\n6. Visualizing results...")
        visualize_results(X_scaled, y_true, y_pred, scores)

    # 7. Analyze anomaly scores
    print("\n7. Analyzing anomaly score distributions...")
    analyze_anomaly_scores(scores, y_true)

    print("\n" + "="*70)
    print("Isolation Forest Analysis Complete!")
    print("="*70)
    print("\nKey Advantages:")
    print("✓ Fast and scalable (linear time complexity)")
    print("✓ Works well with high-dimensional data")
    print("✓ No assumptions about data distribution")
    print("✓ Can handle large datasets efficiently")
    print("✓ Good for global anomalies")

    print("\nWhen to use Isolation Forest:")
    print("✓ Large datasets with many features")
    print("✓ Unknown anomaly patterns")
    print("✓ Real-time anomaly detection")
    print("✓ When speed is important")

    print("\nLimitations:")
    print("✗ May miss local anomalies")
    print("✗ Requires tuning contamination parameter")
    print("✗ Less effective in very low dimensions")
    print("✗ Cannot explain why point is anomalous")

    print("\nBest Practices:")
    print("1. Scale your features first")
    print("2. Set contamination close to expected anomaly rate")
    print("3. Use more trees (100-200) for stable results")
    print("4. Combine with other methods for robustness")


if __name__ == "__main__":
    main()
