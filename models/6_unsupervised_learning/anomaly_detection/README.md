# Anomaly Detection

Methods for identifying outliers and anomalies in data.

## 📋 Algorithms

### Isolation Forest
- **Approach**: Random forest-based
- **Complexity**: O(n log n)
- **Best for**: High-dimensional data, fast detection
- **File**: `isolation_forest/train_isolation_forest.py`

**Key insight:** Anomalies are isolated faster in random trees

### One-Class SVM
- **Approach**: Kernel-based SVM
- **Complexity**: O(n²) to O(n³)
- **Best for**: Non-linear boundaries, small-medium datasets
- **File**: `one_class_svm/train_one_class_svm.py`

**Key insight:** Find hyperplane maximizing margin from origin

### Autoencoder-Based
- **Approach**: Neural network reconstruction
- **Complexity**: Depends on architecture
- **Best for**: Complex patterns, images
- **File**: `autoencoder_based/train_autoencoder.py`

**Key insight:** Anomalies have high reconstruction error

## 🎯 Quick Comparison

| Method | Speed | Scalability | Interpretable | Nonlinear |
|--------|-------|------------|---------------|-----------|
| Isolation Forest | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| One-Class SVM | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Autoencoder | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |

## 📊 Decision Guide

```
Detecting anomalies?
├─ High-dimensional data?: Yes → Use Isolation Forest
├─ Need fast inference?: Yes → Use Isolation Forest
├─ Complex nonlinear?: Yes → Use One-Class SVM or Autoencoder
├─ Image/visual data?: Yes → Use Autoencoder
└─ Want interpretability?: Yes → Use Isolation Forest
```

## 📚 Learn More

See individual subdirectories for implementations and examples.

**Last Updated:** December 2024
