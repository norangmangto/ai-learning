# Principal Component Analysis (PCA)

Linear dimensionality reduction that finds directions of maximum variance.

## 📋 Overview

**Type:** Linear, unsupervised
**Output:** Lower-dimensional projection
**Complexity:** O(min(n,d) × d²)
**Best For:** Visualization, preprocessing, feature extraction

## 🎯 Core Idea

Find axes along which data varies most.

```
Original 2D data:              PCA transformation:

●●    ◆◆ ← Much variance       ●●
●●        ◆◆ ← Little variance
  ●●        ◆◆
   ●●          ◆◆

PC1: Direction of max variance
PC2: Direction of 2nd max variance (orthogonal)

Project onto PC1 → 1D visualization preserving most info!
```

## 📐 Mathematical Foundation

### Variance
$$\text{Var}(PC_1) = \frac{1}{n} \sum_{i=1}^{n} (PC_1)_i^2$$

### Objective
Maximize: $\text{Var}(PC_1)$ such that $||w_1||^2 = 1$

$$w_1 = \arg\max_w \frac{w^T X^T X w}{||w||^2}$$

Eigenvector of covariance matrix with largest eigenvalue!

### Principal Components
$$PC_k = X \cdot w_k$$

where $w_k$ is k-th eigenvector of $X^T X$.

## 🔄 Algorithm

```
1. Standardize data (mean=0, std=1)
   x' = (x - mean) / std

2. Compute covariance matrix
   Σ = (1/n) X^T X

3. Compute eigenvectors and eigenvalues
   (sorted by decreasing eigenvalue)

4. Select top K eigenvectors
   PCs = eigenvectors

5. Project data
   X_new = X @ PCs
```

## 📊 Visualization Example

```
3D data → PCA → 2D plot

Original (hard to visualize):    After PCA:
          ●                      ●●  ●
        ●   ●                   ●    ●
      ●       ●         →       ●  ●
    ●           ●               ●●
  ●               ●             ●●●
                                ●  ●

Can see clusters in 2D!
```

## 🚀 Quick Start

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Data
X = np.random.randn(1000, 50)  # 1000 samples, 50 features

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)  # Reduce to 2D
X_pca = pca.fit_transform(X_scaled)

# Information
print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")
print(f"Components: {pca.components_}")  # (2, 50)

# Visualization
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.show()

# Reconstruct
X_reconstructed = pca.inverse_transform(X_pca)
```

## 📈 Explained Variance

### Variance Explained
```python
# Each PC explains some variance
var_explained = pca.explained_variance_ratio_
# [0.45, 0.28, 0.15, 0.08, ...]

# PC1 explains 45% of variance
# PC1+PC2 explain 73% of variance
```

### Scree Plot
```
Explained variance:
      ↑ 45%
      │ ╱╲
   35%│╱  ╲
      │    ╲ 15%
   25%│     ╲  8%
      │      ╲╲
   15%│       ╲╲ 3%
      │        └─→
      └─────────────
      PC1 PC2 PC3 ...

Elbow indicates "enough" components
```

## 🎯 Choosing Number of Components

### Method 1: Variance Threshold
```python
# Keep 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"Components: {pca.n_components_}")
```

### Method 2: Scree Plot
```python
pca = PCA()
pca.fit(X_scaled)

cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)
plt.axhline(y=0.95, color='r')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# Find where cumsum > 0.95
n_components = np.argmax(cumsum > 0.95) + 1
```

### Method 3: Parallel Analysis
```
Compare PCA variance to random data
If eigenvalue > random, keep component
```

## 💡 PCA Intuition

```
Imagine 1000D medical data (1000 genes)

But only 3 genes drive variation:
Gene1: High cancer samples, Low normal
Gene2: High infection samples, Low normal
Gene3: High immune response samples, Low normal

PCA finds these 3 genes automatically!
Reduces 1000D → 3D
```

## 📊 Linear vs Nonlinear

```
Linear data:            Nonlinear data:

● ● ● ●                ●●●●●
●   ●       vs         ●   ●
● ● ● ●                ● ● ●
                       ●   ●
                       ●●●●●

PCA works! ✓            PCA struggles! ✗
                       Need t-SNE, UMAP
```

## ⚠️ Limitations

1. **Assumes linearity**
   - Fails with nonlinear manifolds
   - Solution: t-SNE, UMAP

2. **Orthogonality requirement**
   - May not find interpretable features
   - Solution: ICA, NMF for sparse components

3. **Global structure**
   - Ignores local neighborhoods
   - Solution: Kernel PCA, t-SNE

## 📈 Applications

| Domain | Use Case |
|--------|----------|
| **Visualization** | Plot high-dimensional data in 2D/3D |
| **Preprocessing** | Reduce features before ML |
| **Compression** | Image compression |
| **Noise reduction** | Denoising |
| **Feature extraction** | Automatic feature creation |

## 🔍 PCA vs t-SNE vs UMAP

| Aspect | PCA | t-SNE | UMAP |
|--------|-----|-------|------|
| **Type** | Linear | Nonlinear | Nonlinear |
| **Speed** | Fast | Slow | Medium |
| **Local structure** | No | Yes | Yes |
| **Global structure** | Yes | No | Yes |
| **Interpretability** | High | Low | Medium |
| **Reproducibility** | Yes | No | Mostly |

## 🎓 Learning Outcomes

- [x] Covariance matrix and eigenvalues
- [x] Principal components interpretation
- [x] Explained variance
- [x] Linear vs nonlinear projections
- [x] Component selection methods

## 📚 Key Papers

- **Original**: "Principal Components" (Pearson, 1901)
- **Modern**: "Principal Component Analysis" (Jolliffe, 2002)

## 💪 Advantages

✅ **Fast** - O(min(n,d)×d²)
✅ **Interpretable** - Principal components are directions
✅ **Scalable** - Works with large datasets
✅ **Well-established** - Understood theory
✅ **Preprocessing** - Decorrelates features

## 🚨 Disadvantages

❌ **Linear only** - Misses nonlinear patterns
❌ **Global** - Ignores local structure
❌ **Orthogonality** - Forced perpendicular components
❌ **Interpretability** - Components may not be meaningful

## 💡 Real-World Tips

1. **Always standardize!**
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Use 95% variance rule**
   ```python
   pca = PCA(n_components=0.95)
   ```

3. **Visualize scree plot**
   ```python
   plt.bar(range(min(20, pca.n_components_)),
           pca.explained_variance_ratio_[:20])
   ```

4. **Check interpretability**
   - Do components make sense?
   - If not, try ICA or NMF

---

**Last Updated:** December 2024
**Status:** ✅ Complete
