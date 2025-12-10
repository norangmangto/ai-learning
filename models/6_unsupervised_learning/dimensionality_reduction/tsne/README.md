# t-SNE (t-Distributed Stochastic Neighbor Embedding)

Nonlinear dimensionality reduction for visualizing high-dimensional data structure.

## 📋 Overview

**Type:** Nonlinear, probabilistic
**Best For:** Visualization, cluster discovery
**Complexity:** O(n²)
**Warning:** Use only for visualization, not preprocessing!

## 🎯 Core Idea

Preserve local neighborhood structure while mapping to low dimensions.

```
High dimensions:            Low dimensions (t-SNE):

Nearby points:              Nearby points:
● near ●                    ● near ●

Distant points:             Distant points:
●     ●                     ●     ●
(can be anywhere)

t-SNE: "Keep neighbors close, push distant apart"
```

## 📐 Key Concept: Perplexity

### Perplexity
```
Controls how many neighbors each point should have
Typical range: 5-50

Low perplexity (5):          High perplexity (50):
●●●●                        ●●●●●●●
● ●                         ● ●●●
● ●                  vs     ● ●●●
● ●                         ● ●●●
●●●●                        ●●●●●●●

Fragmented clusters         Connected structure
Small neighborhoods         Large neighborhoods
```

## 🔄 Algorithm

```
1. Compute pairwise distances in high dimensions
   D_high = distance matrix (n × n)

2. Convert to probabilities
   p_ij = P(point i chooses point j as neighbor)

3. Initialize random low-dimensional positions
   y_i ~ random (d dimensions)

4. Compute low-dimensional probabilities
   q_ij = P(point i chooses point j in low-D)

5. Minimize KL divergence
   KL(p || q) → Use gradient descent
   (Make q_ij similar to p_ij)

6. Iterate until convergence
```

## 📊 KL Divergence Intuition

```
High-D: "Points A and B are close neighbors"
p_AB = 0.8

Low-D: "We placed them far apart"
q_AB = 0.1

KL divergence: High! Cost = penalty
→ Gradient descent pulls them closer

High-D: "Points A and C are far"
p_AC = 0.01

Low-D: "We placed them close"
q_AC = 0.2

KL divergence: High! Cost = penalty
→ Gradient descent pushes them apart
```

## 🚀 Quick Start

```python
from sklearn.manifold import TSNE
import numpy as np

# Data (high-dimensional)
X = np.random.randn(1000, 100)  # 1000 samples, 100 features

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X)

# Visualization
import matplotlib.pyplot as plt
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title('t-SNE Visualization')
plt.show()
```

## 🎯 Hyperparameter Tuning

### Perplexity

```
Rule of thumb: perplexity = 5-50 (or n/100)

Dataset size:
- Small (< 1000): perplexity = 5-10
- Medium (1k-10k): perplexity = 20-30
- Large (> 10k): perplexity = 30-50

Effect:
Low perplexity:         High perplexity:
Fragmented structure    Smooth structure
Small neighborhoods     Large neighborhoods
Noisy                   Cleaner
```

### Learning Rate

```python
tsne = TSNE(learning_rate=200)

Too low:                Too high:
Slow convergence        Overshoots optimum
May get stuck           Unstable

Default 200 usually good
Try 100-300 for tuning
```

### Iterations

```python
# Convergence indicators
tsne = TSNE(n_iter=1000)

Low iterations:         High iterations:
Incomplete structure    Well-converged
Blurry clusters        Clear clusters

Default 1000 usually sufficient
Increase if still converging
```

## 💡 t-SNE Characteristics

### Strength: Local Structure Preservation
```
High-D data structure:
Cluster A: [1, 2, 3, 4, 5]
Cluster B: [6, 7, 8, 9, 10]
Cluster C: [11, 12, 13, 14, 15]

t-SNE output:
●●●●● (points 1-5 together)
  ●●●●● (points 6-10 together)
    ●●●●● (points 11-15 together)

Neighbors stay neighbors! ✓
```

### Weakness: Global Structure

```
True data:              t-SNE result:
●●●●●                 ●●●●●
     ●●●●●       or        ●●●●●
          ●●●●●          ●●●●●

Distance between clusters = arbitrary!
May show false relationships
```

## ⚠️ Common Pitfalls

1. **Using for preprocessing**
   ```python
   # WRONG: Using t-SNE features for ML
   X_tsne = TSNE().fit_transform(X)
   clf.fit(X_tsne, y)  # BAD!

   # RIGHT: Use for visualization only
   # Use original X for ML
   ```
   t-SNE destroys global structure needed for learning.

2. **Over-interpreting distance**
   ```
   t-SNE shows clusters but:
   - Doesn't mean they're separated in original space
   - Distance between clusters is arbitrary
   - Only trust local neighborhoods
   ```

3. **Ignoring randomness**
   ```python
   # Different runs = different results!
   tsne1 = TSNE(random_state=0).fit_transform(X)
   tsne2 = TSNE(random_state=1).fit_transform(X)
   # Results look different but valid
   ```

4. **Wrong perplexity for data size**
   ```
   1000 samples with perplexity=5 → fragmented
   100 samples with perplexity=50 → doesn't work

   Balance needed!
   ```

## 📈 Applications

| Domain | Use Case |
|--------|----------|
| **Clustering** | Visualize cluster quality |
| **Outlier** | Spot unusual points |
| **Class separation** | Check classification difficulty |
| **Gene expression** | Visualize biological samples |
| **Image embeddings** | See what deep networks learn |

## 🔍 t-SNE vs PCA vs UMAP

| Aspect | PCA | t-SNE | UMAP |
|--------|-----|-------|------|
| **Type** | Linear | Nonlinear | Nonlinear |
| **Speed** | Fast | Slow (O(n²)) | Medium |
| **Local structure** | No | Excellent | Excellent |
| **Global structure** | Yes | No | Yes |
| **Reproducibility** | Yes | No | Mostly |
| **For preprocessing** | Yes | No | Maybe |
| **For visualization** | OK | Best | Great |

## 🎓 Learning Outcomes

- [x] Perplexity and its role
- [x] KL divergence minimization
- [x] Local vs global structure
- [x] Common pitfalls (preprocessing, distance)
- [x] Hyperparameter selection

## 📚 Key Papers

- **Original**: "t-SNE: Visualizing High-Dimensional Data" (van der Maaten & Hinton, 2008)

## 💪 Advantages

✅ **Excellent visualization** - Reveals structure clearly
✅ **Nonlinear** - Handles complex manifolds
✅ **Interpretable** - Clusters are visual
✅ **No parameters** - Perplexity usually set automatically

## 🚨 Disadvantages

❌ **Slow** - O(n²) computation
❌ **Non-deterministic** - Randomness in algorithm
❌ **Poor global structure** - Distances arbitrary
❌ **Not for preprocessing** - Destroys information
❌ **High variance** - Different runs look different

## 💡 Real-World Tips

1. **Use for visualization, not preprocessing**
   ```python
   # Good: Understand data
   plt.scatter(*TSNE().fit_transform(X).T)

   # Bad: Features for ML
   X_features = TSNE().fit_transform(X)  # DON'T!
   ```

2. **Set perplexity thoughtfully**
   ```python
   # Rule: 5 to 50, or n/100
   n_samples = X.shape[0]
   perplexity = min(50, max(5, n_samples // 100))
   ```

3. **Use fixed random seed for reproducibility**
   ```python
   tsne = TSNE(random_state=42)
   ```

4. **Combine with other methods**
   ```python
   # First PCA to reduce noise
   from sklearn.decomposition import PCA
   X_pca = PCA(n_components=50).fit_transform(X)

   # Then t-SNE for visualization
   X_tsne = TSNE().fit_transform(X_pca)
   ```

5. **Interpret carefully**
   - Clusters are real ✓
   - Cluster distances are not ✗
   - Relationships to other clusters are not ✗

---

**Last Updated:** December 2024
**Status:** ✅ Complete
