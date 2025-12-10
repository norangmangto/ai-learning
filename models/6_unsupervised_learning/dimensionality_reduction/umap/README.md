# UMAP (Uniform Manifold Approximation and Projection)

Modern nonlinear dimensionality reduction combining local and global structure.

## 📋 Overview

**Type:** Nonlinear, topology-preserving
**Best For:** Visualization AND preprocessing
**Complexity:** O(n log n)
**Speed:** Faster than t-SNE, slower than PCA

## 🎯 Core Idea

Preserve both local neighborhoods AND global structure.

```
t-SNE approach:         UMAP approach:
Local: ✓ ✓ ✓            Local: ✓ ✓ ✓
Global: ✗              Global: ✓

UMAP: "Keep neighbors close + respect global distances"
```

## 📐 Foundation: Topological Theory

### Riemannian Geometry Concept
```
High dimensions:        Low dimensions:
● ● ●                  ● ● ●
●   ●        →         ●   ●
● ● ●                  ● ● ●

Geodesic distances (along manifold) preserved
Not just Euclidean distances
```

### Simplicial Complex
```
Connect nearby points → Build local topology
Maps to low dimensions while preserving structure
```

## 🚀 Quick Start

```python
import umap
import numpy as np

# Data
X = np.random.randn(1000, 100)

# UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_umap = reducer.fit_transform(X)

# Visualization
import matplotlib.pyplot as plt
plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.5)
plt.show()

# Transform new data
new_X = np.random.randn(100, 100)
new_X_umap = reducer.transform(new_X)
```

## 🎯 Key Hyperparameters

### n_neighbors
```
Controls local neighborhood size
Default: 15

Small (5):                Large (50):
●●●●●                   ●●●●●●●●●●
● ●          vs          ● ●●●●●
● ●                      ● ●●●●●
●●●●●                   ●●●●●●●●●●

Fragmented structure     Connected structure
Focus on tiny clusters   Respects global
More detail             More stability
```

### min_dist
```
Minimum distance between points in low dimensions
Default: 0.1

Small (0.01):            Large (0.5):
●●●●●●                  ●●     ●●
● ●●●●      vs          ●       ●
●●●●●●                  ●●     ●●

Points packed tight      Points spread out
Tight clusters           Dispersed layout
More detail              More global view
```

### metric
```
Distance metric in high dimensions
Default: 'euclidean'

Common:
- 'euclidean': Standard distance
- 'manhattan': L1 distance
- 'cosine': Angular distance (text/embeddings)
- 'correlation': Correlation distance
```

## 💡 Parameter Selection Guide

```
Question: What do you want to emphasize?

Local structure?
├─ Yes, tight clusters → n_neighbors=5-15, min_dist=0.01-0.05
└─ No, global structure → n_neighbors=30-50, min_dist=0.1-0.5

Dataset size?
├─ Small (< 1000) → n_neighbors=5-15
├─ Medium (1k-10k) → n_neighbors=15-30
└─ Large (> 10k) → n_neighbors=30-50

Type of data?
├─ Text/embeddings → metric='cosine'
├─ Images → metric='euclidean'
└─ Biological → metric='correlation'
```

## 📊 UMAP vs t-SNE: Visual Comparison

```
Same dataset visualized differently:

t-SNE result:             UMAP result:
●●●●●                    ●●●●●
  ●●●●●                    ●●●●●
    ●●●●●                    ●●●●●

Interpretation:           Interpretation:
- Clusters clear ✓        - Clusters clear ✓
- Distances arbitrary ✗   - Distances meaningful ✓
- Can't use features ✗    - Can use features ✓
```

## ⚠️ Key Differences from t-SNE

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| **Speed** | O(n²) slow | O(n log n) fast |
| **Local structure** | Perfect | Perfect |
| **Global structure** | Lost | Preserved ✓ |
| **Reproducible** | No | Yes (with seed) |
| **Use for features** | No | Yes ✓ |
| **Scalability** | Poor | Good |

## 📈 Applications

| Domain | Use Case |
|--------|----------|
| **Visualization** | Better than t-SNE (faster, global) |
| **Preprocessing** | Unlike t-SNE, can use UMAP features |
| **Outlier detection** | Isolated points in UMAP space |
| **Clustering** | Hierarchical clustering on UMAP |
| **Embeddings** | Visualize word/image embeddings |

## 🔄 Using UMAP Features vs t-SNE

### t-SNE Features (Bad)
```python
# DON'T: t-SNE for preprocessing
X_tsne = TSNE().fit_transform(X)
clf.fit(X_tsne, y)  # Poor performance!

# Why? t-SNE destroys global structure
```

### UMAP Features (Good!)
```python
# OK: UMAP for preprocessing
reducer = umap.UMAP(n_components=10)
X_umap = reducer.fit_transform(X)
clf.fit(X_umap, y)  # Better performance!

# Why? UMAP preserves structure
```

## 📊 Performance Comparison

```
Dataset: MNIST (70k 28×28 images)

                Time (CPU)    Quality
t-SNE:          45 min        Excellent local
UMAP:           2 min         Excellent local + good global
PCA:            0.1 sec       Fair (linear)

UMAP is ~1000× faster than t-SNE!
While preserving both local and global structure
```

## 🎓 Learning Outcomes

- [x] Topology preservation concept
- [x] Local vs global structure balance
- [x] Hyperparameter effects
- [x] When to use UMAP vs t-SNE
- [x] UMAP for preprocessing (unlike t-SNE)

## 📚 Key Papers

- **Original**: "UMAP: Uniform Manifold Approximation and Projection" (McInnes et al., 2018)

## 💪 Advantages

✅ **Fast** - O(n log n), 100-1000× faster than t-SNE
✅ **Global + local** - Preserves both structures
✅ **Scalable** - Works with millions of points
✅ **Can preprocess** - Preserves information for ML
✅ **Reproducible** - Fixed random seed
✅ **Works on new data** - Can transform unseen points

## 🚨 Disadvantages

❌ **Complex theory** - Harder to understand than t-SNE
❌ **More tuning** - Multiple hyperparameters
❌ **Installation** - Requires numba
❌ **Sensitivity** - Results vary with parameters

## 💡 Real-World Tips

1. **Start with defaults**
   ```python
   reducer = umap.UMAP()
   X_umap = reducer.fit_transform(X)
   # Often works well without tuning
   ```

2. **Use for visualization**
   ```python
   # Much faster than t-SNE
   reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
   X_umap = reducer.fit_transform(X)
   plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels)
   ```

3. **Use for preprocessing**
   ```python
   # Unlike t-SNE, features are meaningful
   reducer = umap.UMAP(n_components=10)
   X_features = reducer.fit_transform(X)
   clf.fit(X_features, y)
   ```

4. **Tune for your data**
   ```python
   # Small data: fewer neighbors
   umap.UMAP(n_neighbors=5, min_dist=0.01)

   # Large data: more neighbors
   umap.UMAP(n_neighbors=50, min_dist=0.1)
   ```

5. **Try different metrics**
   ```python
   # Text embeddings: cosine
   reducer = umap.UMAP(metric='cosine')

   # Images: euclidean
   reducer = umap.UMAP(metric='euclidean')
   ```

## 📊 When to Use Each Reduction Method

```
Choice flowchart:

Need to visualize high-D data?
├─ Yes
│  ├─ Also want global structure?
│  │  ├─ Yes → UMAP (best of both)
│  │  └─ No → t-SNE (very local)
│  └─ Need fast speed? → UMAP
└─ Preprocessing for ML?
   ├─ Yes → UMAP or PCA
   └─ No (pure visualization) → t-SNE or UMAP
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete
