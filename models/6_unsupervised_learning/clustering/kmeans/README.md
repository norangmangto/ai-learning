# K-Means Clustering

Partition-based clustering algorithm that groups data into K clusters.

## 📋 Overview

**Type:** Centroid-based, unsupervised
**Clusters:** K (specified)
**Complexity:** O(n·k·i·d) where i = iterations
**Best For:** Large datasets, spherical clusters

## 🎯 Algorithm

### Initialization
```
1. Randomly select K points as initial centroids
   C = {c_1, c_2, ..., c_k}
```

### Iteration
```
2. Assign each point to nearest centroid
   S_i = {x | d(x, c_i) < d(x, c_j) for all j ≠ i}

3. Update centroids to cluster means
   c_i = mean(S_i)

4. Repeat until convergence (centroids don't change)
```

## 📐 Mathematical Formulation

### Objective Function
$$\text{minimize} \sum_{i=1}^{k} \sum_{x \in S_i} ||x - c_i||^2$$

Each point contributes squared distance to its nearest centroid.

### Update Rules

**Assignment step:**
$$S_i^{(t)} = \{x : ||x - c_i^{(t-1)}||^2 \leq ||x - c_j^{(t-1)}||^2 \text{ for all } j\}$$

**Update step:**
$$c_i^{(t)} = \frac{1}{|S_i^{(t)}|} \sum_{x \in S_i^{(t)}} x$$

## 🔄 Iteration Example

```
Data: Points in 2D

ITERATION 1:
Initial centroids: [A, B, C]
Points assigned to nearest centroid

X . X . .      Centroid A
. . X X .      Centroid B
. X . . X      Centroid C

ITERATION 2:
Centroids move to cluster means

. X . . .      New centroid A
X X X . .      New Centroid B
. . . X X      New Centroid C

ITERATION 3:
Converged! (Centroids don't move much)
```

## 📊 Visualization

```
Input data:         After K-means (K=3):
● ● ● ○ ○          ● ● ● ○ ○
● ● ● ○ ○          ● ● ● ○ ○
● ● ○ ○ ○          ○ ○ ○ ○ ○
                   Red  Purple Blue
                   Cluster 1, 2, 3
```

## 🚀 Quick Start

```python
from sklearn.cluster import KMeans
import numpy as np

# Data
X = np.random.randn(300, 2)

# Fit K-means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Centroids
centroids = kmeans.cluster_centers_

# Inertia (sum of squared distances)
inertia = kmeans.inertia_  # Lower is better

# Predict new points
new_points = np.random.randn(10, 2)
predictions = kmeans.predict(new_points)

# Visualization
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='X', s=200, c='red', edgecolors='black')
plt.show()
```

## 🎯 Choosing K

### Elbow Method
```python
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot inertias - look for "elbow"
plt.plot(range(1, 10), inertias)
# Elbow usually at optimal K
```

### Silhouette Score
```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

# Highest score = best K
optimal_k = np.argmax(scores) + 2
```

### Decision Guide
```
Dataset size?
├─ Small (< 1000): K = √(n/2)
├─ Medium (1k-10k): Try K = 5-10
└─ Large (> 10k): Try K = 10-100

Domain knowledge?
└─ Yes: Use known K (e.g., 3 customer segments)
```

## ⚠️ Limitations & Issues

1. **Random initialization**
   - Different runs = different results
   - Solution: `n_init=10` (default in modern sklearn)

2. **Spherical clusters assumption**
   - Works poorly with elongated/irregular clusters
   - Solution: Try DBSCAN or spectral clustering

3. **Sensitivity to outliers**
   - Centroids pulled by extreme points
   - Solution: Remove outliers or use K-medoids

4. **Must specify K**
   - Doesn't automatically determine clusters
   - Solution: Use elbow/silhouette methods

5. **Local optima**
   - May not find global minimum
   - Solution: Use `n_init=20+` or kmeans++ initialization

## 💡 K-Means++ Initialization

Instead of random:
```
1. Pick first centroid randomly
2. For each remaining centroid:
   - Probability ∝ distance² to nearest centroid
   - Spreads out initial centroids
   - Better convergence
```

```python
# Automatic in sklearn
kmeans = KMeans(n_clusters=3, init='k-means++')
```

## 📈 Applications

| Domain | Use Case |
|--------|----------|
| **Retail** | Customer segmentation |
| **Healthcare** | Patient grouping |
| **Image compression** | Color quantization (K colors) |
| **Recommendation** | User clustering |
| **Genomics** | Gene expression clustering |

## 🔍 K-Means vs Alternatives

| Aspect | K-Means | DBSCAN | Hierarchical |
|--------|---------|--------|-------------|
| K required | Yes | No | No |
| Speed | Fast | Moderate | Slow |
| Cluster shape | Spherical | Any | Any |
| Scalability | Excellent | Good | Poor |
| Outliers | Sensitive | Robust | Sensitive |

## 📊 Complexity Analysis

```
Time: O(n·k·i·d)
  n = number of points
  k = number of clusters
  i = iterations (usually 5-20)
  d = dimensions

Space: O(n + k)
```

**Practical:** Very efficient for large datasets

## 🎓 Learning Outcomes

- [x] K-means algorithm (assignment + update)
- [x] Objective function (sum of squared distances)
- [x] How to choose optimal K
- [x] Advantages and limitations
- [x] Comparison with other clustering methods

## 📚 Key Papers

- **Original**: "K-Means Clustering" (MacQueen, 1967)
- **K-Means++**: "k-means++: The advantages of careful seeding" (Arthur & Vassilvitskii, 2007)

## 💪 Advantages

✅ **Simple & interpretable** - Easy to understand and implement
✅ **Fast** - Linear time with respect to number of points
✅ **Scalable** - Handles millions of points
✅ **Automatic** - No threshold parameters
✅ **Versatile** - Works with any distance metric

## 🚨 Disadvantages

❌ **Must specify K** - Requires prior knowledge
❌ **Random initialization** - Can converge to local optima
❌ **Spherical assumption** - Fails with non-convex clusters
❌ **Sensitive to scale** - Features need normalization
❌ **Outlier sensitive** - Centroids affected by extremes

## 💡 Real-World Tips

1. **Always normalize data**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Try multiple K values**
   - Use elbow + silhouette
   - Also consider business logic

3. **Use k-means++**
   ```python
   kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20)
   ```

4. **Check convergence**
   ```python
   print(kmeans.n_iter_)  # Should be reasonable (< 50)
   ```

5. **Visualize results**
   - Plot clusters in 2D (PCA if needed)
   - Check if clusters make sense

---

**Last Updated:** December 2024
**Status:** ✅ Complete
