# DBSCAN Clustering

Density-based clustering that discovers arbitrary-shaped clusters and outliers.

## 📋 Overview

**Type:** Density-based
**K required:** No
**Outliers:** Automatic detection
**Complexity:** O(n log n) with spatial indexing
**Best For:** Non-spherical clusters, automatic outlier detection

## 🎯 Core Idea

Clusters are dense regions separated by sparse regions.

```
DBSCAN's view:        K-Means's view:

●●●       ●          ●●● ●
● ●   ●●●●●          ●   ●●●●
●●●   ●●●●           ●●● ●●●●
  ●●●       ●

Dense regions = clusters
Sparse points = outliers

K-Means forces all into K clusters
DBSCAN finds natural groupings
```

## 📐 Definitions

### Epsilon-neighborhood
$$N_\epsilon(p) = \{q : d(p, q) \leq \epsilon\}$$

All points within distance $\epsilon$ from $p$.

### Core Point
Point $p$ is core if $|N_\epsilon(p)| \geq \text{MinPts}$

Has enough neighbors to define cluster.

### Border Point
Not core, but in $\epsilon$-neighborhood of core point.

### Outlier/Noise Point
Not core and not border point.

## 🔄 Algorithm

```
1. Find all core points
   (points with ≥ MinPts neighbors within ε)

2. Form clusters by connecting core points
   If two core points are within ε, same cluster

3. Add border points to clusters
   Assign to cluster of nearby core point

4. Mark remaining points as outliers
   (noise)
```

## 📊 Visualization

```
ε = radius around each point
MinPts = 3 (minimum neighbors)

Core points (≥3 neighbors):     Border points:        Outliers:
    ●●●                              ◯                    ○
  ●   ●●        vs    ●●●●●●    vs    ◯
    ●●●           ●●●   ○             ◯

Core can connect! Adds borders!  Isolated points!
```

## 🚀 Quick Start

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Data
X = np.random.randn(300, 2)

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1)

print(f"Clusters: {n_clusters}")
print(f"Outliers: {n_outliers}")

# Label -1 = outlier
outlier_mask = labels == -1
```

## 🎯 Choosing Parameters

### Epsilon (ε)

#### K-distance Graph Method
```python
from sklearn.neighbors import NearestNeighbors

# k-distance where k = MinPts
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)
distances = np.sort(distances[:, 4], axis=0)

# Plot distances - look for "elbow"
plt.plot(distances)
# Elbow point ≈ ε value
plt.show()
```

```
Distance plot:
       ↑
       │     ╱    ← Outliers (steep rise)
       │   ╱
       │ ╱        ← Elbow here! ε ≈ 0.5
       └─────────→ Point index

Outliers cause sharp increase in distance
Elbow marks transition to core/border points
```

#### Distance Distribution
```python
# Calculate distances to k-th nearest neighbor
distances = np.sort(distances[:, 4])

# Visual inspection
plt.hist(distances, bins=50)
plt.xlabel('Distance to 5th neighbor')
# Natural gap = good ε threshold
```

### MinPts

**Rule of thumb:**
```
MinPts = 2 × dimensions

For 2D data: MinPts = 4
For 3D data: MinPts = 6
For 10D data: MinPts = 20
```

**Or:** Use k from k-distance graph (typically 4-5)

## 📊 Parameter Sensitivity

```
ε too small:          ε too large:
Almost all outliers   Everything one cluster

●  ●  ●  ●           ●●●●●●
●  ●  ●  ●     vs    ●●●●●●
●  ●  ●  ●           ●●●●●●

MinPts too small:     MinPts too large:
Every point core      Almost all outliers

●●●●●●●              ●  ●  ●  ●
●●●●●●●       vs     ●  ●  ●  ●
●●●●●●●              ●  ●  ●  ●
```

## 💡 Density Intuition

```
Idea: Clusters are dense, surrounded by sparse regions

Low density cluster:          High density cluster:
  ● ● ● ●                      ●●●●●●
  ●   ●         vs             ●●●●●●
  ● ● ● ●                      ●●●●●●

DBSCAN can find both if ε is appropriate!
Advantage over K-Means which forces spherical shapes
```

## 📈 Applications

| Domain | Use Case |
|--------|----------|
| **Spatial data** | Finding geographic clusters |
| **Anomaly** | Automatic outlier detection |
| **Gene expression** | Variable-sized clusters |
| **Traffic** | Congestion regions |
| **Social media** | Community detection |

## 🔍 DBSCAN vs K-Means

| Aspect | DBSCAN | K-Means |
|--------|--------|---------|
| **K required** | No | Yes |
| **Cluster shape** | Any | Spherical |
| **Outliers** | Automatic | Forced in clusters |
| **Speed** | O(n log n) | O(nk) |
| **Scalability** | Good | Excellent |
| **Parameter tuning** | Medium | Easy |

## 🎓 Learning Outcomes

- [x] Core, border, noise points
- [x] Epsilon and MinPts parameters
- [x] Parameter selection methods
- [x] Density-based vs partition-based
- [x] Automatic outlier detection

## 📚 Key Papers

- **Original**: "A Density-Based Algorithm for Discovering Clusters" (Ester et al., 1996)

## 💪 Advantages

✅ **No K needed** - Automatically determines clusters
✅ **Any shape** - Finds non-spherical clusters
✅ **Outlier detection** - Automatic noise identification
✅ **Scalable** - O(n log n) with spatial indexing
✅ **Principled** - Density-based, interpretable

## 🚨 Disadvantages

❌ **Parameter tuning** - Difficult for new datasets
❌ **Varying densities** - Poor with density variations
❌ **High dimensions** - Curse of dimensionality
❌ **Sparse data** - Many outliers detected

## 💡 Real-World Tips

1. **Always use k-distance graph**
   ```python
   # Plot distances to see natural ε
   neighbors = NearestNeighbors(n_neighbors=5)
   neighbors.fit(X)
   distances, _ = neighbors.kneighbors(X)
   distances = np.sort(distances[:, -1])
   plt.plot(distances)
   ```

2. **Start with MinPts = 2×d**
   ```python
   d = X.shape[1]
   min_pts = 2 * d
   ```

3. **Standardize features**
   ```python
   from sklearn.preprocessing import StandardScaler
   X_scaled = StandardScaler().fit_transform(X)
   ```

4. **Check outlier percentage**
   - 0-5% outliers: reasonable
   - >10% outliers: ε might be too small

---

**Last Updated:** December 2024
**Status:** ✅ Complete
