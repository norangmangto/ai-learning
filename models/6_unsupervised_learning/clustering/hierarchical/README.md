# Hierarchical Clustering

Build a hierarchy of clusters through agglomerative or divisive approach.

## 📋 Overview

**Type:** Agglomerative (bottom-up)
**Dendrogram:** Visual tree of clusters
**Complexity:** O(n²) to O(n³)
**Best For:** Understanding cluster relationships, variable number of clusters

## 🏗️ Agglomerative Approach

### Algorithm
```
1. Start: Each point is its own cluster
   {1}, {2}, {3}, {4}, {5}

2. Merge closest pair
   {1,2}, {3}, {4}, {5}

3. Merge closest pair
   {1,2}, {3,4}, {5}

4. Merge closest pair
   {1,2}, {3,4,5}

5. Continue until one cluster
   {1,2,3,4,5}
```

### Distance Metrics

**Single Linkage** (Minimum distance)
```
d({A}, {B}) = min(d(a,b)) for a∈A, b∈B
              │
              └─ Connects closest points
```
⚠️ Forms chains (not ideal)

**Complete Linkage** (Maximum distance)
```
d({A}, {B}) = max(d(a,b)) for a∈A, b∈B
              │
              └─ Connects farthest points
```
✅ Compact, well-separated clusters

**Average Linkage** (Average distance)
```
d({A}, {B}) = mean(d(a,b)) for a∈A, b∈B
              │
              └─ Balanced approach
```
✅ Most popular

**Ward Linkage** (Minimize variance)
```
d({A}, {B}) = increase in sum of squared distances
              when merging A and B
              │
              └─ Matches K-means criteria
```
✅ Produces compact clusters

## 📊 Dendrogram Visualization

```
Height (distance between clusters)
    │
5.0 ├─────┬─────┐
    │     │     │
4.0 │   ┌─┴─┐   │
    │   │   │   │
3.0 │ ┌─┴─┐ └─┬─┘
    │ │   │   │
2.0 ├─┴┬─┐├───┴──
    │ │ │ ││
1.0 │ │ │ ││
    └─┴─┴─┴┘
     1 3 5 2 4    ← Points

Reading dendrogram:
- Horizontal distance = dissimilarity
- Cut at h=2.5 → 3 clusters: {1,3}, {5,2}, {4}
- Cut at h=1.5 → 5 clusters: {1}, {3}, {5}, {2}, {4}
```

## 🚀 Quick Start

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Data
X = np.random.randn(100, 2)

# Hierarchical clustering
clusterer = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'  # or 'complete', 'average', 'single'
)
labels = clusterer.fit_predict(X)

# Dendrogram
Z = linkage(X, method='ward')
dendrogram(Z)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.show()

# Cut dendrogram at specific height
from scipy.cluster.hierarchy import fcluster
cluster_labels = fcluster(Z, t=threshold, criterion='distance')
```

## 🔍 Choosing Linkage Method

```
Decision tree:

Want compact clusters?
├─ Yes → Ward linkage ✅ (matches K-means)
└─ No  → Average linkage

Allow chains?
├─ Yes → Single linkage (rare, chaining issues)
└─ No  → Complete linkage (too restrictive)

Default recommendation: Ward
```

## 📈 Single vs Complete Linkage

```
Complete Linkage (worst-case distance):
    ●           ●
  ●   ●       ●   ●
Merges when farthest pair is close
→ Well-separated, roughly equal-sized

Single Linkage (best-case distance):
    ●           ●
  ●   ●       ●   ●
Merges when closest pair is close
→ Forms long chains (not ideal for most cases)
```

## 📊 Choosing Number of Clusters

### Method 1: Dendrogram Visual Inspection
```
         ↑ Large gaps in dendrogram
         │ = good cut points
     ┌───┴───┐
     │       │
   ┌─┴─┐   ┌─┴─┐
   │   │   │   │  ← Cut here for K=2, 3, or 4
```

### Method 2: Distance Threshold
```python
# Cut dendrogram at distance threshold
cluster_labels = fcluster(Z, t=5.0, criterion='distance')
n_clusters = len(np.unique(cluster_labels))
```

### Method 3: Elbow Method
```python
# Last K merges show largest distance increases
last_k = 10
last_distances = Z[-last_k:, 2]
plt.plot(last_distances)  # Look for elbow
```

## 💡 Dendrogram Interpretation

```
Dendrogram for customer segmentation:

Height
    │
    ├─ Large jump ← Different segments!
    │
    ├─ Small jumps ← Similar customers within segment
    │

Indicates:
- Clear 2-3 customer segments
- No clear 10-cluster structure
```

## ⚠️ Limitations

1. **Cannot handle large datasets**
   - O(n²) to O(n³) complexity
   - Solution: Mini-batch approximations

2. **Hard to choose K**
   - Must cut dendrogram at some height
   - Somewhat subjective

3. **Irreversible**
   - Once merged, clusters can't be split
   - Solution: Agglomerative always better than divisive

4. **Sensitive to outliers**
   - Can affect linkage distances
   - Solution: Remove outliers or use robust distances

## 🎯 Applications

| Domain | Use Case |
|--------|----------|
| **Gene sequencing** | Phylogenetic trees |
| **Social networks** | Community detection |
| **Customer segments** | Understanding relationships |
| **Image segmentation** | Hierarchical regions |
| **Taxonomy** | Biological classification |

## 📊 Hierarchical vs Flat Clustering

| Aspect | Hierarchical | K-Means |
|--------|-----------|---------|
| **Structure** | Tree (dendrogram) | Flat (K clusters) |
| **K needed** | No (can cut anywhere) | Yes |
| **Scalability** | Poor (O(n²)) | Excellent (O(nk)) |
| **Interpretability** | Good (see relationships) | Simple |
| **Speed** | Slow | Fast |

## 🔄 Divisive Approach (Top-Down)

```
Rare, but exists:

1. Start with all points in one cluster
2. Split into 2 clusters
3. Recursively split until single points
4. Build tree from top-down

Why rare?
- More expensive (exponential splits)
- No clear split criterion
- Less useful for most applications
```

## 🎓 Learning Outcomes

- [x] Agglomerative hierarchical clustering
- [x] Different linkage methods
- [x] Dendrogram interpretation
- [x] How to choose number of clusters
- [x] Pros and cons vs K-means

## 📚 Key Papers

- **Original**: "The structure of a cluster" (Jardine & Sibson, 1971)
- **Ward Linkage**: "Hierarchical Grouping for Optimization" (Ward, 1963)

## 💪 Advantages

✅ **No K needed** - Choose clusters from dendrogram
✅ **Interpretable** - See hierarchical relationships
✅ **Deterministic** - Same result every run
✅ **Versatile** - Multiple linkage options

## 🚨 Disadvantages

❌ **Slow** - O(n²) or O(n³) complexity
❌ **Memory intensive** - Stores all distances
❌ **Irreversible** - Bad early merges cannot be undone
❌ **Not for big data** - Limited to thousands of points

## 💡 Real-World Tips

1. **For large datasets**
   - Use K-means or DBSCAN instead
   - Or subsample data for dendrogram

2. **Always visualize dendrogram**
   ```python
   from scipy.cluster.hierarchy import dendrogram
   plt.figure(figsize=(10, 5))
   dendrogram(Z)
   plt.show()
   ```

3. **Use Ward linkage by default**
   - Produces compact, meaningful clusters
   - Matches K-means objective

4. **Cut dendrogram at natural gaps**
   - Look for large jumps in distance
   - Usually indicates true cluster boundaries

---

**Last Updated:** December 2024
**Status:** ✅ Complete
