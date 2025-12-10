# Clustering Models

## Overview

Clustering is an unsupervised learning technique that groups similar data points together without predefined labels. This directory contains implementations of various clustering algorithms.

## What is Clustering?

Clustering algorithms partition data into groups (clusters) where:
- **Intra-cluster similarity**: Points within a cluster are similar
- **Inter-cluster dissimilarity**: Points in different clusters are dissimilar

### Common Use Cases
- Customer segmentation
- Image segmentation
- Anomaly detection
- Document clustering
- Gene sequence analysis
- Market basket analysis
- Social network analysis

## Clustering Algorithms Implemented

### 1. K-Means Clustering

**Directory:** `kmeans/`
**Files:** `train_sklearn.py`, `train_pytorch.py`

**Theory:**
Partitions data into K clusters by minimizing within-cluster variance:
```
Objective: min Σ Σ ||x - μk||²
```

**How it Works:**
1. Initialize K centroids (randomly or K-means++)
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat steps 2-3 until convergence

**When to Use:**
- Know approximate number of clusters
- Clusters are spherical/convex
- Similar cluster sizes
- Fast results needed

**Advantages:**
- ✅ Simple and fast
- ✅ Scalable to large datasets
- ✅ Works well with convex clusters
- ✅ Easy to interpret
- ✅ GPU acceleration available (PyTorch)

**Limitations:**
- ❌ Must specify K in advance
- ❌ Sensitive to initialization
- ❌ Assumes spherical clusters
- ❌ Sensitive to outliers
- ❌ Struggles with non-convex shapes

**Key Hyperparameters:**
```python
n_clusters: 2-20          # Number of clusters
init: 'k-means++'         # Initialization method
max_iter: 300             # Maximum iterations
n_init: 10                # Number of runs with different initializations
```

**Choosing K:**
- **Elbow Method**: Plot inertia vs K, look for "elbow"
- **Silhouette Score**: Measure cluster separation
- **Domain Knowledge**: Business requirements
- **Gap Statistic**: Compare with random data

### 2. DBSCAN (Density-Based Spatial Clustering)

**Directory:** `dbscan/`
**File:** `train_sklearn.py`

**Theory:**
Groups points that are closely packed together, marking outliers as noise:
```
Core Point: Has ≥ min_samples within eps radius
Border Point: Within eps of core point
Noise Point: Neither core nor border
```

**How it Works:**
1. For each point, find neighbors within eps radius
2. If point has ≥ min_samples neighbors, mark as core point
3. Connect core points to form clusters
4. Assign border points to nearest cluster
5. Mark remaining points as noise

**When to Use:**
- Don't know number of clusters
- Arbitrary shaped clusters
- Need to detect outliers/noise
- Varying cluster densities

**Advantages:**
- ✅ No need to specify number of clusters
- ✅ Finds arbitrarily shaped clusters
- ✅ Identifies outliers
- ✅ Robust to noise
- ✅ Only 2 parameters

**Limitations:**
- ❌ Sensitive to eps and min_samples
- ❌ Struggles with varying densities
- ❌ Not fully deterministic (border points)
- ❌ High-dimensional data issues

**Key Hyperparameters:**
```python
eps: 0.1-1.0              # Maximum distance between neighbors
min_samples: 3-10         # Minimum points to form dense region
metric: 'euclidean'       # Distance metric
```

**Choosing Parameters:**
- **eps**: Use k-distance graph, look for elbow
- **min_samples**: Rule of thumb: 2 × dimensions
- **Higher eps**: Fewer, larger clusters, less noise
- **Higher min_samples**: Denser clusters, more noise

### 3. Hierarchical Clustering

**Directory:** `hierarchical/`
**Files:** `train_sklearn.py`

**Theory:**
Builds a hierarchy of clusters using bottom-up (agglomerative) or top-down (divisive) approach.

**Agglomerative (Bottom-Up):**
1. Start with each point as its own cluster
2. Merge closest pairs of clusters
3. Repeat until single cluster or desired number

**Linkage Methods:**
- **Single**: Minimum distance between clusters (tends to chain)
- **Complete**: Maximum distance (compact clusters)
- **Average**: Average distance (balanced)
- **Ward**: Minimizes within-cluster variance (best for K-Means-like)

**When to Use:**
- Need hierarchy/taxonomy
- Dendrogram visualization needed
- Don't know number of clusters
- Small to medium datasets

**Advantages:**
- ✅ No need to specify K initially
- ✅ Produces dendrogram (interpretable)
- ✅ Deterministic results
- ✅ Works with any distance metric
- ✅ Handles non-convex shapes (single/average linkage)

**Limitations:**
- ❌ Computationally expensive O(n³)
- ❌ Cannot undo merges
- ❌ Memory intensive for large data
- ❌ Sensitive to noise (single linkage)

**Key Hyperparameters:**
```python
n_clusters: 2-20          # Number of clusters (cut dendrogram)
linkage: 'ward'           # Linkage criterion
metric: 'euclidean'       # Distance metric
```

**Choosing Linkage:**
- **Ward**: General purpose, similar to K-Means
- **Average**: Balanced approach
- **Complete**: Want compact, separate clusters
- **Single**: Non-convex, elongated clusters

### 4. Gaussian Mixture Models (GMM)

**Theory:**
Probabilistic model assuming data comes from mixture of Gaussian distributions:
```
P(x) = Σ πk · N(x|μk, Σk)
```

**When to Use:**
- Soft clustering (probability of membership)
- Elliptical/Gaussian-shaped clusters
- Overlapping clusters
- Probabilistic predictions needed

**Advantages:**
- ✅ Soft clustering (probabilities)
- ✅ Handles elliptical clusters
- ✅ Probabilistic framework
- ✅ Can handle varying cluster shapes

**Limitations:**
- ❌ Must specify number of components
- ❌ Sensitive to initialization
- ❌ Computationally expensive
- ❌ Assumes Gaussian distributions

### 5. Mean Shift

**Theory:**
Iteratively shifts points toward mode of density function.

**When to Use:**
- Don't know number of clusters
- Arbitrary shaped clusters
- Image segmentation

**Advantages:**
- ✅ No need to specify K
- ✅ Finds arbitrary shapes
- ✅ Single parameter (bandwidth)

**Limitations:**
- ❌ Computationally expensive
- ❌ Sensitive to bandwidth
- ❌ Not scalable to large data

### 6. Spectral Clustering

**Theory:**
Uses graph theory and eigenvalues of similarity matrix.

**When to Use:**
- Non-convex clusters
- Clear separation in data
- Small to medium datasets

**Advantages:**
- ✅ Handles non-convex shapes
- ✅ Clear cluster separation
- ✅ Theoretically sound

**Limitations:**
- ❌ Computationally expensive
- ❌ Must specify K
- ❌ Memory intensive
- ❌ Parameter tuning needed

## Quick Comparison

| Algorithm | K Required? | Shape | Outliers | Speed | Scalability |
|-----------|-------------|-------|----------|-------|-------------|
| K-Means | Yes | Convex | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| DBSCAN | No | Any | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Hierarchical | Optional | Any | ❌ | ⭐⭐ | ⭐⭐ |
| GMM | Yes | Elliptical | ❌ | ⭐⭐⭐ | ⭐⭐⭐ |
| Mean Shift | No | Any | ✅ | ⭐⭐ | ⭐⭐ |
| Spectral | Yes | Non-convex | ❌ | ⭐⭐ | ⭐ |

## Evaluation Metrics

### 1. Silhouette Score
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
# Range: [-1, 1], higher is better
# > 0.7: Strong structure
# > 0.5: Reasonable structure
# > 0.25: Weak structure
# < 0: Wrong clustering
```

### 2. Davies-Bouldin Index
```python
from sklearn.metrics import davies_bouldin_score
score = davies_bouldin_score(X, labels)
# Lower is better, 0 is perfect
```

### 3. Calinski-Harabasz Score
```python
from sklearn.metrics import calinski_harabasz_score
score = calinski_harabasz_score(X, labels)
# Higher is better
```

### 4. Within-Cluster Sum of Squares (WCSS)
```python
# K-Means inertia
wcss = kmeans.inertia_
# Lower is better
```

### 5. Adjusted Rand Index (ARI)
```python
from sklearn.metrics import adjusted_rand_score
# Only if true labels available
ari = adjusted_rand_score(y_true, y_pred)
# Range: [-1, 1], 1 is perfect
```

## Best Practices

### 1. Data Preprocessing

**Always Standardize:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Handle Missing Values:**
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

**Dimensionality Reduction (if high-dimensional):**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)  # or explained_variance=0.95
X_reduced = pca.fit_transform(X)
```

### 2. Choosing the Right Algorithm

**Start with K-Means if:**
- Have approximate number of clusters
- Clusters are roughly spherical
- Need fast results
- Large dataset

**Use DBSCAN if:**
- Don't know number of clusters
- Have arbitrary shaped clusters
- Need to detect outliers
- Varying cluster sizes

**Use Hierarchical if:**
- Need dendrogram/hierarchy
- Small dataset (< 10k samples)
- Want deterministic results
- Exploring different K values

**Use GMM if:**
- Need soft clustering (probabilities)
- Elliptical cluster shapes
- Overlapping clusters
- Bayesian framework needed

### 3. Parameter Tuning

**K-Means:**
```python
from sklearn.model_selection import GridSearchCV
# Try different K values
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    print(f"K={k}: Inertia={kmeans.inertia_}")
```

**DBSCAN:**
```python
from sklearn.neighbors import NearestNeighbors
# Find optimal eps using k-distance graph
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X)
distances, _ = neighbors.kneighbors(X)
distances = np.sort(distances[:, k-1], axis=0)
# Plot and find elbow
```

### 4. Validation Strategies

**Internal Validation:**
```python
# Silhouette analysis
from sklearn.metrics import silhouette_samples
silhouette_vals = silhouette_samples(X, labels)

# Plot silhouette diagram
for i in range(n_clusters):
    cluster_vals = silhouette_vals[labels == i]
    cluster_vals.sort()
    plt.barh(range(len(cluster_vals)), cluster_vals)
```

**Stability Analysis:**
```python
# Run multiple times with different initializations
scores = []
for _ in range(10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    scores.append(silhouette_score(X, kmeans.labels_))

print(f"Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

## Common Issues and Solutions

### Issue: Poor Cluster Quality
**Solutions:**
- ✅ Try different algorithms
- ✅ Standardize features
- ✅ Remove outliers
- ✅ PCA for high dimensions
- ✅ Feature engineering

### Issue: K-Means Converging to Local Optima
**Solutions:**
- ✅ Use k-means++ initialization
- ✅ Increase n_init (try more random starts)
- ✅ Try different K values
- ✅ Use GMM instead

### Issue: DBSCAN Finds Too Much Noise
**Solutions:**
- ✅ Decrease min_samples
- ✅ Increase eps
- ✅ Standardize features
- ✅ Remove extreme outliers first

### Issue: Hierarchical Too Slow
**Solutions:**
- ✅ Sample data first
- ✅ Use K-Means for initial clustering
- ✅ Try mini-batch approach
- ✅ Switch to BIRCH algorithm

### Issue: High-Dimensional Data
**Solutions:**
- ✅ Apply PCA/t-SNE first
- ✅ Feature selection
- ✅ Use cosine distance
- ✅ Subspace clustering methods

## Example Workflows

### K-Means Complete Pipeline
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Find optimal K
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# 3. Choose best K
best_k = np.argmax(silhouettes) + 2

# 4. Final model
kmeans = KMeans(n_clusters=best_k, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 5. Evaluate
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")
```

### DBSCAN for Anomaly Detection
```python
from sklearn.cluster import DBSCAN

# 1. Standardize
X_scaled = StandardScaler().fit_transform(X)

# 2. Train DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# 3. Extract anomalies
anomalies = X[labels == -1]
normal = X[labels != -1]

print(f"Anomalies detected: {len(anomalies)}")
```

## Further Reading

- [K-Means Clustering (MacQueen, 1967)](https://projecteuclid.org/euclid.bsmsp/1200512992)
- [DBSCAN Paper (Ester et al., 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
- [Scikit-Learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Cluster Analysis (Kaufman & Rousseeuw)](https://www.wiley.com/en-us/Finding+Groups+in+Data%3A+An+Introduction+to+Cluster+Analysis-p-9780470317488)

## Next Steps

1. Try **Dimensionality Reduction** (../../dimensionality_reduction/) before clustering
2. Explore **Anomaly Detection** (../../anomaly_detection/) for outlier detection
3. Study **Semi-Supervised Learning** for partial labels
4. Learn **Deep Clustering** methods (autoencoders + clustering)
5. Apply to real datasets (customer segmentation, image segmentation)
