# Dimensionality Reduction

Dimensionality reduction techniques transform high-dimensional data into lower dimensions while preserving important structure. This is crucial for visualization, noise reduction, feature extraction, and improving model performance.

## Overview

| Method | Type | Speed | Preserves | Use Case |
|--------|------|-------|-----------|----------|
| **PCA** | Linear | ⚡⚡⚡ Fast | Global structure, variance | General purpose, preprocessing |
| **t-SNE** | Non-linear | 🐌 Slow | Local structure, clusters | Visualization only |
| **UMAP** | Non-linear | ⚡⚡ Fast | Local + Global structure | Visualization & preprocessing |

## 1. PCA (Principal Component Analysis)

### Theory

PCA is a linear transformation that projects data onto orthogonal axes (principal components) that capture maximum variance.

**Mathematical Formulation:**
- Given data matrix X (n × p), find orthogonal directions with maximum variance
- Eigenvalue decomposition: `X^T X v = λv`
- Principal components are eigenvectors sorted by eigenvalues
- Transformed data: `Z = XV` where V is the matrix of eigenvectors

**Key Properties:**
- Linear transformation
- Preserves global structure
- Components are orthogonal
- Variance maximization

### When to Use PCA

✅ **Good for:**
- High-dimensional data preprocessing
- Noise reduction
- Speeding up training (fewer features)
- Data compression
- Multicollinearity removal
- When features are correlated

❌ **Avoid when:**
- Non-linear relationships are important
- Interpretability is critical (PCs are combinations)
- Data has no redundancy

### Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Always scale first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

### Choosing Number of Components

**Method 1: Explained Variance Threshold**
```python
pca = PCA(n_components=0.95)  # Keep 95% variance
```

**Method 2: Scree Plot (Elbow Method)**
```python
pca = PCA()
pca.fit(X_scaled)
plt.plot(pca.explained_variance_ratio_)
# Look for "elbow" where variance drops significantly
```

**Method 3: Kaiser Criterion**
```python
# Keep components with eigenvalue > 1
n_components = sum(pca.explained_variance_ > 1)
```

### PCA Variants

#### Standard PCA
```python
pca = PCA(n_components=10)
```
- Best for: Most use cases
- Memory: O(n × p)

#### Incremental PCA
```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=10, batch_size=100)
ipca.fit(X)  # Can fit in batches
```
- Best for: Large datasets that don't fit in memory
- Memory: O(batch_size × p)

#### Kernel PCA
```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X)
```
- Best for: Non-linear patterns
- Kernels: 'rbf', 'poly', 'sigmoid', 'cosine'
- Slower than standard PCA

### Best Practices

1. **Always scale your data first**
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Check explained variance**
   ```python
   cumsum = np.cumsum(pca.explained_variance_ratio_)
   print(f"90% variance with {np.argmax(cumsum >= 0.9) + 1} components")
   ```

3. **Visualize in 2D/3D**
   ```python
   pca = PCA(n_components=2)
   X_2d = pca.fit_transform(X_scaled)
   plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
   ```

4. **Use with pipelines**
   ```python
   from sklearn.pipeline import Pipeline

   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('pca', PCA(n_components=0.95)),
       ('classifier', RandomForestClassifier())
   ])
   ```

---

## 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Theory

t-SNE is a non-linear technique that embeds high-dimensional data into low dimensions (typically 2D or 3D) by preserving local neighborhood structure.

**Algorithm:**
1. Compute pairwise similarities in high-dimensional space (Gaussian)
2. Compute pairwise similarities in low-dimensional space (t-distribution)
3. Minimize KL divergence between the two distributions using gradient descent

**Why t-distribution?**
- Heavier tails than Gaussian
- Prevents "crowding problem"
- Allows moderate distances in high-D to become larger in low-D

### When to Use t-SNE

✅ **Good for:**
- Visualizing clusters and patterns
- Exploring high-dimensional data
- Publications and presentations
- Anomaly detection visualization
- Understanding data structure

❌ **Avoid when:**
- Need to embed new data (no transform method)
- Large datasets (> 10,000 samples)
- Need reproducible results
- Interpreting distances between clusters
- Production ML pipelines

### Implementation

```python
from sklearn.manifold import TSNE

# Pre-reduce dimensions if > 50 features
if X.shape[1] > 50:
    pca = PCA(n_components=50)
    X = pca.fit_transform(X)

tsne = TSNE(
    n_components=2,
    perplexity=30,        # Balance local vs global (5-50)
    n_iter=1000,          # At least 1000 iterations
    init='pca',           # Better than random
    random_state=42       # For reproducibility
)

X_tsne = tsne.fit_transform(X)
```

### Hyperparameters

#### Perplexity (Most Important)

Controls balance between local and global structure:

- **5-15**: Very local, tight clusters
- **30-50**: Recommended (default: 30)
- **50-100**: More global structure

**Rule of thumb:** Perplexity should be less than number of samples

```python
# Test different perplexities
for perp in [5, 30, 50, 100]:
    tsne = TSNE(perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X)
    # Visualize and compare
```

#### Iterations

- Minimum: 1000 (default: 1000)
- Increase if: KL divergence hasn't converged
- Check: `tsne.kl_divergence_`

#### Initialization

- **'pca'**: Faster convergence, more stable (recommended)
- **'random'**: More variation, slower

### Best Practices

1. **Pre-process data**
   ```python
   # Scale features
   X_scaled = StandardScaler().fit_transform(X)

   # Pre-reduce with PCA if > 50 features
   if X_scaled.shape[1] > 50:
       X_scaled = PCA(n_components=50).fit_transform(X_scaled)
   ```

2. **Use fixed random_state for reproducibility**
   ```python
   tsne = TSNE(random_state=42)
   ```

3. **Run multiple times with different random states**
   ```python
   for seed in [42, 123, 456]:
       tsne = TSNE(random_state=seed)
       # Compare results
   ```

4. **Don't interpret cluster sizes or distances**
   - Cluster sizes don't reflect actual data density
   - Distances between clusters are meaningless
   - Only local structure is meaningful

5. **Check convergence**
   ```python
   print(f"Final KL divergence: {tsne.kl_divergence_}")
   # Lower is better (< 1.0 is good)
   ```

### Common Issues

❌ **"Crowding" - All points in a ball**
- Solution: Increase perplexity, more iterations

❌ **"Fragmented" - Too many small clusters**
- Solution: Decrease perplexity

❌ **Takes too long**
- Solution: Use PCA pre-processing, reduce sample size, or use UMAP

---

## 3. UMAP (Uniform Manifold Approximation and Projection)

### Theory

UMAP is a modern dimensionality reduction technique based on manifold learning and topological data analysis. It builds a graph representation of data and optimizes a low-dimensional embedding.

**Key Ideas:**
- Data lies on a lower-dimensional manifold
- Constructs fuzzy topological representation
- Optimizes cross-entropy between high-D and low-D representations

**Advantages over t-SNE:**
- Much faster (5-10x)
- Preserves both local AND global structure
- Can transform new data
- More stable results
- Better mathematical foundations

### When to Use UMAP

✅ **Good for:**
- Everything t-SNE is good for
- Large datasets (scales better)
- Production ML pipelines (has transform)
- When you need both local and global structure
- Embedding new data points
- Reproducible results

❌ **Limitations:**
- Requires installation (`pip install umap-learn`)
- More hyperparameters than PCA
- Still non-linear (less interpretable than PCA)

### Implementation

```python
import umap

reducer = umap.UMAP(
    n_neighbors=15,       # Local vs global balance
    min_dist=0.1,         # How tight to pack points
    n_components=2,       # Usually 2 or 3
    metric='euclidean',   # Distance metric
    random_state=42
)

X_umap = reducer.fit_transform(X)

# Can transform new data!
X_new_umap = reducer.transform(X_new)
```

### Hyperparameters

#### n_neighbors (Most Important)

Controls local vs global structure:

- **5-15**: Very local structure, tight clusters
- **15-30**: Balanced (recommended, default: 15)
- **30-100**: More global structure

```python
# Similar to perplexity in t-SNE but different scale
umap.UMAP(n_neighbors=15)  # Good default
```

#### min_dist

Controls how tightly points are packed:

- **0.0**: Very tight clusters, maximize separation
- **0.1**: Tight (recommended for clusters)
- **0.3-0.5**: More spread out, preserve topology
- **0.8-1.0**: Very spread out

```python
# For cluster visualization
umap.UMAP(min_dist=0.1)

# For topology preservation
umap.UMAP(min_dist=0.5)
```

#### metric

Distance metric for high-dimensional space:

- **'euclidean'**: Default, most common
- **'cosine'**: Good for text embeddings
- **'manhattan'**: L1 distance
- **'correlation'**: For gene expression data

```python
# For text data
umap.UMAP(metric='cosine')
```

### Supervised UMAP

UMAP can use labels to guide the embedding:

```python
# Supervised (uses labels)
X_umap = reducer.fit_transform(X, y=y)

# Unsupervised (ignores labels)
X_umap = reducer.fit_transform(X)
```

**When to use supervised:**
- ✅ Better class separation
- ✅ When classification is the goal
- ❌ Can overfit to labels
- ❌ May miss other interesting structure

### Best Practices

1. **Start with defaults**
   ```python
   reducer = umap.UMAP(random_state=42)
   ```

2. **Scale your data**
   ```python
   X_scaled = StandardScaler().fit_transform(X)
   ```

3. **Tune n_neighbors for your dataset size**
   ```python
   # Small datasets (< 1000)
   n_neighbors = 10

   # Medium datasets (1000-10000)
   n_neighbors = 15-30

   # Large datasets (> 10000)
   n_neighbors = 30-50
   ```

4. **Use for preprocessing**
   ```python
   # UMAP as feature extraction
   reducer = umap.UMAP(n_components=10)
   X_reduced = reducer.fit_transform(X_train)

   # Train classifier
   clf.fit(X_reduced, y_train)

   # Transform test data
   X_test_reduced = reducer.transform(X_test)
   ```

5. **Parallel processing for large datasets**
   ```python
   reducer = umap.UMAP(n_jobs=-1)  # Use all CPUs
   ```

---

## Comparison Matrix

| Feature | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| **Type** | Linear | Non-linear | Non-linear |
| **Speed (10K samples)** | < 1s | 30-60s | 5-10s |
| **Preserves** | Global structure | Local structure | Local + Global |
| **Deterministic** | ✅ Yes | ❌ No* | ✅ Yes* |
| **Transform new data** | ✅ Yes | ❌ No | ✅ Yes |
| **Interpretability** | Medium | Low | Low |
| **Hyperparameters** | Few | Medium | Many |
| **Memory** | O(n × p) | O(n²) | O(n × n_neighbors) |
| **Best for** | Preprocessing | Visualization | Both |
| **Max samples** | No limit | ~10K | ~1M+ |

*With fixed random_state

---

## Choosing the Right Method

### Decision Tree

```
Need to visualize data?
├─ Yes
│  ├─ Dataset < 10K samples?
│  │  ├─ Yes → Try t-SNE and UMAP, compare
│  │  └─ No → Use UMAP
│  └─ Want fast results?
│     └─ Yes → Use PCA
│
└─ No (need preprocessing/features)
   ├─ Linear relationships?
   │  └─ Yes → Use PCA
   └─ Non-linear patterns?
      ├─ Need to embed new data?
      │  ├─ Yes → Use UMAP
      │  └─ No → Can use t-SNE or UMAP
      └─ Large dataset (> 100K)?
         └─ Yes → Use UMAP or Incremental PCA
```

### By Use Case

**Data Visualization**
1. PCA (baseline)
2. UMAP (recommended)
3. t-SNE (if < 10K samples)

**Feature Extraction**
1. PCA (linear)
2. UMAP (non-linear, can transform)
3. Kernel PCA (non-linear, no transform)

**Noise Reduction**
1. PCA (fast, effective)
2. Denoising Autoencoder (deep learning)

**Speeding Up Training**
1. PCA (very fast)
2. UMAP (for non-linear)
3. Feature selection (mutual information, etc.)

---

## Common Workflows

### Workflow 1: Exploratory Data Analysis

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Quick PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title(f'PCA: {pca.explained_variance_ratio_.sum():.1%} variance')

# Better visualization with UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y)
plt.title('UMAP: Better cluster separation')
```

### Workflow 2: Preprocessing for ML

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('clf', RandomForestClassifier())
])

# Train (PCA is fitted on training data only)
pipeline.fit(X_train, y_train)

# Predict (PCA transforms test data)
y_pred = pipeline.predict(X_test)
```

### Workflow 3: Dimensionality Analysis

```python
# 1. How much variance can we explain?
pca_full = PCA()
pca_full.fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(cumsum)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

# 2. How does dimensionality affect performance?
for n_comp in [2, 5, 10, 20, 50]:
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_train)

    clf = RandomForestClassifier()
    scores = cross_val_score(clf, X_pca, y_train, cv=5)
    print(f"{n_comp} components: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Tips & Tricks

### 1. Feature Scaling

**Always scale before dimensionality reduction!**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. PCA for Preprocessing

Use PCA before t-SNE/UMAP for large feature spaces:

```python
# Pre-reduce with PCA
if X.shape[1] > 50:
    pca = PCA(n_components=50)
    X = pca.fit_transform(X)

# Then apply t-SNE or UMAP
reducer = umap.UMAP()
X_reduced = reducer.fit_transform(X)
```

### 3. Evaluate Clustering Quality

```python
from sklearn.metrics import silhouette_score

# Higher silhouette = better separation
score = silhouette_score(X_reduced, labels)
print(f"Silhouette Score: {score:.3f}")
```

### 4. Save Fitted Models

```python
import pickle

# Save
with open('reducer.pkl', 'wb') as f:
    pickle.dump(reducer, f)

# Load
with open('reducer.pkl', 'rb') as f:
    reducer = pickle.load(f)

# Transform new data
X_new_reduced = reducer.transform(X_new)
```

---

## Troubleshooting

### PCA Issues

**Problem:** Low explained variance
- **Solution:** May need non-linear method (UMAP, Kernel PCA)

**Problem:** First PC dominates
- **Solution:** Check for outliers, scale features

**Problem:** Can't interpret PCs
- **Solution:** Look at component loadings: `pca.components_`

### t-SNE Issues

**Problem:** Points all bunched together
- **Solution:** Increase perplexity, more iterations

**Problem:** Too fragmented
- **Solution:** Decrease perplexity

**Problem:** Very slow
- **Solution:** Pre-reduce with PCA, use UMAP instead

**Problem:** Different results each run
- **Solution:** Set `random_state`, run multiple times

### UMAP Issues

**Problem:** Too slow
- **Solution:** Reduce n_neighbors, use `n_jobs=-1`

**Problem:** Clusters too spread out
- **Solution:** Decrease min_dist

**Problem:** Can't install umap-learn
- **Solution:** `pip install umap-learn` (not `umap`!)

---

## References & Resources

### Papers
- **PCA**: Pearson (1901) - "On Lines and Planes of Closest Fit"
- **t-SNE**: van der Maaten & Hinton (2008) - "Visualizing Data using t-SNE"
- **UMAP**: McInnes et al. (2018) - "UMAP: Uniform Manifold Approximation and Projection"

### Documentation
- Scikit-learn: https://scikit-learn.org/stable/modules/decomposition.html
- UMAP: https://umap-learn.readthedocs.io/

### Interactive Tools
- TensorFlow Projector: https://projector.tensorflow.org/
- Distill.pub t-SNE: https://distill.pub/2016/misread-tsne/

---

## Summary

**Quick Reference:**

| Task | Recommended Method | Alternative |
|------|-------------------|-------------|
| Visualization (< 10K) | UMAP or t-SNE | PCA baseline |
| Visualization (> 10K) | UMAP | PCA |
| Feature extraction | PCA | UMAP |
| Noise reduction | PCA | Autoencoder |
| Speeding up ML | PCA | Feature selection |
| Exploration | UMAP | t-SNE, PCA |
| Production pipeline | PCA or UMAP | Not t-SNE |

**Default parameters:**
- PCA: `PCA(n_components=0.95)`
- t-SNE: `TSNE(perplexity=30, init='pca', random_state=42)`
- UMAP: `UMAP(n_neighbors=15, min_dist=0.1, random_state=42)`
