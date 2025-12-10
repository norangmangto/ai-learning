# Gaussian Mixture Model (GMM)

Probabilistic clustering using mixture of Gaussian distributions.

## 📋 Overview

**Type:** Probabilistic, soft clustering
**Clusters:** K (specified)
**Algorithm:** Expectation-Maximization (EM)
**Best For:** Soft assignments, uncertainty quantification

## 🎯 Core Idea

Instead of hard clusters, each point has probability of belonging to each cluster.

```
K-Means (hard):              GMM (soft):
Point → Cluster 1            Point → 60% Cluster 1
        100% certain              → 30% Cluster 2
        0% Cluster 2             → 10% Cluster 3

More realistic! Points on cluster boundary have uncertainty.
```

## 📐 Mathematical Foundation

### Mixture Model
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

Where:
- $\pi_k$ = mixture weight (prior probability)
- $\mathcal{N}(x | \mu_k, \Sigma_k)$ = Gaussian with mean $\mu_k$, covariance $\Sigma_k$

### Gaussian Distribution
$$\mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

## 🔄 EM Algorithm

### E-step: Responsibilities
```
For each point, compute probability of being in each cluster:

γ_k(x) = (π_k * N(x | μ_k, Σ_k)) / Σ_j(π_j * N(x | μ_j, Σ_j))

Higher responsibility = more likely point belongs to cluster k
```

### M-step: Update Parameters
```
Update cluster parameters based on responsibilities:

π_k ← (1/N) Σ_i γ_k(x_i)
μ_k ← Σ_i γ_k(x_i) * x_i / Σ_i γ_k(x_i)
Σ_k ← Σ_i γ_k(x_i) * (x_i - μ_k)(x_i - μ_k)^T / Σ_i γ_k(x_i)
```

## 📊 Iteration Visualization

```
Initial: Random clusters

┌─────────┐
│ ●       │ Gaussian 1
│ ●  ●    │ Gaussian 2
│    ●    │ Gaussian 3
└─────────┘

After E-step:
Points have soft assignments to clusters

After M-step:
Gaussian parameters update based on responsibilities

After 10 iterations:
Well-fit Gaussians to data
```

## 🚀 Quick Start

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Data
X = np.random.randn(300, 2)

# Fit GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Soft assignments (responsibilities)
soft_labels = gmm.predict_proba(X)
# Shape: (300, 3), each row sums to 1

# Hard assignments (highest responsibility)
hard_labels = gmm.predict(X)

# Model parameters
means = gmm.means_  # (3, 2)
covariances = gmm.covariances_  # (3, 2, 2)
weights = gmm.weights_  # (3,)

# Likelihood
log_likelihood = gmm.score(X)

# BIC/AIC for model selection
bic = gmm.bic(X)
aic = gmm.aic(X)

# Generate samples
samples = gmm.sample(n_samples=100)
```

## 📊 Covariance Types

```
Different covariance structures:

'full': Σ_k unrestricted
┌────────┐
│  ●●    │ Elliptical, any orientation
│ ●  ●   │
│  ●     │
└────────┘

'tied': Σ_k = Σ (shared covariance)
┌────────┐
│  ●●    │ Same shape/size for all
│ ●  ●   │ clusters
│  ●     │
└────────┘

'diag': Diagonal covariance (no correlation)
┌────────┐
│  ●●    │ Axis-aligned ellipses
│ ●  ●   │
│  ●     │
└────────┘

'spherical': Σ_k = σ_k²I (circles)
┌────────┐
│  ●●    │ Circular clusters
│ ●  ●   │
│  ●     │
└────────┘
```

## 🎯 Choosing Number of Components

### Method 1: BIC/AIC
```python
components_range = range(1, 10)
bic_scores = []
aic_scores = []

for n in components_range:
    gmm = GaussianMixture(n_components=n)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

# Lower is better
optimal_k = components_range[np.argmin(bic_scores)]
```

### Method 2: Silhouette Score
```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k)
    labels = gmm.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

optimal_k = np.argmax(scores) + 2
```

## 💡 GMM vs K-Means

```
Point in cluster boundary:

K-Means: 100% Cluster A, 0% Cluster B
         (hard, unrealistic)

GMM: 55% Cluster A, 45% Cluster B
     (soft, captures uncertainty)

When to use each:
- Hard assignments needed → K-Means
- Uncertainty matters → GMM
- Probability distribution needed → GMM
- Speed critical → K-Means
- Well-separated clusters → K-Means
- Overlapping clusters → GMM
```

## 📈 Applications

| Domain | Use Case |
|--------|----------|
| **Finance** | Portfolio clustering with uncertainty |
| **Biology** | Gene expression soft clusters |
| **Speech** | GMM-HMM for speech recognition |
| **Anomaly** | Likelihood-based outlier detection |
| **Vision** | Soft image segmentation |

## ⚠️ Common Issues

1. **Singularity**
   - Covariance becomes singular (non-invertible)
   - Solution: Add regularization (`reg_covar=1e-6`)

2. **Wrong K**
   - Use BIC/AIC for model selection
   - Solution: Systematically test multiple K

3. **Slow convergence**
   - Many iterations needed
   - Solution: Increase `max_iter` or use `n_init=10`

4. **Local optima**
   - EM can get stuck locally
   - Solution: Try multiple initializations (`n_init=10`)

## 🎓 Learning Outcomes

- [x] Mixture model concept
- [x] EM algorithm (E-step, M-step)
- [x] Soft vs hard clustering
- [x] Covariance matrix types
- [x] Model selection (BIC/AIC)

## 📚 Key Papers

- **Original**: "Maximum Likelihood Estimation" (Dempster et al., 1977)
- **GMM**: "Mixture Models" (McLachlan & Peel, 2000)

## 💪 Advantages

✅ **Probabilistic** - Principled framework with likelihoods
✅ **Soft assignments** - Uncertainty quantification
✅ **Flexible** - Different covariance types
✅ **Model selection** - BIC/AIC for choosing K
✅ **Generative** - Can sample from model

## 🚨 Disadvantages

❌ **Slower** - EM iterations vs K-means
❌ **Singularity issues** - Can fail with high dimensions
❌ **Assumes Gaussians** - Poor for non-Gaussian data
❌ **More parameters** - Covariance matrices to estimate

---

**Last Updated:** December 2024
**Status:** ✅ Complete
