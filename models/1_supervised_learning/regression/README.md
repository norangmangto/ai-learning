# Regression Models

## Overview

Regression models predict continuous numerical values based on input features. This directory contains implementations of various regression algorithms across multiple frameworks.

## What is Regression?

Regression analysis is a supervised learning technique used to model the relationship between:
- **Dependent variable (target)**: The value we want to predict (continuous)
- **Independent variables (features)**: Input variables used to make predictions

### Common Use Cases
- House price prediction
- Stock price forecasting
- Temperature prediction
- Sales forecasting
- Demand estimation

## Models Implemented

### 1. Linear Regression

**Theory:**
Linear regression models the relationship between variables using a linear equation:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```
Where:
- y = predicted value
- β₀ = intercept
- β₁...βₙ = coefficients
- x₁...xₙ = features
- ε = error term

**When to Use:**
- Linear relationships between features and target
- Quick baseline model
- Interpretable predictions needed
- Small to medium datasets

**Advantages:**
- ✅ Simple and interpretable
- ✅ Fast training
- ✅ Low computational cost
- ✅ Works well with linear relationships

**Limitations:**
- ❌ Assumes linear relationships
- ❌ Sensitive to outliers
- ❌ Cannot capture complex patterns
- ❌ Multicollinearity issues

### 2. Polynomial Regression

**Theory:**
Extends linear regression by adding polynomial features:
```
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
```

**When to Use:**
- Non-linear relationships
- Curved patterns in data
- When linear model underfits

**Advantages:**
- ✅ Captures non-linear relationships
- ✅ More flexible than linear regression
- ✅ Still interpretable

**Limitations:**
- ❌ Overfitting risk with high degrees
- ❌ Sensitive to outliers
- ❌ Extrapolation issues

### 3. Ridge Regression (L2 Regularization)

**Theory:**
Adds L2 penalty to prevent overfitting:
```
Loss = MSE + α × Σ(βᵢ²)
```

**When to Use:**
- Many correlated features
- Overfitting concerns
- Feature selection not needed

**Advantages:**
- ✅ Reduces overfitting
- ✅ Handles multicollinearity
- ✅ Keeps all features

**Limitations:**
- ❌ Doesn't eliminate features
- ❌ Requires hyperparameter tuning

### 4. Lasso Regression (L1 Regularization)

**Theory:**
Adds L1 penalty for feature selection:
```
Loss = MSE + α × Σ|βᵢ|
```

**When to Use:**
- Feature selection needed
- Sparse models preferred
- Many irrelevant features

**Advantages:**
- ✅ Automatic feature selection
- ✅ Sparse models
- ✅ Reduces overfitting

**Limitations:**
- ❌ May eliminate important features
- ❌ Unstable with correlated features

## Implementation Files

### Scikit-Learn Implementation
**File:** `train_sklearn.py`

```python
# Key features:
- LinearRegression, Ridge, Lasso
- Easy to use API
- Built-in cross-validation
- StandardScaler for preprocessing
```

**Best for:**
- Quick prototyping
- Classical ML pipelines
- Small to medium datasets
- Production-ready models

### PyTorch Implementation
**File:** `train_pytorch.py`

```python
# Key features:
- Custom neural network approach
- Gradient descent optimization
- GPU acceleration support
- Flexible architecture
```

**Best for:**
- Deep learning pipelines
- GPU acceleration needed
- Custom loss functions
- Integration with deep models

### TensorFlow/Keras Implementation
**File:** `train_tensorflow.py`, `train_tensorflow_v2.py`

```python
# Key features:
- Sequential model API
- Easy integration with TF ecosystem
- Distributed training support
- TensorBoard integration
```

**Best for:**
- Production deployment
- Large-scale training
- Model serving with TF Serving
- Enterprise applications

### JAX Implementation
**File:** `train_jax.py`

```python
# Key features:
- Functional programming approach
- Automatic differentiation
- JIT compilation
- NumPy-like API
```

**Best for:**
- Research projects
- High-performance computing
- Custom optimization algorithms
- Scientific computing

## Quick Start

### 1. Scikit-Learn (Recommended for beginners)
```bash
python train_sklearn.py
```

### 2. PyTorch (GPU acceleration)
```bash
python train_pytorch.py
```

### 3. TensorFlow
```bash
python train_tensorflow_v2.py
```

### 4. JAX (Advanced)
```bash
python train_jax.py
```

## Evaluation Metrics

### Mean Squared Error (MSE)
```python
MSE = (1/n) × Σ(y_true - y_pred)²
```
- Lower is better
- Sensitive to outliers
- Same units as target²

### Root Mean Squared Error (RMSE)
```python
RMSE = √MSE
```
- Lower is better
- Same units as target
- Interpretable

### Mean Absolute Error (MAE)
```python
MAE = (1/n) × Σ|y_true - y_pred|
```
- Lower is better
- Less sensitive to outliers
- Same units as target

### R² Score (Coefficient of Determination)
```python
R² = 1 - (SS_res / SS_tot)
```
- Range: -∞ to 1
- 1 = perfect fit
- 0 = baseline (mean)
- Negative = worse than mean

## Hyperparameters

### Linear Regression
- `fit_intercept`: Whether to calculate intercept (default: True)
- `normalize`: Normalize features (default: False)

### Ridge Regression
- `alpha`: Regularization strength (default: 1.0)
  - Higher values = more regularization
  - Typical range: 0.01 to 100

### Lasso Regression
- `alpha`: Regularization strength (default: 1.0)
- `max_iter`: Maximum iterations (default: 1000)

### Neural Network (PyTorch/TensorFlow)
- `learning_rate`: Step size for optimization (default: 0.001)
- `batch_size`: Number of samples per update (default: 32)
- `epochs`: Number of training iterations (default: 100)
- `hidden_layers`: Network architecture

## Best Practices

### 1. Data Preprocessing
```python
# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Remove outliers (optional)
from scipy import stats
z_scores = np.abs(stats.zscore(X))
X_clean = X[(z_scores < 3).all(axis=1)]
```

### 2. Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Cross-Validation
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5,
                        scoring='neg_mean_squared_error')
```

### 4. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

## Common Issues and Solutions

### Issue: Poor R² Score
**Solutions:**
- Add polynomial features
- Try different algorithms
- Feature engineering
- Collect more data
- Check for data leakage

### Issue: Overfitting (Training >> Test performance)
**Solutions:**
- Use Ridge/Lasso regularization
- Reduce model complexity
- Get more training data
- Cross-validation
- Early stopping

### Issue: Underfitting (Both low)
**Solutions:**
- Add more features
- Polynomial features
- Try more complex models
- Reduce regularization
- Check feature scaling

### Issue: Slow Training
**Solutions:**
- Use GPU (PyTorch/TensorFlow)
- Reduce data size
- Optimize batch size
- Use simpler model
- JAX with JIT compilation

## Comparison: When to Use Which?

| Framework | Best For | Speed | Ease of Use |
|-----------|----------|-------|-------------|
| Scikit-Learn | Quick prototypes, classical ML | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| PyTorch | Research, custom models | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| TensorFlow | Production, serving | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| JAX | High-performance computing | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## Further Reading

- [Scikit-Learn Linear Models Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [PyTorch Linear Regression Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [TensorFlow Regression Guide](https://www.tensorflow.org/tutorials/keras/regression)
- [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)

## Next Steps

After mastering regression:
1. Try **Classification** models (../classification/)
2. Explore **Ensemble Methods** (../ensemble_methods/)
3. Learn **Feature Engineering** techniques
4. Study **Time Series** regression
5. Dive into **Deep Learning** approaches (../../4_sequence_models/)
