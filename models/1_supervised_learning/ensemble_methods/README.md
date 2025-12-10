# Ensemble Methods

## Overview

Ensemble methods combine multiple models to create a stronger predictor than any single model. This directory contains implementations of powerful tree-based ensemble algorithms.

## What are Ensemble Methods?

Ensemble learning combines predictions from multiple models to improve:
- **Accuracy**: Often outperforms single models
- **Robustness**: Less sensitive to noise and outliers
- **Generalization**: Better performance on unseen data

### Ensemble Strategies

1. **Bagging** (Bootstrap Aggregating)
   - Train models on random subsets
   - Average predictions
   - Example: Random Forest

2. **Boosting**
   - Train models sequentially
   - Each model corrects previous errors
   - Examples: XGBoost, Gradient Boosting

3. **Stacking**
   - Train meta-model on predictions
   - Combines different model types

## Models Implemented

### 1. Random Forest

**Theory:**
Ensemble of decision trees trained on random subsets of data and features:
```
Final Prediction = Average/Majority Vote of all trees
Tree 1, Tree 2, ..., Tree N → Aggregation → Prediction
```

**How it Works:**
1. Bootstrap sampling: Create N datasets by sampling with replacement
2. Random feature selection: At each split, consider random subset of features
3. Build N decision trees independently
4. Aggregate predictions (average for regression, vote for classification)

**When to Use:**
- Default choice for tabular data
- Handles mixed feature types well
- Need feature importances
- Robust predictions required
- Interpretability less critical

**Advantages:**
- ✅ Excellent out-of-the-box performance
- ✅ Handles non-linear relationships
- ✅ Resistant to overfitting
- ✅ Provides feature importance
- ✅ Handles missing values
- ✅ No feature scaling needed
- ✅ Parallel training (fast)

**Limitations:**
- ❌ Black box (less interpretable than single tree)
- ❌ Large model size (many trees)
- ❌ Slower inference than single tree
- ❌ Can struggle with high cardinality features
- ❌ Memory intensive

**Hyperparameters:**
```python
n_estimators: 100-500         # Number of trees
max_depth: 10-50              # Maximum tree depth (None = unlimited)
min_samples_split: 2-10       # Minimum samples to split node
min_samples_leaf: 1-5         # Minimum samples in leaf
max_features: 'sqrt', 'log2'  # Features per split
bootstrap: True               # Use bootstrap sampling
n_jobs: -1                    # Parallel jobs (-1 = all cores)
```

**Feature Importance:**
Random Forest provides built-in feature importance based on:
- Mean decrease in impurity (Gini importance)
- Permutation importance (more reliable)

### 2. XGBoost (Extreme Gradient Boosting)

**Theory:**
Advanced implementation of gradient boosting with regularization:
```
F(x) = f₀(x) + η·f₁(x) + η·f₂(x) + ... + η·fₙ(x)
Each fᵢ corrects errors of F(x)ᵢ₋₁
```

**How it Works:**
1. Start with initial prediction (e.g., mean)
2. Calculate residuals (errors)
3. Train tree to predict residuals
4. Add tree to ensemble with learning rate η
5. Repeat steps 2-4 for N iterations
6. Final prediction = sum of all trees

**When to Use:**
- Kaggle competitions / best performance needed
- Structured/tabular data
- Classification or regression
- Feature importance needed
- GPU acceleration available

**Advantages:**
- ✅ State-of-the-art performance on tabular data
- ✅ Handles missing values automatically
- ✅ Built-in regularization (prevents overfitting)
- ✅ GPU acceleration
- ✅ Feature importance
- ✅ Early stopping support
- ✅ Handles imbalanced data well
- ✅ Cross-validation built-in

**Limitations:**
- ❌ Requires careful hyperparameter tuning
- ❌ Sensitive to outliers
- ❌ Can overfit with too many iterations
- ❌ Slower training than Random Forest
- ❌ Sequential (harder to parallelize than RF)

**Key Hyperparameters:**
```python
# Tree parameters
max_depth: 3-10               # Typical 6
learning_rate (eta): 0.01-0.3 # Typical 0.1
n_estimators: 100-1000        # More with lower learning rate
min_child_weight: 1-10        # Minimum sum of weights in leaf
gamma: 0-5                    # Minimum loss reduction to split
subsample: 0.5-1.0            # Row sampling ratio
colsample_bytree: 0.5-1.0     # Column sampling ratio

# Regularization
reg_alpha: 0-1                # L1 regularization
reg_lambda: 0-1               # L2 regularization

# Training
objective:                    # 'reg:squarederror', 'binary:logistic', 'multi:softmax'
eval_metric:                  # 'rmse', 'logloss', 'auc', 'error'
early_stopping_rounds: 10-50  # Stop if no improvement
```

**XGBoost Tips:**
- Start with defaults, then tune
- Lower learning rate → more estimators
- Use cross-validation for tuning
- Enable GPU for large datasets: `tree_method='gpu_hist'`
- Monitor training with eval_set

### 3. Gradient Boosting (GBM)

**Theory:**
Original gradient boosting implementation (sklearn):
```
Similar to XGBoost but:
- No built-in regularization
- No GPU support
- Simpler implementation
```

**When to Use:**
- Simple boosting needed
- No XGBoost available
- Quick prototyping

**Advantages:**
- ✅ Good performance
- ✅ Built into scikit-learn
- ✅ Easy to use

**Limitations:**
- ❌ Slower than XGBoost
- ❌ Less features than XGBoost
- ❌ No GPU support

### 4. LightGBM (Light Gradient Boosting Machine)

**Theory:**
Microsoft's fast implementation with leaf-wise tree growth:
```
Leaf-wise growth (LightGBM) vs Level-wise (XGBoost)
- Faster training
- Better accuracy with proper tuning
- Risk of overfitting if not careful
```

**When to Use:**
- Very large datasets (millions of rows)
- Speed is critical
- Categorical features present

**Advantages:**
- ✅ Extremely fast training
- ✅ Memory efficient
- ✅ Handles large datasets
- ✅ Native categorical feature support
- ✅ GPU and parallel learning

**Limitations:**
- ❌ Can overfit on small datasets
- ❌ Sensitive to hyperparameters
- ❌ Less stable than XGBoost on small data

## Implementation Files

### Scikit-Learn Implementation
**File:** `train_sklearn.py`

**Models:**
- RandomForestClassifier / RandomForestRegressor
- GradientBoostingClassifier / GradientBoostingRegressor

```python
# Key features:
- Simple API
- Built-in cross-validation
- Feature importances
- Parallel training
```

**Best for:**
- Quick prototyping
- Simple pipelines
- Medium datasets

### XGBoost Native Implementation
**File:** `train.py` or `train_xgboost.py`

```python
# Key features:
- Native XGBoost API
- GPU acceleration
- Advanced features
- Cross-validation
- Early stopping
```

**Best for:**
- Best performance
- Large datasets
- GPU available
- Production deployment

### PyTorch Implementation
**File:** `train_pytorch.py`

```python
# Key features:
- Neural network ensembles
- Custom ensemble architectures
- Deep learning integration
```

**Best for:**
- Research
- Hybrid models (trees + neural nets)
- Custom implementations

## Quick Start

### 1. Random Forest (Fastest to train)
```bash
python train_sklearn.py
```

### 2. XGBoost (Best performance)
```bash
python train.py
```

### 3. PyTorch Ensemble
```bash
python train_pytorch.py
```

## Evaluation & Comparison

### Performance Comparison

| Model | Training Speed | Inference Speed | Accuracy | Memory |
|-------|---------------|-----------------|----------|--------|
| Random Forest | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| XGBoost | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| LightGBM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GBM (sklearn) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### When to Choose Which?

**Random Forest:**
- ✅ First try on new dataset
- ✅ Quick baseline
- ✅ Parallel training important
- ✅ Interpretability via feature importance

**XGBoost:**
- ✅ Need best accuracy
- ✅ Kaggle competitions
- ✅ Production deployment
- ✅ GPU available

**LightGBM:**
- ✅ Very large datasets
- ✅ Speed critical
- ✅ Many categorical features
- ✅ Memory constrained

## Best Practices

### 1. Data Preparation
```python
# Minimal preprocessing needed!
# No feature scaling required
# But handle missing values if needed

import pandas as pd
import numpy as np

# Option 1: Fill missing values
df.fillna(df.median(), inplace=True)

# Option 2: Use -999 or -1 (trees handle this well)
df.fillna(-999, inplace=True)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])
```

### 2. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Hyperparameter Tuning

**Random Forest:**
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, n_jobs=-1)
search.fit(X_train, y_train)
```

**XGBoost:**
```python
import xgboost as xgb

# Start simple
params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

model = xgb.XGBClassifier(**params)

# Use early stopping
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train,
          early_stopping_rounds=10,
          eval_set=eval_set,
          verbose=True)

# Then tune with Optuna or GridSearch
```

### 4. Feature Importance
```python
# Random Forest
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# XGBoost
xgb.plot_importance(model)

# Permutation importance (more reliable)
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(model, X_test, y_test)
```

### 5. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# Random Forest
scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')

# XGBoost with CV
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
cv_results = xgb.cv(
    params, dtrain, num_boost_round=100,
    nfold=5, metrics='auc', early_stopping_rounds=10
)
```

## Common Issues and Solutions

### Issue: Overfitting
**Solutions:**
- ✅ Reduce max_depth
- ✅ Increase min_samples_leaf
- ✅ Reduce n_estimators (XGBoost)
- ✅ Increase learning_rate (XGBoost)
- ✅ Add regularization (XGBoost: reg_alpha, reg_lambda)
- ✅ Use cross-validation

### Issue: Slow Training
**Solutions:**
- ✅ Use LightGBM instead
- ✅ Enable GPU (XGBoost/LightGBM)
- ✅ Reduce n_estimators
- ✅ Parallelize (Random Forest: n_jobs=-1)
- ✅ Subsample data

### Issue: Poor Performance
**Solutions:**
- ✅ Feature engineering
- ✅ Hyperparameter tuning
- ✅ Increase n_estimators
- ✅ Try different model (RF ↔ XGBoost)
- ✅ Check for data leakage
- ✅ Handle imbalanced data

### Issue: Memory Issues
**Solutions:**
- ✅ Use LightGBM (more memory efficient)
- ✅ Reduce max_depth
- ✅ Reduce n_estimators
- ✅ Batch processing
- ✅ Feature selection

## Example Workflow

```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load data
X, y = load_data()

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train XGBoost
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False
)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train,
          early_stopping_rounds=10,
          eval_set=eval_set,
          verbose=False)

# 4. Predict and evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# 5. Feature importance
xgb.plot_importance(model, max_num_features=10)
```

## Further Reading

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Random Forest Paper (Breiman, 2001)](https://link.springer.com/article/10.1023/A:1010933404324)
- [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)

## Next Steps

1. Master feature engineering for tree-based models
2. Learn stacking and blending techniques
3. Explore AutoML (auto-sklearn, FLAML)
4. Try deep learning ensembles (../../4_sequence_models/)
5. Kaggle competitions for practice
