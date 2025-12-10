# Classification Models

## Overview

Classification models predict categorical labels (classes) for input data. This directory contains implementations of various classification algorithms for binary and multi-class problems.

## What is Classification?

Classification is a supervised learning task where the goal is to predict discrete class labels:
- **Binary Classification**: Two classes (e.g., spam/not spam, fraud/legitimate)
- **Multi-class Classification**: More than two classes (e.g., digit recognition 0-9)
- **Multi-label Classification**: Multiple labels per sample

### Common Use Cases
- Email spam detection
- Disease diagnosis
- Image recognition
- Sentiment classification
- Credit risk assessment
- Customer churn prediction

## Models Implemented

### 1. Logistic Regression

**Theory:**
Despite the name, it's a classification algorithm using the sigmoid function:
```
P(y=1|x) = 1 / (1 + e^(-z))
where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**When to Use:**
- Binary classification
- Probability estimates needed
- Interpretable model required
- Baseline model
- Linearly separable data

**Advantages:**
- ✅ Simple and interpretable
- ✅ Provides probability scores
- ✅ Fast training and prediction
- ✅ Works well with linear boundaries
- ✅ Regularization built-in (L1/L2)

**Limitations:**
- ❌ Assumes linear decision boundaries
- ❌ Struggles with complex patterns
- ❌ Sensitive to outliers
- ❌ Requires feature scaling

**Hyperparameters:**
- `C`: Inverse regularization strength (default: 1.0)
- `penalty`: 'l1', 'l2', 'elasticnet' (default: 'l2')
- `solver`: 'liblinear', 'lbfgs', 'saga' (default: 'lbfgs')
- `max_iter`: Maximum iterations (default: 100)

### 2. Multi-Layer Perceptron (MLP)

**Theory:**
Feed-forward neural network with multiple hidden layers:
```
Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output
Each layer: h = activation(Wx + b)
```

**When to Use:**
- Non-linear decision boundaries
- Complex pattern recognition
- Large datasets
- Feature interactions important

**Advantages:**
- ✅ Learns non-linear patterns
- ✅ Automatic feature learning
- ✅ Flexible architecture
- ✅ Scales to large datasets
- ✅ GPU acceleration available

**Limitations:**
- ❌ Requires more data
- ❌ Longer training time
- ❌ Hyperparameter sensitive
- ❌ Black box (less interpretable)
- ❌ Risk of overfitting

**Architecture Options:**
- **Input Layer**: Size = number of features
- **Hidden Layers**: 1-3 layers typical
  - Small data: 1 layer, 32-128 neurons
  - Medium data: 2 layers, 64-256 neurons
  - Large data: 3+ layers, 128-512 neurons
- **Output Layer**:
  - Binary: 1 neuron, sigmoid activation
  - Multi-class: n neurons, softmax activation

**Activation Functions:**
- `ReLU`: Default choice, fast, avoids vanishing gradients
- `LeakyReLU`: Prevents dead neurons
- `Tanh`: Range [-1, 1], centered around 0
- `Sigmoid`: Range [0, 1], binary classification

**Hyperparameters:**
- `hidden_layer_sizes`: (100,) default, e.g., (128, 64)
- `activation`: 'relu', 'tanh', 'logistic' (default: 'relu')
- `learning_rate_init`: 0.001 default
- `batch_size`: 32, 64, 128, 256
- `epochs`: 50-500 depending on data
- `dropout`: 0.2-0.5 for regularization

### 3. Deep Neural Network (DNN)

**Theory:**
Extended MLP with deeper architectures (4+ layers):
```
Input → Conv/Dense → BatchNorm → Activation → Dropout → ... → Output
```

**When to Use:**
- Very large datasets (10k+ samples)
- Complex hierarchical features
- State-of-the-art performance needed
- GPU resources available

**Advantages:**
- ✅ Best performance on large data
- ✅ Learns hierarchical features
- ✅ Transfer learning possible
- ✅ Handles high-dimensional data

**Limitations:**
- ❌ Requires large datasets
- ❌ Computationally expensive
- ❌ Many hyperparameters
- ❌ Overfitting risk

**Advanced Techniques:**
- **Batch Normalization**: Stabilizes training
- **Dropout**: Prevents overfitting (0.2-0.5)
- **Early Stopping**: Monitors validation loss
- **Learning Rate Scheduling**: Decay over time
- **Data Augmentation**: Increases training data

## Implementation Files

### Scikit-Learn Implementation
**Files:** `train_sklearn.py`

**Models Included:**
- Logistic Regression
- MLPClassifier (Multi-Layer Perceptron)

```python
# Key features:
- Simple API
- Built-in preprocessing
- Cross-validation support
- Grid search for hyperparameters
```

**Best for:**
- Quick prototyping
- Small to medium datasets (< 100k samples)
- Classical ML pipelines
- Interpretable models

### PyTorch Implementation
**Files:** `train_pytorch.py`, `train_pytorch_v2.py`

```python
# Key features:
- Custom neural network architectures
- GPU acceleration
- Flexible loss functions
- Advanced optimizers (Adam, AdamW, SGD)
- Learning rate schedulers
- TensorBoard integration
```

**Best for:**
- Research and experimentation
- Custom architectures
- Large datasets with GPU
- Deep learning pipelines

### TensorFlow/Keras Implementation
**Files:** `train_tensorflow.py`, `train_tensorflow_v2.py`

```python
# Key features:
- Sequential and Functional API
- Pre-built layers and models
- Easy deployment (TF Serving)
- Distributed training
- TensorBoard visualization
```

**Best for:**
- Production deployment
- Model serving
- Enterprise applications
- Mobile deployment (TF Lite)

### JAX Implementation
**File:** `train_jax.py`

```python
# Key features:
- Functional programming
- JIT compilation for speed
- Automatic differentiation
- NumPy-compatible API
```

**Best for:**
- High-performance computing
- Research projects
- Custom optimization

## Quick Start

### 1. Logistic Regression (Fastest)
```bash
# Scikit-Learn - best for quick results
python train_sklearn.py
```

### 2. MLP/DNN (More powerful)
```bash
# PyTorch - flexible and popular
python train_pytorch.py

# PyTorch v2 - advanced features
python train_pytorch_v2.py

# TensorFlow - production-ready
python train_tensorflow_v2.py
```

### 3. JAX (High-performance)
```bash
python train_jax.py
```

## Evaluation Metrics

### Accuracy
```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Simple, intuitive
- ⚠️ Misleading with imbalanced data

### Precision
```python
Precision = TP / (TP + FP)
```
- Of predicted positives, how many are correct?
- Use when false positives are costly

### Recall (Sensitivity)
```python
Recall = TP / (TP + FN)
```
- Of actual positives, how many did we find?
- Use when false negatives are costly

### F1-Score
```python
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- Good for imbalanced datasets

### ROC-AUC
```python
AUC = Area Under ROC Curve
ROC = True Positive Rate vs False Positive Rate
```
- Range: 0.5 (random) to 1.0 (perfect)
- Threshold-independent
- Good for binary classification

### Confusion Matrix
```
              Predicted
           Positive  Negative
Actual
Positive      TP        FN
Negative      FP        TN
```

## Best Practices

### 1. Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Scale features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Handle imbalanced data
from sklearn.utils import resample
# Oversample minority class or undersample majority
```

### 2. Train-Test-Validation Split
```python
from sklearn.model_selection import train_test_split

# First split: train + val, test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: train, val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)
```

### 3. Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified K-Fold maintains class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
```

### 4. Handling Imbalanced Data
```python
# Option 1: Class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced',
                                    classes=np.unique(y),
                                    y=y)

# Option 2: SMOTE (Synthetic Minority Oversampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Option 3: Adjust decision threshold
# Instead of 0.5, use optimal threshold from ROC curve
```

### 5. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# MLP
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)
```

## Common Issues and Solutions

### Issue: Low Accuracy
**Diagnosis:**
- Check data quality
- Verify labels are correct
- Look at confusion matrix

**Solutions:**
- ✅ Feature engineering
- ✅ Try different algorithms
- ✅ More training data
- ✅ Hyperparameter tuning
- ✅ Ensemble methods

### Issue: Overfitting (Training >> Test)
**Solutions:**
- ✅ Regularization (L1/L2, dropout)
- ✅ Reduce model complexity
- ✅ More training data
- ✅ Early stopping
- ✅ Data augmentation
- ✅ Cross-validation

### Issue: Underfitting (Both low)
**Solutions:**
- ✅ Increase model complexity
- ✅ More features
- ✅ Reduce regularization
- ✅ Train longer
- ✅ Better feature engineering

### Issue: Imbalanced Classes
**Solutions:**
- ✅ Use class weights
- ✅ SMOTE oversampling
- ✅ Adjust decision threshold
- ✅ Use appropriate metrics (F1, AUC)
- ✅ Ensemble with balanced sampling

### Issue: Slow Training
**Solutions:**
- ✅ Use GPU (PyTorch/TensorFlow)
- ✅ Batch size optimization
- ✅ Reduce model size
- ✅ Feature selection
- ✅ Data subsampling

## Framework Comparison

| Framework | Logistic Reg | MLP | DNN | GPU | Ease | Speed |
|-----------|-------------|-----|-----|-----|------|-------|
| Scikit-Learn | ✅ | ✅ | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| PyTorch | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| TensorFlow | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| JAX | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Example Workflows

### Binary Classification
```python
# 1. Load and preprocess
X, y = load_data()
X_scaled = StandardScaler().fit_transform(X)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y
)

# 3. Train model
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")
print(f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")
```

### Multi-class Classification
```python
# Use one-vs-rest or softmax
model = LogisticRegression(multi_class='ovr')  # or 'multinomial'
# or
model = MLPClassifier(hidden_layer_sizes=(128, 64))

# Metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Further Reading

- [Scikit-Learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [PyTorch Neural Network Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Pattern Recognition and Machine Learning (Bishop)](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

## Next Steps

1. Explore **Ensemble Methods** (../ensemble_methods/) for better performance
2. Learn **Support Vector Machines** for non-linear boundaries
3. Try **Convolutional Neural Networks** (../../3_computer_vision/) for image classification
4. Study **Recurrent Networks** (../../4_sequence_models/) for sequential data
5. Master **Natural Language Processing** (../../2_nlp_models/) for text classification
