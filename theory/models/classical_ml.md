# Classical Machine Learning Models

This document covers the fundamental "shallow" learning models that form the basis of ML.

## 1. Linear Regression

### Concept
Linear Regression attempts to model the relationship between two or more variables by fitting a linear equation to observed data. It assumes a linear relationship between the input variables ($X$) and the single output variable ($y$).

$$y = wX + b$$

### Pros & Cons
*   **Pros**: Simple, interpretable, fast to train.
*   **Cons**: Cannot model complex non-linear relationships. Sensitive to outliers.

### Use Cases
*   Predicting house prices based on square footage.
*   Estimating sales based on ad spend.

### Code
*   [Scikit-Learn Implementation](../../models/basics/linear_regression/train_sklearn.py)
*   [PyTorch Implementation](../../models/basics/linear_regression/train_pytorch.py)
*   [TensorFlow Implementation](../../models/basics/linear_regression/train_tensorflow.py)

---

## 2. Logistic Regression

### Concept
Despite its name, this is a **classification** algorithm. It estimates the probability that an instance belongs to a particular class using the logistic function (sigmoid).

$$P(y=1|X) = \frac{1}{1 + e^{-(wX + b)}}$$

### Pros & Cons
*   **Pros**: Probabilistic interpretation, easy to regularize, efficient.
*   **Cons**: Assumes linear decision boundaries.

### Use Cases
*   Spam detection (Spam vs Linear).
*   Credit default prediction.

### Code
*   [Scikit-Learn Implementation](../../models/basics/logistic_regression/train_sklearn.py)

---

## 3. Support Vector Machine (SVM)

### Concept
SVM finds the hyperplane that best divides a dataset into two classes. "Best" is defined as the hyperplane with the largest **margin** between the two classes. It can handle non-linear data using the **Kernel Trick** (mapping data to higher dimensions).

### Pros & Cons
*   **Pros**: Effective in high dimensional spaces. Versatile (Linear, RBF kernels).
*   **Cons**: Not suitable for large datasets (training time is cubic). Sensitive to noise/overlapping classes.

### Use Cases
*   Image classification (historically).
*   Handwriting recognition.

### Code
*   [Scikit-Learn Implementation](../../models/basics/svm/train_sklearn.py)
