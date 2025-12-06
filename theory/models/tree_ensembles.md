# Tree Ensembles

Tree-based models are non-linear and handle tabular data exceptionally well.

## 1. Random Forest

### Concept
Random Forest is an **ensemble** method that trains many Decision Trees.
*   **Bagging (Bootstrap Aggregating)**: Each tree is trained on a random subset of the data.
*   **Random Subspace**: Each split considers only a random subset of features.
The final prediction is the average (regression) or majority vote (classification) of all trees.

### Pros & Cons
*   **Pros**: Robust to overfitting, handles missing data well, provides feature importance.
*   **Cons**: Slow prediction time (must run 100+ trees), Model file size can be large.

### Use Cases
*   Customer churn prediction.
*   Medical diagnosis.

### Code
*   [Scikit-Learn Implementation](../../models/basics/random_forest/train_sklearn.py)

---

## 2. XGBoost (Gradient Boosting)

### Concept
**eXtreme Gradient Boosting** is an optimized implementation of Gradient Boosted Decision Trees (GBDT). Unlike Random Forest (which builds independent trees), Boosting builds trees **sequentially**. Each new tree attempts to correct the errors (residuals) made by the previous trees.

### Pros & Cons
*   **Pros**: Often achieves state-of-the-art results on structured data. Highly optimized/fast training.
*   **Cons**: Harder to tune (many hyperparameters). Can overfit if not regulated.

### Use Cases
*   Kaggle competitions (Tabular data).
*   Risk assessment.
*   Recommendation ranking.

### Code
*   [XGBoost Implementation](../../models/basics/xgboost/train.py)
