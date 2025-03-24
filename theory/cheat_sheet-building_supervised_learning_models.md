# Cheat Sheet: Building Supervised Learning Models

## Common supervised learning models

<table>
    <!-- header -->
    <tr>
        <th>Process Name</th>
        <th>Brief Description</th>
        <th>Code Syntax</th>
    </tr>
    <!-- One vs One classifier -->
    <tr>
        <td>One vs One classifier (using logistic regression)</td>
        <td>
            <b>Process</b>: This method trains one classifier for each pair of classes.<br/>
            <b>Key hyperparameters</b>:<br/>
            <ul>
                <li><code>estimator</code>: Base classifier (e.g., logistic regression)
            </ul>
            <b>Pros</b>: Can work well for small datasets.<br/>
            <b>Cons</b>: Computationally expensive for large datasets.<br/>
            <b>Common applications</b>: Multiclass classification problems where the number of classes is relatively small.<br/>
        </td>
        <td>

```python
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
model = OneVsOneClassifier(LogisticRegression())
```

</td>
    </tr>
    <!-- One vs All classifier -->
    <tr>
        <td>One vs All classifier (using logistic regression)</td>
        <td>
            <b>Process</b>: Trains one classifier per class, where each classifier distinguishes between one class and the rest.<br/>
            <b>Key hyperparameters</b>:<br/>
            <ul>
                <li><code>estimator</code>: Base classifier (e.g., Logistic Regression)
                <li><code>multi_class</code>: Strategy to handle multiclass classification (`ovr`)
            </ul>
            <b>Pros</b>: Simpler and more scalable than One vs One.<br/>
            <b>Cons</b>: Less accurate for highly imbalanced classes.<br/>
            <b>Common applications</b>: Common in multiclass classification problems such as image classification.<br/>
        </td>
        <td>

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
model = OneVsRestClassifier(LogisticRegression())
```
or
```python
from sklearn.linear_model import LogisticRegression
model_ova = LogisticRegression(multi_class='ovr')
```

</td>
    </tr>
    <!-- Decision tree classifier -->
    <tr>
        <td>Decision tree classifier</td>
        <td>
            <b>Process</b>: A tree-based classifier that splits data into smaller subsets based on feature values.<br/>
            <b>Key hyperparameters</b>:<br/>
            <ul>
                <li><code>max_depth</code>: Maximum depth of the tree
            </ul>
            <b>Pros</b>: Easy to interpret and visualize.<br/>
            <b>Cons</b>: Prone to overfitting if not pruned properly.<br/>
            <b>Common applications</b>: Classification tasks, such as credit risk assessment.<br/>
        </td>
        <td>

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5)
```

</td>
    </tr>
    <!-- Decision tree regressor -->
    <tr>
        <td>Decision tree regressor</td>
        <td>
            <b>Process</b>: Similar to the decision tree classifier, but used for regression tasks to predict continuous values.<br/>
            <b>Key hyperparameters</b>:<br/>
            <ul>
                <li><code>max_depth</code>: Maximum depth of the tree
            </ul>
            <b>Pros</b>: Easy to interpret, handles nonlinear data.<br/>
            <b>Cons</b>: Can overfit and perform poorly on noisy data.<br/>
            <b>Common applications</b>: Regression tasks, such as predicting housing prices.<br/>
        </td>
        <td>

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5)
```

</td>
    </tr>
    <!-- Linear SVM classifier -->
    <tr>
        <td>Linear SVM classifier</td>
        <td>
            <b>Process</b>: A linear classifier that finds the optimal hyperplane separating classes with a maximum margin.<br/>
            <b>Key hyperparameters</b>:<br/>
            <ul>
                <li><code>C</code>: Regularization parameter
                <li><code>kernel</code>: Type of kernel function (<code>linear</code>, <code>poly</code>, <code>rbf</code>, etc.)
                <li><code>gamma</code>: Kernel coefficient (only for <code>rbf</code>, <code>poly</code>, etc.)
            </ul>
            <b>Pros</b>: Effective for high-dimensional spaces.<br/>
            <b>Cons</b>: Not ideal for nonlinear problems without kernel tricks.<br/>
            <b>Common applications</b>: Text classification and image recognition.<br/>
        </td>
        <td>

```python
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0)
```

</td>
    </tr>
    <!-- K-nearest neighbors classifier -->
    <tr>
        <td>K-nearest neighbors classifier</td>
        <td>
            <b>Process</b>: Classifies data based on the majority class of its nearest neighbors.<br/>
            <b>Key hyperparameters</b>:<br/>
            <ul>
                <li><code>n_neighbors</code>: Number of neighbors to use
                <li><code>weights</code>: Weight function used in prediction (<code>uniform</code> or <code>distance</code>)
                <li><code>algorithm</code>: Algorithm used to compute the nearest neighbors (<code>auto</code>, <code>ball_tree</code>, <code>kd_tree</code>, <code>brute</code>)
            </ul>
            <b>Pros</b>: Simple and effective for small datasets.<br/>
            <b>Cons</b>: Computationally expensive as the dataset grows.<br/>
            <b>Common applications</b>: Recommendation systems, image recognition.<br/>
        </td>
        <td>

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
```

</td>
    </tr>
    <!-- Random Forest regressor -->
    <tr>
        <td>Random Forest regressor</td>
        <td>
            <b>Process</b>: An ensemble method using multiple decision trees to improve accuracy and reduce overfitting.<br/>
            <b>Key hyperparameters</b>:<br/>
            <ul>
                <li><code>n_estimators</code>: Number of trees in the forest
                <li><code>max_depth</code>: Maximum depth of each tree
            </ul>
            <b>Pros</b>: Less prone to overfitting than individual decision trees.<br/>
            <b>Cons</b>: Model complexity increases with the number of trees.<br/>
            <b>Common applications</b>: Regression tasks such as predicting sales or stock prices.<br/>
        </td>
        <td>

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=5)
```

</td>
    </tr>
    <!-- XGBoost regressor -->
    <tr>
        <td>XGBoost regressor</td>
        <td>
            <b>Process</b>: A gradient boosting method that builds trees sequentially to correct errors from previous trees.<br/>
            <b>Key hyperparameters</b>:<br/>
            <ul>
                <li><codel>n_estimators</code>: Number of boosting rounds
                <li><codel>learning_rate</code>: Step size to improve accuracy
                <li><codel>max_depth</code>: Maximum depth of each tree
            </ul>
            <b>Pros</b>: High accuracy and works well with large datasets.<br/>
            <b>Cons</b>: Computationally intensive, complex to tune.<br/>
            <b>Common applications</b>: Predictive modeling, especially in Kaggle competitions.<br/>
        </td>
        <td>

```python
import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
```

</td>
    </tr>
</table>

## Associated functions used

<table>
    <!-- header -->
    <tr>
        <th>Method Name</th>
        <th>Brief Description</th>
        <th width="60%">Code Syntax</th>
    </tr>
    <!-- OneHotEncoder -->
    <tr>
        <td>OneHotEncoder</td>
        <td>Transforms categorical features into a one-hot encoded matrix.</td>
        <td>

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(categorical_data)
```

</td>
    </tr>
    <!-- accuracy_score -->
    <tr>
        <td>accuracy_score</td>
        <td>Computes the accuracy of a classifier by comparing predicted and true labels.</td>
        <td>

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
```

</td>
    </tr>
    </tr>
    <!-- LabelEncoder -->
    <tr>
        <td>LabelEncoder</td>
        <td>Encodes labels (target variable) into numeric format.</td>
        <td>

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
```

</td>
    </tr>
    <!-- plot_tree -->
    <tr>
        <td>plot_tree</td>
        <td>Plots a decision tree model for visualization.</td>
        <td>

```python
from sklearn.tree import plot_tree
plot_tree(model, max_depth=3, filled=True)
```

</td>
    </tr>
    <!-- normalize -->
    <tr>
        <td>normalize</td>
        <td>Scales each feature to have zero mean and unit variance (standardization).</td>
        <td>

```python
from sklearn.preprocessing import normalize
normalized_data = normalize(data, norm='l2')
```

</td>
    </tr>
    <!-- compute_sample_weight -->
    <tr>
        <td>compute_sample_weight</td>
        <td>Computes sample weights for imbalanced datasets.</td>
        <td>

```python
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight(class_weight='balanced', y=y)
```

</td>
    </tr>
    <!-- roc_auc_score -->
    <tr>
        <td>roc_auc_score</td>
        <td>Computes the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for binary classification models.</td>
        <td>

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_score)
```

</td>
    </tr>
</table>
