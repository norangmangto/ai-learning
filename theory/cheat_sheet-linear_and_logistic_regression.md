# Cheat Sheet: Linear and Logistic Regression

## Comparing different regression types

<table>
    <!-- header -->
    <tr>
        <th>Model Name</th>
        <th>Description</th>
        <th>Code Syntax</th>
    </tr>
    <!-- Simple Linear Regression -->
    <tr>
        <td>Simple linear regression</td>
        <td>
            <ul>
                <li><b>Purpose</b>: To predict a dependent variable based on one independent variable.</li>
                <li><b>Pros</b>: Easy to implement, interpret, and efficient for small datasets.</li>
                <li><b>Cons</b>: Not suitable for complex relationships; prone to underfitting.</li>
                <li><b>Modeling equation</b>: y = b0 + b1x</li>
            </ul>
        </td>
        <td>

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression() model.fit(X, y)
```

</td>
    </tr>
    <!-- Polynomial Regression -->
    <tr>
        <td>Polynomial regression</td>
        <td>
            <ul>
                <li><b>Purpose</b>: To capture nonlinear relationships between variables.</li>
                <li><b>Pros</b>: Better at fitting nonlinear data compared to linear regression.</li>
                <li><b>Cons</b>: Prone to overfitting with high-degree polynomials.</li>
                <li><b>Modeling equation</b>: y = b0 + b1x + b2x<sup>2</sup> + ...</li>
            </ul>
        </td>
        <td>

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
```

</td>
    </tr>
    <!-- Multiple Linear Regression -->
    <tr>
        <td>Multiple linear regression</td>
        <td>
            <ul>
                <li><b>Purpose</b>: To predict a dependent variable based on multiple independent variables.</li>
                <li><b>Pros</b>: Accounts for multiple factors influencing the outcome.</li>
                <li><b>Cons</b>: Assumes a linear relationship between predictors and target.</li>
                <li><b>Modeling equation</b>: y = b0 + b1x<sup>1</sup> + b2x<sup>2</sup> + ...</li>
            </ul>
        </td>
        <td>

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

</td>
    </tr>
    <!-- Logistic Regression -->
    <tr>
        <td>Logistic regression</td>
        <td>
            <ul>
                <li><b>Purpose</b>: To predict probabilities of categorical outcomes.</li>
                <li><b>Pros</b>: Efficient for binary classification problems.</li>
                <li><b>Modeling equation</b>: log(p/(1-p)) = b0 + b1x<sup>1</sup> + ...</li>
            </ul>
        </td>
        <td>

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

</td>
    </tr>
</table>

## Associated functions commonly used

<table>
    <!-- header -->
    <tr>
        <th>Function/Method Name</th>
        <th>Brief Description</th>
        <th width="60%">Code Syntax</th>
    </tr>
    <!-- train_test_split -->
    <tr>
        <td>train_test_split</td>
        <td>
            Splits the dataset into training and testing subsets to evaluate the model's performance.
        </td>
        <td>

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

</td>
    </tr>
    <!-- StandardScaler -->
    <tr>
        <td>StandardScaler</td>
        <td>
            Standardizes features by removing the mean and scaling to unit variance.
        </td>
        <td>

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

</td>
    </tr>
    <!-- log_loss-->
    <tr>
        <td>log_loss</td>
        <td>
            Calculates the logarithmic loss, a performance metric for classification models.
        </td>
        <td>

```python
from sklearn.metrics import log_loss
loss = log_loss(y_true, y_pred_proba)
```

</td>
    </tr>
    <!-- mean_absolute_error -->
    <tr>
        <td>mean_absolute_error</td>
        <td>
            Calculates the mean absolute error between actual and predicted values.
        </td>
        <td>

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

</td>
    </tr>
    <!-- mean_squared_error -->
    <tr>
        <td>mean_squared_error</td>
        <td>
            Computes the mean squared error between actual and predicted values.
        </td>
        <td>

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

</td>
    </tr>
    <!-- root_mean_squared_error -->
    <tr>
        <td>root_mean_squared_error</td>
        <td>
            Calculates the root mean squared error (RMSE), a commonly used metric for regression tasks.
        </td>
        <td>

```python
from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

</td>
    </tr>
    <!-- r2_score -->
    <tr>
    <td>r2_score</td>
    <td>
        Computes the R-squared value, indicating how well the model explains the variability of the target variable.
    </td>
    <td>

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

</td>
    </tr>
</table>
