# Regression

## Regression algorithms

* Ordinal regression
* Poisson regression
* Fast forest quantile regression
* Linear, Polynomial, Lasso, Stepwise, Ridge regression
* Bayesian linear regression
* Neural network regression
* Decision forest regression
* Boosted decision tree regression
* KNN (K-nearest neighbors)

## Linear Regression

### Types

* Simple Regression
  * Simple Linear Regression
  * Simple Non-linear Regression
* Multiple Regressions
  * Multiple Linear Regression
  * Multiple Non-linear Regression

### Applications

* Sales forecasting
* Satisfaction analysis
* Price estimation
* Employment income

### Model evaluation

#### Training Accuracy VS Out-of-sample Accuracy

* Training Accuracy
  * High training accuracy isn't necessarily a good thing
  * Result of over-fitting: the model is overly trained to the sataset, which may capture noise and produce a non-generalized model

* Out-of-Sample Accuracy
  * It's important that our models have a high, out-of-smaple accuracy
  * How can we improve out-of-sample accuracy?

#### Train/Test split evaluation approach

Split the entire dataset into 2 datasets, training and testing sets repectively that are mutually exclusive.
The training set is used to train the model and the testing set is used to evaluate the trained model.

#### K-fold cross-validation

Split the entire dataset into several folds that are mutually exclusive.
Use each fold for testing and the rest dataset for training and calculate the accuracy for each training.
Finally, the average of the accuracies are the finaly accuracy of the model.

#### Evaluation Metrics

The Error: measure of how far the data is from the fitted regression line.

* MAE (Mean Absolute Error): Σ|actual - predicted| / n
* MSE (Mean Squared Error): Σ(|actual - predicted|)^2 / n
* RMSE (Root Mean Squared Error): sqrt(MSE) = sqrt(Σ(|actual - predicted|)^2 / n)
* RAE (Relative Absolute Error): Measures the average absolute difference between actual and predicted values relative to the average absolute difference between actual values and their mean. Σ|actual - predicted| / Σ|actual - mean|
* RSE (Relative Squared Error)
* RSS (Residual Sum of Squares): Calculates the sum of the squared differences between actual and predicted values.Σ(actual - predicted)^2

Choosing the metric depends on the model type, data type, and so on.

### Multiple Linear Regression

* Independent variables effectiveness on prediction
  * Does revision time, test anxiety, lecture attendance and gender have any effect on the exam performance of students?

* Predicting impacts of changes
  * How much does blood pressure go up (or down) for every unit increase (or decrease) in the BMI of a patient?

#### An optimization algorithm

* Gradient Descent
* Proper approach if you have a very large dataset

### Q&A

* How to determine whether to use simple or multiple linear regression?
* How many independet variables should you use?
* Should the independent variable be continuous?
* What are the linear relationships between the dependent variable and the independent variables?
  