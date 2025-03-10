# Logistic Regression

`Logistic regression` is a **classification** algorithm for categorical variables.
It's used for **binary classification**, where the goal is to predict one of two possible outcomes.

## Logistic regresssion applications

* Predicting the probability of a person having a heart attack
* Predicting the mortality in injured patients
* Predicting a customer's propensity to purchase a product or halt a subscription
* Predicting the probability of failure of a given process or product
* Predicting the liklihood of a homeowner defaulting on a mortgage

## When is logistic regression suitable?

* If your data is binary
  * 0/1, YES/NO, True/False
* If you need probabilistic results
* When you need a linear decision boundary
* If you need to understand the impact of a feature

## Logistic Regression vs Linear Regression

### Sigmoid function

$$S(x)=\frac{1}{1+e^{-x}}$$
or
$$\sigma(x)=\frac{1}{1+e^{-x}}$$

It maps the linear combination of input features to a probilbity.

* P(Y=1|X)
* P(Y=2|X) = 1 - P(Y=1|X)

### Cost function (Loss function)

Cost is used to modify weight.

$$Cost(ŷ,y) = \frac{1}{2}(\sigma(\theta^TX)-y)^2$$

#### Jaccard distance

`J(θ) = sum of Cost(ŷ,y) for a dataset`

## The training process

1. Initialize θ
2. Calculate ŷ = 𝜎(θ^T*X)
3. Compare the output of ŷ with actual output y, and record it as error
4. Change the θ to reduce the cost
5. Go back to step 2

## Minimizing the cost function of the model

* How to find the best parameters for out model?
  * minimize the cost function
* How to minimize the cost function?
  * Using Gradient Descent
* What is gradient descent?
  * A techinique to use the derivative of a cost function to change the parameter values, in order to minimize the cost

### Using gradient descent to minimize the cost

#### Gradient Descent and Learning Rate

$$New θ = old θ - \eta \nabla J$$

* Gradient Descent: taking steps in the current direction of the slope
* Learning Rate: the length of the step when we