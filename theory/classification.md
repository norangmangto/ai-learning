# Classification

* A supervised learning approach
* Categorizing some unknown items into a discret set of categories or "classes"

## Classification algorithms

* Decision Trees (ID3, C4.5, C5.0)
* Naïve Bayes
* Linear Discriminant Analysis
* K-Nearest Neighbor (KNN)
* Logistic Regression
* Neural Networks
* Support Vector Machines (SVM)

## K-Nearest Neighbor (KNN)

* A method for classifying cases based on their similarity to other cases
* Cases that are near each other are said to be "neighbors"
* Based on similar cases with same class labels are near each other

## Evaluation Metrics

### Jaccard index

### F1-score

Uses a `Confusion matrix`

* Precision = TP / (TP + FP)
* Recall = TP / (TP + FN)
* F1-score = 2x(precision x recall) / (precision + recall)

(TP: True Positive, FP: False Positive, FN: False Negative)

### Log loss

Performace of a classifier where the predicted output is a probability value between 0 and 1.

## Decision Trees

### Algorithm

1. Choose an attribute from the dataset
2. Calculate the significance of attribute in splitting of data
3. Split data based on the value of the best attribute
4. Go to step 1

### Choose the best attribute

More **Predictiveness** = Less **Impurity** = Lower **Entropy**

#### What is Entropy?

: Measure of randomness or uncertainty

`Entropy = -p(A)log2(p(A))-p(B)log2(p(B))`

> The lower the Entropy, the less uniform the distribution, the purer the node.

> The tree with the higher `Information Gain` after splitting

#### What is Information Gain?

**`Information gain`** is the information that can increase the leel of certainty after splitting.

`Information Gain = (Entropy before split) - (weighted entrypy after split)`

