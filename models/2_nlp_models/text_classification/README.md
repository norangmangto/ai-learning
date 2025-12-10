# Text Classification

Assign documents to predefined categories or labels.

## 📋 Overview

**Task:** Document → Category
**Examples:** Spam detection, sentiment, topic, intent
**Baseline:** Bag-of-words + logistic regression
**Modern:** Transformers (BERT, RoBERTa)

## 🎯 Approaches

### 1. Traditional ML

```
Text → Tokenize → TF-IDF vectorize → ML classifier
              ↓
        Logistic Regression
        Naive Bayes
        SVM
        Random Forest
```

**Advantages:**
- Fast training
- Interpretable
- Works with small data

**Disadvantages:**
- Limited context
- Poor with long documents
- Requires feature engineering

### 2. Deep Learning

```
Text → Tokenize → Embeddings → RNN/CNN → Dense → Output
              ↓
        LSTM/GRU
        CNN filters
        Attention
```

**Advantages:**
- Better accuracy
- Automatic feature learning
- Handles longer contexts

**Disadvantages:**
- Need more data
- Slower training
- Less interpretable

### 3. Transformers

```
Text → Tokenize → BERT/GPT → [CLS] pooling → Dense → Output
              ↓
        Fine-tune transformer
```

**Advantages:**
- State-of-the-art accuracy
- Transfer learning friendly
- Understands context deeply

**Disadvantages:**
- Computationally expensive
- Large memory requirements
- Requires GPU

## 📊 Model Comparison

| Model | Speed | Accuracy | Data | Interpretability |
|-------|-------|----------|------|-----------------|
| Logistic Reg | ⚡⚡⚡ | 75% | Low | ✓✓✓ |
| Naive Bayes | ⚡⚡⚡ | 77% | Low | ✓✓✓ |
| SVM | ⚡⚡ | 80% | Low | ✓✓ |
| LSTM | ⚡ | 85% | Medium | ✓ |
| CNN | ⚡ | 86% | Medium | ✓ |
| BERT | 🐢 | 92% | High | ✗ |

## 🚀 Quick Start: Traditional ML

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Data
texts = ["I love this movie!", "Terrible film."]
labels = [1, 0]  # 1=positive, 0=negative

# Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression())
])

model.fit(texts, labels)

# Predict
pred = model.predict(["Amazing performance!"])
proba = model.predict_proba(["Amazing performance!"])
```

## 🚀 Quick Start: Deep Learning

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Simple LSTM classifier
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden.squeeze(0))
        return x

# Training loop
model = TextClassifier(vocab_size=5000, embedding_dim=100, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for batch_x, batch_y in train_loader:
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 🚀 Quick Start: Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pretrained
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare data
texts = ["I love this!", "This is bad."]
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits
    predictions = logits.argmax(dim=1)

# Fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 📈 Multi-class vs Multi-label

### Multi-class
```
Each document → One category

Example: Topic classification
"Apple releases new iPhone"
→ Category: Technology (not Politics or Sports)

→ Use: Softmax, CrossEntropyLoss
```

### Multi-label
```
Each document → Multiple categories

Example: Movie genres
"The Matrix"
→ Tags: Sci-Fi, Action, Thriller, Neo-Noir
→ All apply!

→ Use: Sigmoid, BCELoss
```

## 📊 Evaluation Metrics

### Binary Classification
```
Accuracy: (TP + TN) / (TP + TN + FP + FN)

Precision: TP / (TP + FP)
           → Of predicted positive, how many correct?

Recall: TP / (TP + FN)
        → Of actual positive, how many found?

F1: 2 × (Precision × Recall) / (Precision + Recall)
    → Harmonic mean
```

### Multi-class
```
Micro-F1: Calculate globally
          Useful for imbalanced data

Macro-F1: Calculate per class, average
          Treats all classes equally
```

## ⚠️ Common Pitfalls

1. **Class imbalance**
   ```
   Dataset: 95% class A, 5% class B
   Model predicts everything as A
   Accuracy: 95% (terrible!)

   Solution:
   - Use class weights
   - Oversampling/undersampling
   - Evaluate on F1, not accuracy
   ```

2. **Text preprocessing matters**
   ```python
   # Affects results significantly
   - Lowercase
   - Remove punctuation
   - Remove stopwords
   - Stemming/lemmatization
   ```

3. **Data leakage**
   ```
   Don't include metadata as features:
   - Email address → might identify
   - Author name → correlates but unfair
   ```

4. **Hyperparameter tuning**
   ```
   Don't use test set for tuning!
   Use validation set instead
   ```

## 🎯 Decision Guide

```
How to choose model?

Small data (< 1k samples)?
├─ Yes → Logistic Regression or pretrained BERT
└─ No → Deep learning if possible

Limited compute?
├─ Yes → Logistic Regression, Naive Bayes
└─ No → BERT, RoBERTa

Need interpretability?
├─ Yes → Logistic Regression (linear coefficients)
└─ No → Deep learning or transformer

Time budget?
├─ Hours → Logistic Regression
├─ Days → LSTM/CNN
└─ Weeks → BERT fine-tuning
```

## 🎓 Learning Outcomes

- [x] Traditional ML baselines
- [x] Deep learning approaches
- [x] Transformer fine-tuning
- [x] Multi-class vs multi-label
- [x] Evaluation metrics
- [x] Common pitfalls

## 📚 Key Papers

- **TF-IDF baseline**: Foundational NLP
- **Deep learning**: "Learning Phrase Representations" (Mikolov et al., 2013)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)

## 💡 Modern Best Practice

```
1. Start with pretrained transformer
   bert = AutoModelForSequenceClassification.from_pretrained('bert-base')

2. Fine-tune on your data
   2-3 epochs, learning rate 2e-5 to 5e-5

3. Evaluate on test set
   Report F1, precision, recall
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete
