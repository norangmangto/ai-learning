# Sentiment Analysis Models

## Overview

Sentiment analysis determines the emotional tone or opinion expressed in text. This directory contains implementations using traditional ML and modern transformer-based approaches.

## What is Sentiment Analysis?

Sentiment analysis (opinion mining) classifies text into sentiment categories:
- **Binary**: Positive / Negative
- **Multi-class**: Positive / Neutral / Negative
- **Fine-grained**: Very Negative → Very Positive (5 classes)
- **Aspect-based**: Sentiment toward specific aspects

### Common Use Cases
- Social media monitoring
- Customer review analysis
- Brand reputation management
- Market sentiment analysis
- Customer support prioritization
- Product feedback analysis

## Approaches Implemented

### 1. Traditional ML (TF-IDF + Classifier)

**File:** `train_sklearn.py`

**Theory:**
```
Text → TF-IDF Vectorization → Logistic Regression/SVM → Sentiment
```

**TF-IDF (Term Frequency-Inverse Document Frequency):**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
TF(t, d) = (count of term t in document d) / (total terms in d)
IDF(t) = log(N / (number of documents containing t))
```

**When to Use:**
- Small to medium datasets (< 100k samples)
- Quick baseline needed
- Limited computational resources
- Interpretable features important
- Fast inference required

**Advantages:**
- ✅ Fast training and inference
- ✅ Low memory footprint
- ✅ Interpretable (feature weights)
- ✅ Works well with small data
- ✅ No GPU needed

**Limitations:**
- ❌ Ignores word order
- ❌ No context understanding
- ❌ Fixed vocabulary
- ❌ Poor with sarcasm/irony
- ❌ Doesn't handle negation well

**Pipeline:**
1. Text preprocessing (lowercase, remove punctuation)
2. TF-IDF vectorization (max_features=5000-10000)
3. Train classifier (Logistic Regression, SVM, Naive Bayes)
4. Evaluate on test set

### 2. BERT-based Models (PyTorch)

**File:** `train_pytorch.py`

**Models:**
- **BERT**: Bidirectional Encoder Representations from Transformers
- **DistilBERT**: Faster, lighter BERT (40% smaller, 60% faster)
- **RoBERTa**: Robustly Optimized BERT
- **ALBERT**: A Lite BERT (parameter sharing)

**Theory:**
```
Text → Tokenization → BERT Encoder → [CLS] token → Dense → Softmax → Sentiment
```

**How BERT Works:**
1. Tokenize text with WordPiece/BPE
2. Add [CLS] and [SEP] tokens
3. Pass through transformer encoder (12-24 layers)
4. Use [CLS] token embedding for classification
5. Fine-tune entire model or freeze encoder

**When to Use:**
- Large datasets (10k+ samples)
- State-of-the-art performance needed
- Context understanding important
- GPU available
- Fine-tuning possible

**Advantages:**
- ✅ Best accuracy
- ✅ Understands context
- ✅ Handles negation, sarcasm better
- ✅ Pre-trained knowledge
- ✅ Transfer learning

**Limitations:**
- ❌ Requires GPU for reasonable speed
- ❌ Large model size (110M-340M parameters)
- ❌ Slower inference
- ❌ Needs more training data
- ❌ Less interpretable

**Model Variants:**

| Model | Parameters | Speed | Accuracy | When to Use |
|-------|-----------|-------|----------|-------------|
| BERT-base | 110M | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Best accuracy |
| DistilBERT | 66M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Good speed/accuracy |
| RoBERTa | 125M | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Highest accuracy |
| ALBERT | 12M-235M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Memory efficient |

**Fine-tuning Strategies:**
```python
# 1. Full fine-tuning (best accuracy, slowest)
- Unfreeze all layers
- Train entire model

# 2. Freeze encoder, train classifier only (fast)
for param in model.bert.parameters():
    param.requires_grad = False

# 3. Gradual unfreezing (good balance)
- Start with classifier only
- Gradually unfreeze encoder layers
- Fine-tune with lower learning rate
```

### 3. TensorFlow/Keras Implementation

**File:** `train_tensorflow.py`

**Models:**
- TensorFlow Hub BERT models
- Keras preprocessing layers
- Pre-trained embeddings

**When to Use:**
- Production deployment (TF Serving)
- Mobile deployment (TF Lite)
- Distributed training
- Enterprise applications

## Datasets

### Common Sentiment Datasets

1. **IMDb Movie Reviews**
   - 50k reviews (25k train, 25k test)
   - Binary sentiment (positive/negative)
   - Long reviews (average 233 words)

2. **Stanford Sentiment Treebank (SST)**
   - Movie reviews with fine-grained labels
   - 5 classes or binary
   - Phrase-level annotations

3. **Amazon Product Reviews**
   - Millions of reviews
   - 5-star ratings
   - Multiple categories

4. **Twitter Sentiment140**
   - 1.6M tweets
   - Binary sentiment
   - Short text, informal language

5. **Yelp Reviews**
   - Business reviews
   - 5-star ratings
   - Restaurant/service context

## Quick Start

### 1. Traditional ML (Fastest)
```bash
# TF-IDF + Logistic Regression
python train_sklearn.py
```

### 2. BERT Fine-tuning (Best Accuracy)
```bash
# Requires GPU
python train_pytorch.py
```

### 3. TensorFlow
```bash
python train_tensorflow.py
```

## Evaluation Metrics

### Binary Sentiment

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Accuracy: Overall correctness
accuracy = accuracy_score(y_true, y_pred)

# F1-Score: Harmonic mean of precision/recall
f1 = f1_score(y_true, y_pred, average='binary')

# AUC: Area under ROC curve
auc = roc_auc_score(y_true, y_pred_proba)
```

### Multi-class Sentiment

```python
# Weighted F1 (accounts for imbalance)
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# Macro F1 (equal weight to all classes)
f1_macro = f1_score(y_true, y_pred, average='macro')

# Per-class metrics
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

## Best Practices

### 1. Text Preprocessing

**For Traditional ML:**
```python
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Optional: Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]

    # Optional: Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)
```

**For BERT:**
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_for_bert(text):
    # BERT handles lowercase, special tokens automatically
    # Just basic cleaning
    text = re.sub(r'http\S+', '', text)  # Remove URLs

    # Tokenize with BERT tokenizer
    encoded = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return encoded
```

### 2. Handling Imbalanced Data

```python
# Option 1: Class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)

# Option 2: Oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Option 3: Focal Loss (for BERT)
# Focuses on hard examples
```

### 3. Train-Val-Test Split

```python
from sklearn.model_selection import train_test_split

# First split: train+val, test
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Second split: train, val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)
```

### 4. Hyperparameters

**TF-IDF + LogisticRegression:**
```python
# TF-IDF
max_features = 5000-10000
ngram_range = (1, 2)  # unigrams + bigrams
min_df = 2
max_df = 0.95

# Logistic Regression
C = 1.0  # inverse regularization
max_iter = 1000
solver = 'lbfgs'
```

**BERT Fine-tuning:**
```python
# Model
model_name = 'bert-base-uncased'  # or 'distilbert-base-uncased'
num_labels = 2  # binary or 3+ for multi-class

# Training
learning_rate = 2e-5  # typical: 2e-5 to 5e-5
batch_size = 16-32    # depends on GPU memory
epochs = 3-5          # BERT needs few epochs
warmup_steps = 100-500
max_length = 512      # max sequence length

# Optimization
optimizer = AdamW     # BERT optimizer
weight_decay = 0.01   # regularization
```

### 5. Early Stopping & Checkpointing

```python
# PyTorch
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Or manual
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        break
```

## Common Issues and Solutions

### Issue: Poor Performance on Negation
**Example:** "not good" → predicted as positive

**Solutions:**
- ✅ Use BERT (understands context)
- ✅ N-grams with TF-IDF (captures "not good")
- ✅ Custom negation handling in preprocessing
- ✅ Sentiment lexicons with negation rules

### Issue: Sarcasm Detection Fails
**Example:** "Great, another delay!" → positive

**Solutions:**
- ✅ Use RoBERTa or BERT-large (better context)
- ✅ Include punctuation/emoji features
- ✅ Collect sarcasm-labeled data
- ✅ Consider aspect-based sentiment

### Issue: Overfitting (BERT)
**Solutions:**
- ✅ Reduce epochs (3-5 is enough)
- ✅ Increase dropout
- ✅ Add weight decay
- ✅ More training data
- ✅ Data augmentation

### Issue: Slow Inference (BERT)
**Solutions:**
- ✅ Use DistilBERT (60% faster)
- ✅ Quantization (INT8)
- ✅ ONNX Runtime
- ✅ Batch predictions
- ✅ Model distillation

### Issue: Imbalanced Classes
**Solutions:**
- ✅ Class weights
- ✅ Oversampling minority class
- ✅ Focal loss
- ✅ Use F1-score instead of accuracy

## Example Workflows

### Traditional ML Pipeline
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(C=1.0, max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1: {f1_score(y_test, y_pred, average='weighted')}")
```

### BERT Fine-tuning
```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate(test_dataset)
```

## Further Reading

- [BERT Paper: "Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Stanford Sentiment Analysis Course](http://web.stanford.edu/class/cs224n/)
- [TextBlob: Sentiment Analysis Tutorial](https://textblob.readthedocs.io/)

## Next Steps

1. Try **Text Classification** (../text_classification/) for multi-label tasks
2. Explore **Text Summarization** (../text_summarization/) for longer texts
3. Learn **Named Entity Recognition** for information extraction
4. Study **Aspect-Based Sentiment Analysis**
5. Deploy models with FastAPI or TF Serving
