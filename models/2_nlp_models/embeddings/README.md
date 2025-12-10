# Word and Document Embeddings

Comprehensive guide to learning dense vector representations for words and documents in NLP.

## Table of Contents
- [Overview](#overview)
- [Word Embeddings](#word-embeddings)
- [Document Embeddings](#document-embeddings)
- [Comparison Matrix](#comparison-matrix)
- [Implementation Guide](#implementation-guide)
- [Best Practices](#best-practices)

---

## Overview

**Embeddings** transform discrete text (words, sentences, documents) into continuous vector representations that capture semantic meaning. These dense vectors enable machines to understand relationships between textual elements.

### Why Embeddings?

Traditional one-hot encoding:
- High dimensionality (vocabulary size)
- No semantic information
- Sparse representation

Embeddings provide:
- Low dimensionality (50-300 dimensions)
- Semantic relationships (similar words → similar vectors)
- Dense representation
- Transfer learning capability

---

## Word Embeddings

Word embeddings map individual words to fixed-length vectors.

### Word2Vec

**Architecture:**
- **CBOW (Continuous Bag of Words):** Predicts target word from context
- **Skip-gram:** Predicts context words from target word

**Key Concepts:**
```
Context window: Number of surrounding words considered
Vector size: Dimensionality of embeddings (typically 100-300)
Negative sampling: Efficient training technique
Subsampling: Reduces impact of frequent words
```

**Training Objective:**
- CBOW: Maximize P(target | context)
- Skip-gram: Maximize P(context | target)

**When to Use:**
- **CBOW:** Faster training, better for frequent words, smaller datasets
- **Skip-gram:** Better for rare words, captures semantic relationships better

**Example Relationships:**
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
big - bigger + small ≈ smaller
```

**Hyperparameters:**
| Parameter | Typical Range | Impact |
|-----------|--------------|--------|
| Vector size | 50-300 | Quality vs memory trade-off |
| Window size | 5-10 | Larger = more topical, smaller = more functional |
| Min count | 2-5 | Vocabulary size control |
| Epochs | 5-20 | Convergence quality |

### GloVe (Global Vectors)

**Architecture:**
Global matrix factorization + local context windows

**Key Innovation:**
Combines advantages of:
- Global matrix factorization (LSA)
- Local context windows (Word2Vec)

**Training Process:**
1. Build word co-occurrence matrix
2. Weight co-occurrences by distance
3. Factorize using weighted least squares

**Objective Function:**
```
J = Σ f(X_ij) * (w_i^T * w_j + b_i + b_j - log(X_ij))^2

where:
- X_ij = co-occurrence count
- f(x) = weighting function
- w_i, w_j = word vectors
- b_i, b_j = bias terms
```

**Weighting Function:**
```python
f(x) = (x / x_max)^α  if x < x_max
f(x) = 1              otherwise

Typical: x_max = 100, α = 0.75
```

**When to Use:**
- Better on word analogy tasks
- Deterministic training (reproducible)
- Good for larger corpora
- When global statistics are important

**GloVe vs Word2Vec:**
| Aspect | GloVe | Word2Vec |
|--------|-------|----------|
| Approach | Global co-occurrence | Local context |
| Training | Matrix factorization | Neural network |
| Speed | Faster for large corpora | Faster for small corpora |
| Results | Better analogies | Better rare words |
| Deterministic | Yes | No |

---

## Document Embeddings

Document embeddings create fixed-length vectors for variable-length documents.

### Doc2Vec (Paragraph Vectors)

**Architectures:**

**1. PV-DM (Distributed Memory):**
- Similar to Word2Vec CBOW
- Adds document vector to context
- Predicts target word from context + document
- Captures word order information

```
Input: [doc_vector, context_words] → Predict: target_word
```

**2. PV-DBOW (Distributed Bag of Words):**
- Similar to Word2Vec Skip-gram
- Predicts words from document vector only
- Ignores word order
- Faster training

```
Input: doc_vector → Predict: random_words_from_doc
```

**Training Process:**
```python
# PV-DM
for document in corpus:
    for word in document:
        context = surrounding_words(word)
        predict(word | [doc_vector, context])

# PV-DBOW
for document in corpus:
    sample_words = random_sample(document)
    predict(sample_words | doc_vector)
```

**Key Features:**
- Learns fixed-length vectors for variable-length documents
- Can infer vectors for new documents
- Captures document-level semantic meaning
- Outperforms bag-of-words approaches

**Hyperparameters:**
| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Vector size | 100-400 | Larger for complex documents |
| Window | 5-10 | Context size |
| DM/DBOW | PV-DM | PV-DM usually better |
| Min count | 2-5 | Vocabulary filtering |
| Epochs | 20-50 | More needed than Word2Vec |

### Alternative Approaches

**1. Averaged Word2Vec:**
```python
doc_vector = mean([word2vec(word) for word in document])
```
- Simple and fast
- Loses word order information
- Baseline approach

**2. Weighted Averaging (TF-IDF):**
```python
doc_vector = Σ (tfidf(word) * word2vec(word)) / Σ tfidf(word)
```
- Weights important words higher
- Better than simple averaging
- Still loses word order

**3. Sentence-BERT (Modern):**
- Fine-tuned BERT for sentence/document embeddings
- State-of-the-art performance
- Requires more compute

---

## Comparison Matrix

### Word Embeddings Comparison

| Method | Training Speed | Memory | Rare Words | Analogies | Deterministic |
|--------|---------------|--------|------------|-----------|---------------|
| Word2Vec CBOW | Fast | Low | Poor | Good | No |
| Word2Vec Skip-gram | Medium | Low | Good | Excellent | No |
| GloVe | Fast* | Medium | Good | Excellent | Yes |

*Faster for very large corpora after co-occurrence matrix built

### Document Embeddings Comparison

| Method | Quality | Speed | Inference | Word Order |
|--------|---------|-------|-----------|------------|
| Averaged Word2Vec | Baseline | Very Fast | Easy | Lost |
| TF-IDF Weighted | Good | Fast | Easy | Lost |
| Doc2Vec PV-DM | Excellent | Medium | Infer needed | Preserved |
| Doc2Vec PV-DBOW | Very Good | Fast | Infer needed | Lost |

---

## Implementation Guide

### 1. Word2Vec Training

```python
from gensim.models import Word2Vec

# Prepare corpus
corpus = [['word1', 'word2', ...], ['sentence2', ...], ...]

# Train Skip-gram
model = Word2Vec(
    sentences=corpus,
    vector_size=100,      # Embedding dimension
    window=5,             # Context window
    min_count=2,          # Ignore rare words
    sg=1,                 # 1=Skip-gram, 0=CBOW
    workers=4,            # Parallel threads
    epochs=20
)

# Get word vector
vector = model.wv['computer']

# Find similar words
similar = model.wv.most_similar('computer', topn=5)

# Word analogy
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
```

### 2. GloVe Training

```python
# Build co-occurrence matrix
cooccurrence = {}
for sentence in corpus:
    for i, word in enumerate(sentence):
        for j in range(max(0, i-window), min(len(sentence), i+window+1)):
            if i != j:
                cooccurrence[(word, sentence[j])] += 1 / abs(i-j)

# Train GloVe
# (Simplified - see implementation for details)
for epoch in range(epochs):
    for (word_i, word_j), count in cooccurrence.items():
        # Update vectors using gradient descent
        loss = (dot(w_i, w_j) + b_i + b_j - log(count))^2
        # Apply gradients...
```

### 3. Doc2Vec Training

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Prepare tagged documents
tagged_docs = [
    TaggedDocument(words=['word1', 'word2', ...], tags=['doc_0']),
    TaggedDocument(words=['word3', 'word4', ...], tags=['doc_1']),
    ...
]

# Train PV-DM
model = Doc2Vec(
    documents=tagged_docs,
    vector_size=100,
    window=5,
    min_count=2,
    dm=1,                 # 1=PV-DM, 0=PV-DBOW
    epochs=50,
    workers=4
)

# Get document vector
doc_vector = model.dv['doc_0']

# Infer vector for new document
new_vector = model.infer_vector(['new', 'document', 'words'])

# Find similar documents
similar = model.dv.most_similar('doc_0', topn=5)
```

---

## Best Practices

### Data Preprocessing

```python
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # 3. Tokenize
    tokens = text.split()

    # 4. Remove stopwords (optional - can hurt embeddings)
    # tokens = [t for t in tokens if t not in stopwords]

    # 5. Lemmatization (optional)
    # tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens
```

### Hyperparameter Tuning

**Vector Size:**
- Small datasets: 50-100
- Medium datasets: 100-200
- Large datasets: 200-300
- Very large: 300-500

**Window Size:**
```
Small window (2-3):
  + Captures syntactic relationships
  + Words with similar functions
  - Limited context

Large window (10-15):
  + Captures topical relationships
  + Broader semantic context
  - May mix different meanings
```

**Training Epochs:**
- Word2Vec: 5-20 epochs
- GloVe: 15-50 epochs
- Doc2Vec: 20-100 epochs (needs more)

### Evaluation Strategies

**Intrinsic Evaluation:**
```python
# 1. Word Similarity
# Compare with human similarity ratings
# Spearman correlation on WordSim353, SimLex-999

# 2. Word Analogies
# Test relationships: king - man + woman = ?
# Accuracy on Google analogy dataset

# 3. Nearest Neighbors
# Check if similar words are close
similarity = model.wv.similarity('cat', 'dog')
```

**Extrinsic Evaluation:**
```python
# Use embeddings in downstream tasks:
# - Text classification
# - Named Entity Recognition (NER)
# - Sentiment analysis
# - Machine translation

# Measure improvement in task performance
```

### Common Pitfalls

**1. Not Enough Data:**
- Word2Vec needs millions of tokens for good results
- Solution: Use pre-trained embeddings (Word2Vec, GloVe, FastText)

**2. Rare Words:**
- May have poor embeddings
- Solutions:
  - Use Skip-gram over CBOW
  - Character n-grams (FastText)
  - Subword tokenization

**3. Out-of-Vocabulary (OOV):**
- New words not in training
- Solutions:
  - Use character-based models (FastText)
  - Average similar words
  - Use contextual embeddings (BERT)

**4. Multiple Word Meanings:**
- Same word, different contexts
- Solutions:
  - Multiple embeddings per word
  - Contextual embeddings (ELMo, BERT)
  - Sense embeddings

### Optimization Tips

```python
# 1. Use negative sampling (Word2Vec)
model = Word2Vec(..., negative=5, ns_exponent=0.75)

# 2. Subsample frequent words
model = Word2Vec(..., sample=1e-3)

# 3. Use hierarchical softmax for small vocab
model = Word2Vec(..., hs=1)

# 4. Parallel training
model = Word2Vec(..., workers=8)

# 5. Load pre-trained embeddings
from gensim.models import KeyedVectors
vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors.bin', binary=True)
```

---

## Use Cases by Task

### Text Classification
```
Best: Doc2Vec PV-DM, averaged Word2Vec + TF-IDF weights
Why: Captures document-level semantics
```

### Information Retrieval
```
Best: Doc2Vec, Sentence-BERT
Why: Semantic similarity between queries and documents
```

### Named Entity Recognition (NER)
```
Best: Word2Vec Skip-gram, contextual embeddings
Why: Captures entity context and relationships
```

### Machine Translation
```
Best: Skip-gram with large window, cross-lingual embeddings
Why: Captures semantic equivalence across languages
```

### Sentiment Analysis
```
Best: Doc2Vec PV-DM, fine-tuned embeddings
Why: Captures sentiment polarity and document context
```

### Recommendation Systems
```
Best: Doc2Vec, item2vec
Why: Learns user/item representations for similarity
```

---

## Advanced Topics

### Transfer Learning with Embeddings

```python
# 1. Load pre-trained embeddings
pretrained = KeyedVectors.load_word2vec_format('embeddings.bin')

# 2. Create embedding matrix for your vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in word_to_id.items():
    if word in pretrained:
        embedding_matrix[idx] = pretrained[word]
    else:
        # Initialize OOV words randomly
        embedding_matrix[idx] = np.random.randn(embedding_dim) * 0.01

# 3. Use in neural network
embedding_layer = nn.Embedding.from_pretrained(
    torch.FloatTensor(embedding_matrix),
    freeze=False  # Allow fine-tuning
)
```

### Cross-Lingual Embeddings

Map embeddings from different languages to shared space:
```python
# Learn transformation matrix
# en_vector @ W ≈ fr_vector
W = learn_mapping(en_embeddings, fr_embeddings, word_pairs)

# Translate
french_vector = english_vector @ W
```

### Contextualized Embeddings (Modern Approach)

```
Static embeddings (Word2Vec, GloVe):
  - Same vector regardless of context
  - "bank" always has same embedding

Contextual embeddings (ELMo, BERT):
  - Different vectors based on context
  - "bank" (financial) vs "bank" (river) different embeddings
```

---

## Resources

### Pre-trained Embeddings
- **Word2Vec:** Google News (300d, 3M words)
- **GloVe:** Common Crawl, Wikipedia (50-300d)
- **FastText:** 157 languages, includes OOV handling

### Datasets for Evaluation
- WordSim353: Word similarity
- SimLex-999: Semantic similarity
- Google Analogy: Word relationships
- SemEval: Various NLP tasks

### Tools & Libraries
- **Gensim:** Word2Vec, Doc2Vec, FastText
- **spaCy:** Pre-trained embeddings integrated
- **Hugging Face:** Modern transformers and embeddings

---

## Quick Reference

### Decision Tree

```
Need embeddings for...

├─ Individual words?
│  ├─ Small corpus (< 10M tokens)?
│  │  └─ Use pre-trained embeddings (GloVe, FastText)
│  └─ Large corpus?
│     ├─ Fast training needed? → Word2Vec CBOW
│     ├─ Best quality? → Word2Vec Skip-gram
│     └─ Reproducible? → GloVe
│
└─ Whole documents?
   ├─ Quick baseline? → Averaged Word2Vec
   ├─ Better quality? → TF-IDF weighted Word2Vec
   └─ Best quality? → Doc2Vec PV-DM

Need to handle...
├─ Rare words? → Skip-gram or FastText
├─ Out-of-vocabulary? → FastText (character n-grams)
├─ Multiple meanings? → Contextual embeddings (BERT)
└─ Multiple languages? → Cross-lingual embeddings
```

### Performance Tips

| Operation | Speed | Quality | When to Use |
|-----------|-------|---------|-------------|
| CBOW | ⚡⚡⚡ | ⭐⭐⭐ | Frequent words, speed critical |
| Skip-gram | ⚡⚡ | ⭐⭐⭐⭐ | Rare words, best quality |
| GloVe | ⚡⚡⚡ | ⭐⭐⭐⭐ | Large corpus, reproducibility |
| Doc2Vec DM | ⚡⚡ | ⭐⭐⭐⭐⭐ | Document quality critical |
| Doc2Vec DBOW | ⚡⚡⚡ | ⭐⭐⭐⭐ | Document speed important |

---

## Implementations

This directory contains three complete implementations:

1. **`word_embeddings/train_word2vec.py`**: Word2Vec with CBOW and Skip-gram
2. **`word_embeddings/train_glove.py`**: GloVe from scratch with co-occurrence matrix
3. **`document_embeddings/train_doc2vec.py`**: Doc2Vec with PV-DM and PV-DBOW

Each includes:
- ✅ Complete training pipeline
- ✅ Evaluation on similarity and analogy tasks
- ✅ Visualization with t-SNE
- ✅ Hyperparameter comparison
- ✅ Best practices and optimization

---

**Happy Embedding! 🚀**
