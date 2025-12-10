# Word Embeddings (Word2Vec, GloVe, FastText)

Dense vector representations of words capturing semantic meaning.

## 📋 Overview

**Type:** Unsupervised representation learning
**Dimensions:** Typically 100-300
**Output:** Word → Vector
**Best For:** NLP preprocessing, semantic similarity

## 🎯 Core Idea

Convert words to vectors where similar words are close together.

```
Vector space:
         ┌─────────────────┐
         │  king           │
         │    - man        │
         │    + woman  =   queen  ✓
         │                 │
         │  Paris - France │
         │    + Germany = Berlin  ✓
         │                 │
         │  Good - Bad     │
         │    + Worse  = Terrible ✓
         └─────────────────┘

Semantic relationships encoded as vector operations!
```

## 🏗️ Word2Vec: Skip-gram Model

### Concept
```
Training: Predict context words from target word

Input: "The quick brown fox jumps"
       word="quick"

Predict: ["The", "brown"] (window=1)

Network: Input word → Hidden (embedding) → Output (context)
Result: Learn embeddings that predict context well
```

### Architecture
```
Word index i
     │
     ↓ (one-hot or embedding lookup)
  Embedding layer (d dimensions)
     │
     ↓ (hidden layer)
  Hidden layer (shared for all positions)
     │
     ↓ (linear)
  Output softmax (vocabulary size)
     │
     ↓
Predict context word j
```

## 📐 Word2Vec Mathematics

### Skip-gram Objective
Maximize: $$\sum_{t=1}^{T} \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} | w_t)$$

Where:
- $w_t$ = target word at position t
- $w_{t+j}$ = context word
- m = window size

### Softmax Probability
$$P(w_j | w_i) = \frac{\exp(v_j^T v_i)}{\sum_{k=1}^{V} \exp(v_k^T v_i)}$$

Where $v_i$ is embedding of word i.

## 🎨 Different Embedding Models

### Word2Vec
```
Skip-gram: Word → Predict context
CBOW: Context → Predict word

Advantages:
✓ Fast training
✓ Well-understood
✓ Produces good embeddings

Disadvantages:
✗ One embedding per word (polysemy issue)
✗ Unknown words → special token
```

### GloVe (Global Vectors)
```
Combines:
- Global statistics (like LSA)
- Local context windows (like Word2Vec)

Advantages:
✓ Better on small datasets
✓ Captures global structure
✓ Fast

Disadvantages:
✗ Still one vector per word
✗ Requires preprocessing for vocabulary
```

### FastText
```
Words → Character n-grams → Embeddings

Advantages:
✓ Handles unknown words (compose from n-grams)
✓ Better for morphologically rich languages
✓ Useful for rare words

Disadvantages:
✗ Slower training
✗ Larger model size
✗ Still not contextual
```

## 🚀 Quick Start

### Word2Vec (Gensim)
```python
from gensim.models import Word2Vec

# Data
sentences = [
    ['the', 'quick', 'brown', 'fox'],
    ['a', 'lazy', 'dog'],
    ['the', 'brown', 'dog']
]

# Train
model = Word2Vec(
    sentences,
    vector_size=100,  # Embedding dimension
    window=5,         # Context window
    min_count=1,      # Ignore words appearing < min_count times
    sg=1              # 1=Skip-gram, 0=CBOW
)

# Embeddings
dog_vector = model.wv['dog']  # (100,)

# Similarity
similarity = model.wv.similarity('dog', 'cat')  # ~0.8

# Most similar
similar_words = model.wv.most_similar('dog', topn=5)
# [('cat', 0.82), ('puppy', 0.79), ...]

# Analogies
result = model.wv.most_similar(positive=['king', 'woman'],
                               negative=['man'], topn=1)
# Should find 'queen'
```

### GloVe
```python
from glove import Corpus, Glove

# Build corpus
corpus = Corpus()
corpus.fit(texts, window=10)

# Train GloVe
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4)

# Embeddings
word_vector = glove.word_vectors[glove.dictionary['dog']]
```

### FastText
```python
from gensim.models import FastText

# Train
model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1
)

# Embeddings
# Even unknown words get vectors (from n-grams)
unknown_vector = model.wv['unknownword123']
```

## 📊 Comparing Embeddings

| Aspect | Word2Vec | GloVe | FastText |
|--------|----------|-------|----------|
| **Type** | Predictive | Count-based | Hybrid |
| **Speed** | Fast | Medium | Slow |
| **Unknown words** | Special token | OOV | N-gram compose ✓ |
| **Morphology** | Poor | Medium | Good ✓ |
| **Quality** | Good | Good | Good |
| **Small datasets** | Fair | Good ✓ | Good ✓ |

## 🎯 Applications

| Task | Best Model |
|------|-----------|
| **General NLP** | Word2Vec (standard) |
| **Morphological** | FastText (inflections) |
| **Small data** | GloVe |
| **Similarity** | All work well |
| **Analogy** | Word2Vec |

## ⚠️ Important Limitations

```
Static embeddings: One vector per word

Problem: Polysemy (multiple meanings)
Example: "bank" (financial institution vs riverbank)
Solution: Contextual embeddings (BERT, GPT)

Problem: No context
Example: Same embedding for "good" always
         Ignores sentiment context
Solution: Contextual embeddings

These methods became less common with transformers
But still useful for:
- Quick baselines
- Feature engineering
- Lightweight models
```

## 🎓 Learning Outcomes

- [x] Skip-gram and CBOW training
- [x] Embedding visualization
- [x] Similarity and analogy tasks
- [x] Different embedding types
- [x] Word vs contextual embeddings

## 📚 Key Papers

- **Word2Vec**: "Efficient Estimation of Word Representations" (Mikolov et al., 2013)
- **GloVe**: "GloVe: Global Vectors for Word Representation" (Pennington et al., 2014)
- **FastText**: "Enriching Word Vectors with Subword Information" (Bojanowski et al., 2017)

## 💪 Advantages

✅ **Fast training** - Minutes on CPU
✅ **Well-understood** - Decades of research
✅ **Interpretable** - Vector operations show relationships
✅ **Lightweight** - Small memory footprint
✅ **Widely available** - Pretrained models everywhere

## 🚨 Disadvantages

❌ **Not contextual** - Same vector regardless of usage
❌ **Polysemy** - Can't distinguish multiple meanings
❌ **Static** - Fixed for all tasks
❌ **Vocabulary** - Needs coverage of words
❌ **Limited info** - Doesn't capture fine-grained semantics

## 💡 Modern Perspective

```
These methods (Word2Vec, GloVe, FastText) are foundational
but largely superseded by contextual embeddings:

✓ Historical importance: Very high
✓ Modern use: Lower (but still used for features)
✓ Learning value: Essential for understanding NLP

Progression:
Word2Vec (2013) → ELMo (2018) → BERT/GPT (2018+)
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete
