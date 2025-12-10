# Sentence Embeddings

Dense vector representations of entire sentences capturing semantic meaning.

## 📋 Overview

**Task:** Sentence → Vector
**Dimensions:** 384-1024 typical
**Use:** Semantic similarity, clustering, retrieval
**Modern models:** Sentence-BERT, SimCSE, E5

## 🎯 Evolution

```
Word Embeddings (Word2Vec):
  "bank" → [0.2, -0.1, 0.5, ...]
  Single fixed vector regardless of context

Contextual (BERT):
  "bank" in "river bank" → [different embedding]
  "bank" in "bank account" → [different embedding]
  Context-dependent!

Sentence Embeddings:
  "I love this movie" → [0.1, 0.2, 0.3, ...]
  Full sentence represented as one vector
  Captures semantic meaning of entire sentence
```

## 📐 Creating from Word/Contextual Embeddings

### Mean Pooling
```
Sentence: "I love this movie"
Tokens: ["I", "love", "this", "movie"]

BERT embeddings:
[I]:      [0.1, 0.2, 0.3, ...]  (768-dim)
[love]:   [0.4, 0.1, 0.2, ...]
[this]:   [0.2, 0.3, 0.1, ...]
[movie]:  [0.5, 0.2, 0.4, ...]

Mean pooling:
         [(0.1+0.4+0.2+0.5)/4, (0.2+0.1+0.3+0.2)/4, ...]
Result:  [0.3, 0.2, 0.25, ...]  (sentence embedding)

Simple but works reasonably!
```

### CLS Token Pooling
```
BERT format:
[CLS] I love this movie [SEP]

First token ([CLS]) is pre-trained to aggregate info
Use [CLS] embedding directly as sentence embedding
```

### Attention Pooling
```
Weight tokens by attention scores
Important tokens (high attention) → more weight
Common tokens (low attention) → less weight

Result: More meaningful aggregation
```

## 🚀 Quick Start: Sentence-BERT

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "I love this movie",
    "This film is great",
    "The weather is nice",
    "I enjoy sunny days"
]

embeddings = model.encode(sentences)
# Shape: (4, 384)  # 4 sentences, 384-dim vectors

# Semantic similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(embeddings)
# Shows which sentences are similar

print(f"Similarity (0 vs 1): {similarity[0, 1]:.3f}")  # 0.85 (similar)
print(f"Similarity (0 vs 2): {similarity[0, 2]:.3f}")  # 0.12 (dissimilar)
```

## 📊 Popular Sentence Embedding Models

### Sentence-BERT (SBERT)
```
Based on BERT with contrastive training
Models:
- all-MiniLM-L6-v2: 22M params, 384-dim, very fast
- all-mpnet-base-v2: 110M params, 768-dim, best quality
- all-roberta-large-v1: 335M params, 1024-dim, best if compute ok

Advantages:
✓ Fast inference
✓ Pre-trained on many tasks
✓ Excellent quality

Disadvantages:
✗ Fixed, not fine-tuned to your domain
```

### SimCSE
```
Contrastive learning approach
Positive: Same sentence with different dropout masks
Negative: Other sentences in batch

Advantages:
✓ Simple, effective training
✓ Interpretable method

Disadvantages:
✗ Requires large batch sizes
```

### E5 (Embedding by Bidirectional Encoder Representations from Transformers)
```
Recent approach
Advantages:
✓ Very strong performance
✓ Multilingual support
✓ Open-sourced

Models:
- e5-small: 33M params
- e5-base: 109M params
- e5-large: 333M params
```

## 📈 Applications

```
1. Semantic Search
   Query: "What's a good restaurant?"
   Documents sorted by embedding similarity

2. Duplicate Detection
   Embeddings: If similarity > threshold → duplicates

3. Clustering
   K-means on embeddings
   Clusters semantically similar documents

4. Recommendation
   User interests → embeddings
   Similar document embeddings → recommendations

5. Paraphrase Mining
   Similar embeddings → paraphrases

6. Intent Classification
   User query embeddings vs intent embeddings
   Closest intent = user intent
```

## 🔍 Semantic Similarity Search

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus
documents = [
    "The cat sat on the mat",
    "Dogs are loyal pets",
    "I enjoy reading books",
    "Cats and dogs are animals"
]

# Queries
queries = [
    "Tell me about cats",
    "What about pets?"
]

# Encode
corpus_embeddings = model.encode(documents)
query_embeddings = model.encode(queries)

# Search for each query
from sklearn.metrics.pairwise import cosine_similarity

for i, query in enumerate(queries):
    similarities = cosine_similarity([query_embeddings[i]], corpus_embeddings)[0]
    # Sort and get top-k
    top_k_idx = np.argsort(similarities)[::-1][:2]
    print(f"Query: {query}")
    for idx in top_k_idx:
        print(f"  → {documents[idx]} (similarity: {similarities[idx]:.3f})")
```

## 📊 Sentence Embedding Quality

```
Evaluation on SBERT:

Task                 Accuracy/Correlation
─────────────────────────────────────────
STS12 (semantic textual similarity)  75%
STS13                                 83%
STS14                                 80%
Paraphrase detection                  87%
Clustering                            72%
Semantic search                       92%

Note: Different models have different strengths
all-mpnet-base-v2: Best overall quality
all-MiniLM-L6-v2: Fastest, still good quality
```

## 💡 Understanding Semantic Similarity

```
Cosine similarity = angle between vectors

Identical direction (parallel):
  cos(θ) = 1.0 → Identical meaning

Similar direction:
  cos(θ) = 0.8 → Similar meaning

Perpendicular:
  cos(θ) = 0.0 → Unrelated

Opposite direction:
  cos(θ) = -1.0 → Opposite meaning
```

## ⚠️ Common Pitfalls

1. **Using wrong similarity metric**
   ```python
   # Right: Cosine similarity (accounts for direction)
   from sklearn.metrics.pairwise import cosine_similarity

   # Wrong: Euclidean distance (doesn't work well with embeddings)
   from scipy.spatial.distance import euclidean
   ```

2. **Not normalizing embeddings**
   ```python
   # Some models output normalized, some don't
   # Always normalize for fair comparison
   from sklearn.preprocessing import normalize
   embeddings = normalize(embeddings)
   ```

3. **Selecting wrong model for task**
   ```
   - Fast baseline: all-MiniLM-L6-v2
   - Best quality: all-mpnet-base-v2
   - Multilingual: mMiniLMv2-L12-H384-uncased
   - Domain-specific: Train on your data
   ```

4. **Not considering domain shift**
   ```
   Models trained on general text
   May not work well for:
   - Technical/scientific papers
   - Medical text
   - Code/programming
   - Other specialized domains

   Solution: Fine-tune on your domain
   ```

## 🎯 When to Use

```
Sentence embeddings good for:
✓ Similarity search
✓ Clustering documents
✓ Finding duplicates
✓ Recommendations

Sentence embeddings not ideal for:
✗ Specific entailment (need more context)
✗ Question-answering (need more structure)
✗ Long document representation (limit context)
```

## 🎓 Learning Outcomes

- [x] Word vs sentence embeddings
- [x] Pooling strategies
- [x] Popular models (SBERT, SimCSE, E5)
- [x] Semantic similarity
- [x] Common applications

## 📚 Key Papers

- **Sentence-BERT**: "Sentence-BERT: Sentence Embeddings" (Reimers & Gurevych, 2019)
- **SimCSE**: "Simple Contrastive Learning" (Gao et al., 2021)
- **E5**: "Multilingual E5 Text Embeddings" (Wang et al., 2022)

## 💡 Practical Guide

```
1. For most use cases: Use all-MiniLM-L6-v2
   - Fast
   - Good quality
   - Widely tested

2. For production:
   - Use all-mpnet-base-v2 if compute allows
   - Better quality

3. For your domain:
   - Fine-tune existing model
   - Or train from scratch with contrastive loss

4. Always normalize embeddings
5. Use cosine similarity for comparison
6. Test on your specific task
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete
