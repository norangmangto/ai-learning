# Hybrid Search System

This directory contains an implementation of hybrid search that combines lexical (keyword-based) and semantic (meaning-based) retrieval methods.

## Overview

Hybrid search combines the best of both worlds:
- **BM25** (lexical): Fast keyword matching, good for exact terms
- **Semantic Search** (embeddings): Understanding meaning, handles paraphrases

## Architecture

```
Query: "How do neural networks learn?"
   ↓
┌─────────────────────┬──────────────────────┐
│   BM25 Search       │  Semantic Search     │
│   (keyword-based)   │  (meaning-based)     │
└─────────────────────┴──────────────────────┘
   ↓                           ↓
   Results with scores         Results with scores
   ↓                           ↓
   └──────────── Fusion ────────────┘
                  ↓
            Final Ranked Results
```

## Search Methods

### 1. BM25 (Best Match 25)
- **Type**: Lexical/keyword search
- **Speed**: Very fast
- **Strengths**:
  - Exact keyword matches
  - No ML model needed
  - Interpretable scores
- **Weaknesses**:
  - Vocabulary mismatch problem
  - Misses semantic meaning

### 2. Semantic Search
- **Type**: Dense vector search
- **Model**: Sentence-BERT embeddings
- **Strengths**:
  - Understands meaning
  - Handles paraphrases
  - Cross-lingual capability
- **Weaknesses**:
  - May miss exact keywords
  - Requires pre-computed embeddings

### 3. Hybrid Search (α-weighted)
- **Formula**: `score = (1-α) × BM25_score + α × semantic_score`
- **α = 0**: Pure BM25
- **α = 1**: Pure semantic
- **α = 0.5**: Balanced (recommended)

### 4. Reciprocal Rank Fusion (RRF)
- **Formula**: `RRF_score = Σ(1 / (k + rank))`
- **Advantage**: Rank-based, no score normalization needed
- **Use Case**: Combining multiple rankers

## Usage

```bash
# Run hybrid search demo
python train_pytorch.py

# Test with custom queries
python train_pytorch.py --corpus_size 5000 --top_k 10
```

## Configuration

```python
CONFIG = {
    'semantic_model': 'all-MiniLM-L6-v2',  # Sentence-BERT model
    'corpus_size': 1000,                    # Number of documents
    'top_k': 10,                            # Results to return
    'alpha': 0.5,                           # Hybrid weight (0=BM25, 1=semantic)
}
```

## Example Results

**Query**: "How do neural networks learn?"

**BM25 Results** (keyword matching):
1. "Neural networks learn through backpropagation..." ⭐ (contains exact terms)
2. "Deep learning training methods..."

**Semantic Results** (meaning-based):
1. "Training algorithms for artificial neural systems..." ⭐ (similar meaning)
2. "How gradient descent optimizes network parameters..."

**Hybrid Results** (combined):
1. Best of both approaches ⭐⭐
2. Balanced relevance and precision

## When to Use Each Method

### Use BM25 When:
- Exact keyword matches are important
- User queries contain specific technical terms
- Fast performance is critical
- No ML infrastructure available

### Use Semantic Search When:
- Understanding intent is important
- Users phrase queries differently
- Cross-lingual search needed
- Conceptual similarity matters

### Use Hybrid Search When:
- Building production search systems
- Want robust performance across query types
- Need both precision and recall
- General-purpose search application

## Tuning the α Parameter

```python
# More weight on keywords (technical docs, legal)
alpha = 0.3  # 70% BM25, 30% semantic

# Balanced (general purpose)
alpha = 0.5  # 50% BM25, 50% semantic

# More weight on meaning (conversational, QA)
alpha = 0.7  # 30% BM25, 70% semantic
```

## Performance Comparison

| Method | Speed | Accuracy | Handles Paraphrase | Keyword Precision |
|--------|-------|----------|-------------------|-------------------|
| BM25 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | ⭐⭐⭐⭐⭐ |
| Semantic | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Hybrid | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| RRF | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## Applications

- **Document Retrieval**: Search large document collections
- **Question Answering**: Find relevant passages for questions
- **E-commerce**: Product search with both keywords and intent
- **Enterprise Search**: Internal knowledge bases
- **Legal Discovery**: Find relevant cases and documents
- **Academic Search**: Research paper retrieval

## Implementation Details

### Score Normalization
```python
# Normalize BM25 scores to [0, 1]
normalized_score = (score - min_score) / (max_score - min_score)

# Semantic scores already in [0, 1] (cosine similarity)
```

### Reciprocal Rank Fusion
```python
# k = 60 (common choice)
rrf_score = sum(1 / (60 + rank) for rank in all_ranks)
```

## Requirements

```bash
pip install torch sentence-transformers rank-bm25 nltk
```

## Advanced Features

### Multi-language Support
```python
# Use multilingual sentence transformers
model = 'paraphrase-multilingual-MiniLM-L12-v2'
```

### Custom Tokenization
```python
# Improve BM25 with better tokenization
- Remove stopwords
- Stemming/lemmatization
- Domain-specific tokenizers
```

### Caching
```python
# Cache embeddings for faster search
embeddings = model.encode(corpus, show_progress_bar=True)
np.save('embeddings.npy', embeddings)
```

## References

- BM25: [Robertson & Walker, 1994](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz)
- Sentence-BERT: [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)
- RRF: [Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
