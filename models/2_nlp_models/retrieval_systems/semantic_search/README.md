# Semantic Search

Find most relevant documents from a corpus using semantic similarity.

## 📋 Overview

**Task:** Query + Corpus → Similar documents
**Output:** Ranked documents by relevance
**Speed:** Milliseconds to seconds depending on corpus size
**Method:** Dense embeddings + similarity

## 🎯 How It Works

```
User Query:
"How do I train a neural network?"

Corpus:
1. "Deep learning tutorial"
2. "Python basics"
3. "Training neural networks efficiently"
4. "JavaScript guide"
5. "Advanced machine learning"

Semantic Search:
Convert to embeddings
Compare similarities
Rank by relevance

Results (ranked):
1. "Training neural networks efficiently" (0.92)
2. "Advanced machine learning" (0.78)
3. "Deep learning tutorial" (0.75)
4. "Python basics" (0.15)
5. "JavaScript guide" (0.08)
```

## 🏗️ Architecture

### Basic Pipeline
```
Query
  ↓
Encode with sentence transformer
  ↓
[0.1, 0.2, 0.3, ..., 0.5]  (e.g., 384-dim)
  ↓
Compare with corpus embeddings
  ↓
Cosine similarity to each document
  ↓
Sort by similarity score
  ↓
Return top-k documents
```

### With Indexing (Large Corpus)
```
Corpus documents
  ↓
Pre-encode and index
  ↓
Store in:
- FAISS (fast approximate search)
- Elasticsearch (hybrid)
- Milvus (vector database)
- Weaviate (knowledge graphs)
  ↓
Query
  ↓
Fast approximate nearest neighbors
  ↓
Re-rank top results
  ↓
Return results
```

## 🚀 Quick Start: Basic Search

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus
documents = [
    "How to train a neural network",
    "Python programming basics",
    "Advanced deep learning techniques",
    "Web development with JavaScript",
    "Training machine learning models efficiently"
]

# Encode corpus (do once, save)
corpus_embeddings = model.encode(documents)

# Query
query = "What's the best way to train neural networks?"
query_embedding = model.encode(query)

# Search
similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]

# Rank results
top_k = 3
top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"Query: {query}\n")
for idx in top_indices:
    print(f"{idx+1}. {documents[idx]}")
    print(f"   Similarity: {similarities[idx]:.3f}\n")
```

## 🚀 Quick Start: FAISS (Fast Index)

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Encode corpus
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = ["doc1", "doc2", ..., "doc1000000"]  # Large corpus

corpus_embeddings = model.encode(documents)
corpus_embeddings = corpus_embeddings.astype('float32')

# Build FAISS index (fast approximate search)
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# Save index
faiss.write_index(index, 'corpus.index')

# Query
query = "my question"
query_embedding = model.encode([query]).astype('float32')

# Search (very fast even with millions!)
k = 5
distances, indices = index.search(query_embedding, k)

# Results
for i in range(k):
    doc_idx = indices[0][i]
    distance = distances[0][i]
    print(f"{i+1}. {documents[doc_idx]} (distance: {distance:.3f})")
```

## 📊 Semantic vs Keyword Search

### Keyword Search (BM25)
```
Query: "good movie"

Docs with keywords:
- "This movie is good" ✓ (exact match)
- "Is this film good?" ✗ (different words)

Problem: Misses synonyms, different phrasings
```

### Semantic Search
```
Query: "good movie"

Docs by meaning:
- "This movie is good" ✓ (0.95 similarity)
- "Is this film good?" ✓ (0.88 similarity) - "film" = "movie"!
- "Excellent cinema" ✓ (0.82 similarity) - "excellent" ≈ "good"

Advantage: Understands meaning beyond exact words
```

## 📈 Applications

| Domain | Use Case |
|--------|----------|
| **Search engines** | Web search, document retrieval |
| **E-commerce** | Product search by description |
| **Help systems** | Find relevant FAQs/docs |
| **Legal** | Find similar cases/contracts |
| **Scientific** | Paper discovery, citation matching |
| **Chat/QA** | Find relevant context for answers |
| **Recommendation** | Recommend similar products |

## 🎯 Implementation Patterns

### Pattern 1: Small Corpus (< 10k docs)
```python
# Pre-encode and store
embeddings = model.encode(documents)

# Query-time
query_emb = model.encode(query)
similarities = cosine_similarity([query_emb], embeddings)[0]
top_k = np.argsort(similarities)[::-1][:k]
```

**Pros:** Simple, no external dependencies
**Cons:** Slow with large corpus (O(n))

### Pattern 2: Medium Corpus (10k-1M)
```python
# Use FAISS for speed
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))

# Query-time
_, indices = index.search(query_emb.astype('float32'), k)
```

**Pros:** Fast O(log n) with approximate search
**Cons:** Need to manage index

### Pattern 3: Large Corpus (> 1M)
```python
# Use vector database
from pinecone import Pinecone

pc = Pinecone(api_key="...", index_name="documents")
# Upload embeddings (batched)
pc.upsert(vectors=[(id, embedding) for id, embedding in ...])

# Query
results = pc.query(query_emb, top_k=10, include_metadata=True)
```

**Pros:** Scalable, managed service
**Cons:** External dependency, latency

## 📊 Performance Metrics

```
For semantic search evaluation:

Recall@k: Of relevant documents, how many in top-k?
Precision@k: Of top-k results, how many are relevant?
MRR (Mean Reciprocal Rank): Average position of first relevant
NDCG (Normalized DCG): Ranking quality

Example:
Recall@10 = 0.85 → Found 85% of relevant docs in top 10
Precision@10 = 0.70 → 7 out of 10 results are relevant
```

## 💡 Improving Search Quality

### 1. Chunking Large Documents
```
Long documents don't fit in embeddings well
Solution: Split into chunks

Document:
"Introduction... methods... results... conclusion..."

Chunks:
- chunk_1: "Introduction..."
- chunk_2: "methods..."
- chunk_3: "results..."
- chunk_4: "conclusion..."

Index chunks, return document + chunk when found
```

### 2. Reranking
```
Fast retrieval → Rerank with slow model

Step 1: FAISS gets top-100 (fast)
Step 2: Cross-encoder reranks top-100 (slow)
Step 3: Return top-10 (accurate)

Result: Fast + accurate!
```

### 3. Hybrid Search
```
Combine semantic + keyword:
score = 0.7 * semantic_score + 0.3 * bm25_score

Benefits:
- Keyword catches exact matches
- Semantic catches meaning
- More robust
```

### 4. Domain Fine-tuning
```
Models trained on general text
Your domain has specific vocabulary

Solution:
1. Collect query-document pairs
2. Fine-tune embedding model
3. Better domain-specific search

Example:
- Medical papers: Fine-tune on PubMed
- Code search: Fine-tune on GitHub
- Legal: Fine-tune on contracts
```

## ⚠️ Common Pitfalls

1. **Query-document mismatch**
   ```
   Query: Sentence (short, question-like)
   Document: Long paragraph or page

   Problem: Different lengths → poor matching

   Solution: Use appropriate embedding model
   - "all-MiniLM-L6-v2" for similar length
   - Fine-tune if very different
   ```

2. **Language mismatch**
   ```
   Query in English, docs in mixed languages

   Solution: Use multilingual model
   - "multilingual-e5-base"
   - Detects language automatically
   ```

3. **Not updating corpus embeddings**
   ```
   Add new documents without updating index
   → New docs not searchable

   Solution:
   - Regularly re-index
   - Use incremental indexing
   ```

4. **Ignoring preprocessing**
   ```
   Different preprocessing → different results

   Solution:
   - Document preprocessing steps
   - Apply consistently
   ```

## 🎓 Learning Outcomes

- [x] Semantic vs keyword search
- [x] Dense embedding methods
- [x] Indexing strategies (FAISS, etc.)
- [x] Reranking and hybrid approaches
- [x] Domain-specific optimization

## 📚 Key Papers

- **Dense Retrieval**: "Dense Passage Retrieval" (Karpukhin et al., 2020)
- **Reranking**: "CrossEncoder: A Framework for Cross-Ranking" (Thakur et al., 2021)

## 💡 Production Checklist

```
✓ Choose embedding model (e.g., all-mpnet-base-v2)
✓ Encode corpus documents
✓ Set up index (FAISS or vector DB)
✓ Implement query encoding
✓ Test on sample queries
✓ Set up monitoring (latency, accuracy)
✓ Plan for updates (new documents)
✓ Consider chunking (for long docs)
✓ Plan reranking if needed (accuracy)
✓ Document preprocessing
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete
