# Text Summarization

Generate concise summaries from longer texts automatically.

## 📋 Overview

**Task:** Long text → Short summary
**Types:** Extractive, Abstractive, Hybrid
**Baseline:** TF-IDF sentence ranking
**Modern:** Transformer-based (BART, T5, Pegasus)

## 🎯 Two Approaches

### Extractive Summarization
```
Original text:
"John went to the park. He played basketball.
He made many friends. He went home happy."

Extract key sentences:
"John went to the park. He made many friends."

How: Rank sentences by importance, select top K
```

**Advantages:**
- Fast and simple
- Preserves original wording
- High ROUGE scores
- Works without training

**Disadvantages:**
- Jerky, unnatural flow
- Can't rephrase
- May include redundant info

### Abstractive Summarization
```
Original text:
"John went to the park. He played basketball.
He made many friends. He went home happy."

Generated summary:
"John had fun at the park playing basketball."

How: Generate new sentences capturing essence
```

**Advantages:**
- Natural, fluent output
- Can rephrase and condense
- More human-like

**Disadvantages:**
- Requires training data
- Can hallucinate
- Slower inference
- Needs evaluation metric

## 📊 Extractive Methods

### 1. TF-IDF Ranking
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

sentences = text.split('.')
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

# Score each sentence
scores = tfidf_matrix.sum(axis=1).A1
top_indices = np.argsort(scores)[-k:]
summary_sentences = [sentences[i] for i in sorted(top_indices)]
summary = '. '.join(summary_sentences)
```

### 2. TextRank (Graph-based)
```
Build graph:
- Nodes = sentences
- Edges = similarity
- Use PageRank algorithm

Example:
Sentence 1 → (similar) → Sentence 3
          ↘             ↙
           Sentence 2

PageRank: Which sentences are central?
```

### 3. Latent Semantic Analysis (LSA)
```
SVD decomposition of TF-IDF matrix
Find dominant concepts
Select sentences best representing them
```

## 🚀 Quick Start: Extractive

```python
# Simple TF-IDF based
from sklearn.feature_extraction.text import TfidfVectorizer

def extractive_summary(text, num_sentences=3):
    sentences = text.split('. ')
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    scores = tfidf.sum(axis=1).A1
    top_idx = np.argsort(scores)[-num_sentences:]
    return '. '.join([sentences[i] for i in sorted(top_idx)])

text = """Machine learning is powerful.
          It powers recommendation systems.
          It also drives computer vision.
          Neural networks learn from data."""

summary = extractive_summary(text, num_sentences=2)
# Output: "Machine learning is powerful. Neural networks learn from data."
```

## 📊 Abstractive Methods

### Sequence-to-Sequence
```
Encoder-Decoder architecture:

Long text
    ↓ (Encoder)
  [Context vector]
    ↓ (Decoder)
  Summary (word-by-word)
```

### Transformer-based (BART, T5)

```
Original transformer with:
- Pretrained on denoising task
- Fine-tuned on summary data
- Can generate abstractive summaries

Input: [Full article text]
Output: [Generated summary]
```

## 🚀 Quick Start: Abstractive (Transformers)

```python
from transformers import pipeline

# Using Hugging Face pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
Machine learning is a type of artificial intelligence.
It enables computers to learn from data without explicit programming.
Deep learning is a subset of machine learning.
It uses neural networks with multiple layers.
"""

summary = summarizer(text, max_length=50, min_length=30)[0]['summary_text']
print(summary)
# "Machine learning is AI that learns from data.
#  Deep learning uses multi-layer neural networks."
```

## 📈 Popular Abstractive Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| BART | 400M | Medium | High | General |
| T5 | 220M-11B | Slow | High | Flexible |
| Pegasus | 568M | Medium | High | News |
| mBART | 600M | Medium | High | Multilingual |

## 📊 Evaluation Metrics

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

```
ROUGE-1: Unigram (single word) overlap
         Compares individual words

ROUGE-2: Bigram (word pair) overlap
         Compares consecutive word pairs

ROUGE-L: Longest common subsequence
         Captures word order

Example:
Reference: "The cat sat on the mat"
Generated: "A cat is on the mat"

ROUGE-1 overlap: 5/6 = 83%
         (all words except "the" vs "A")
```

### BLEU (Machine Translation)
```
Precision of n-grams
Penalizes brevity

Less suitable for summarization
(Doesn't measure content preservation)
```

### Semantic Similarity
```
Modern approach: Use embeddings
Compute cosine similarity between summary and text
Higher similarity = more faithful

Example:
Text embedding ≈ Summary embedding
→ Summary captured essence
```

## 🔄 Extractive vs Abstractive Comparison

| Aspect | Extractive | Abstractive |
|--------|-----------|-----------|
| **Quality** | 60-70 | 75-85 |
| **Fluency** | Poor | Good ✓ |
| **Training** | None needed | Need data |
| **Speed** | ⚡⚡⚡ | ⚡ |
| **Hallucination** | No | Possible |
| **Reproducibility** | 100% | Sampling |

## 🎯 Hybrid Approach

```
Combine both:

1. Extract key sentences (extractive)
2. Rephrase/fuse them (abstractive)

Advantages:
- Better content preservation
- More natural language
- Reduced hallucinations
```

## ⚠️ Common Issues

1. **Hallucination in abstractive**
   ```
   Text: "John is tall"
   Generated: "John is a basketball player" ✗

   Solution:
   - Use copy mechanism
   - Add faithfulness constraint
   - Use extractive component
   ```

2. **Redundancy in extractive**
   ```
   Selected sentences:
   "Apple released iPhone 15."
   "Apple announced iPhone 15."

   Solution:
   - Maximum marginal relevance
   - Penalize similar sentences
   ```

3. **Missing key points**
   ```
   Important facts not in summary

   Solution:
   - Use content selection
   - Fine-tune on domain data
   - Increase summary length
   ```

4. **Length control**
   ```
   Model ignores max_length parameter

   Solution:
   - Use beam search with constraints
   - Post-processing truncation
   - Fine-tune on similar lengths
   ```

## 📈 Applications

| Domain | Type | Example |
|--------|------|---------|
| **News** | Both | Headline + summary |
| **Scientific** | Extractive | Abstract from paper |
| **Medical** | Extractive | Clinical note summary |
| **Customer reviews** | Both | Product review digest |
| **Social media** | Abstractive | Thread summary |

## 🎯 Decision Guide

```
When to use extractive:

Small documents?         → Extractive
Need no training?        → Extractive
Want guaranteed truthfulness? → Extractive
Domain-specific?         → Extractive
Limited compute?         → Extractive

When to use abstractive:

Long documents?          → Abstractive
Have training data?      → Abstractive
Want fluent prose?       → Abstractive
General domain?          → Abstractive
Compute available?       → Abstractive
```

## 🎓 Learning Outcomes

- [x] Extractive vs abstractive
- [x] TF-IDF and graph-based extraction
- [x] Sequence-to-sequence models
- [x] Transformer-based abstractive
- [x] ROUGE evaluation metrics
- [x] Evaluation challenges

## 📚 Key Papers

- **Extractive**: "A Linear Selection-to-Sequence Model" (Nallapati et al., 2017)
- **BART**: "BART: Denoising Sequence-to-Sequence Pre-training" (Lewis et al., 2019)
- **Pegasus**: "Pre-training with Extracted Gap-sentences" (Zhang et al., 2019)

## 💡 Modern Approach

```
1. For production use → BART or Pegasus
2. For custom domain → Fine-tune on your data
3. For interpretability → Extractive + visualization
4. For speed → TF-IDF extractive baseline

Example pipeline:
User input → BART summary → Post-process → Output
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete
