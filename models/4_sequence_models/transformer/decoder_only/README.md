# Decoder-Only Transformer (GPT-style)

Autoregressive decoders for text generation using causal attention.

## 📋 Overview

**Architecture:** Transformer decoder only
**Masking:** Causal (cannot attend to future positions)
**Best For:** Text generation, completion, language modeling

## 🏗️ Architecture

```
Input Tokens
        ↓
Positional Encoding
        ↓
Transformer Decoder Layers × N
  ├─ Masked Multi-head Self-Attention (Causal)
  ├─ Feed-Forward
  └─ Layer Normalization & Residuals
        ↓
Output Logits (next token prediction)
```

## 🎯 Key Features

### Causal Masking
```
Position i can only attend to positions j where j ≤ i

Attention Matrix (lower triangular):
     0  1  2  3
0 [  1  0  0  0 ]
1 [  1  1  0  0 ]
2 [  1  1  1  0 ]
3 [  1  1  1  1 ]
     ↑ autoregressive property
```

### Autoregressive Generation
```
Step 1: Input [<START>]
        Output: logits for token 1
        Sample: "The"

Step 2: Input [<START>, "The"]
        Output: logits for token 2
        Sample: "cat"

Step 3: Input [<START>, "The", "cat"]
        Output: logits for token 3
        Sample: "sat"
        ...continues until <END> or max_length
```

## 💡 GPT vs BERT

| Aspect | GPT (Decoder) | BERT (Encoder) |
|--------|---|---|
| Masking | Causal ✅ | None |
| Training | Next token prediction ✅ | Masked LM |
| Generation | Natural ✅ | Awkward |
| Understanding | Limited | Excellent ✅ |
| Training Speed | Medium | Fast ✅ |

## 🚀 Quick Start

```python
from train_pytorch import GPTDecoder

# Create model
model = GPTDecoder(
    vocab_size=100,
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=1024
)

# Training: next token prediction
logits, _ = model(input_ids)
loss = CrossEntropyLoss(logits.view(-1, vocab_size),
                        targets.view(-1))

# Generation: autoregressive sampling
generated = model.generate(
    start_tokens,
    max_length=50,
    temperature=1.0,
    top_k=10
)
```

## 📊 Generation Strategies

### 1. Greedy Decoding
```python
next_token = logits.argmax()
```
- Deterministic
- Repetitive
- Not ideal

### 2. Temperature Sampling
```python
logits = logits / temperature
probs = softmax(logits)
next_token = multinomial(probs)
```
- τ = 0.1: Sharp, confident
- τ = 1.0: Balanced
- τ = 2.0: Very random

### 3. Top-K Sampling
```python
# Only sample from top K tokens
mask[logits < top_k_value] = -inf
probs = softmax(logits)
next_token = multinomial(probs)
```
- Better quality than pure sampling
- Less repetition than greedy

### 4. Nucleus (Top-P) Sampling
```python
# Sample from smallest set of tokens with cumulative prob ≥ p
probs = softmax(logits)
sorted_probs = sort(probs, descending=True)
cumsum = cumsum(sorted_probs)
mask = cumsum > p
probs[mask] = 0
next_token = multinomial(probs)
```

## 📈 Applications

| Task | Example |
|------|---------|
| **Text Generation** | Continue story, creative writing |
| **Code Completion** | Auto-complete function, IDE |
| **Dialogue** | Chatbot response generation |
| **Summarization** | Generate summary from document |
| **Translation** | Sequence-to-sequence with encoder |

## ⚠️ Common Issues

1. **Repetition**: Use higher temperature or top-k
2. **Incoherence**: Model too small or undertrained
3. **Hallucination**: Model generates false information
4. **Slow Generation**: Decoding is inherently sequential

## 🎓 Learning Outcomes

- [x] Causal masking and autoregressive property
- [x] Next token prediction
- [x] Generation strategies (greedy, sampling, top-k)
- [x] Temperature control
- [x] Limitations and biases

## 📚 Key Papers

- **GPT**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- **Transformers**: "Attention Is All You Need" (Vaswani et al., 2017)

## 💡 Insights

**Why Causal Masking Works:**
- Prevents information leakage from future
- Enables efficient autoregressive generation
- Natural for language modeling objective

**Why Generation is Slow:**
- Must generate token-by-token
- Each token depends on all previous
- O(seq_len) sequential steps

**Perplexity vs Generation Quality:**
- Low perplexity ≠ Good generation
- Different objectives (likelihood vs human preference)
- Requires different evaluation metrics

---

**Last Updated:** December 2024
**Status:** ✅ Complete
