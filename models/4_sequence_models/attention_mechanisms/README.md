# Attention Mechanisms in Deep Learning

Attention allows models to focus on relevant parts of input when producing output. This is crucial for sequence-to-sequence tasks and transformers.

## 📋 Overview

Attention answers the question: **"What should I focus on now?"**

Instead of compressing all information into a fixed vector, attention dynamically selects relevant input parts.

## 🏗️ Core Concept

### Attention Function

```
Attention(Query, Key, Value) = softmax(score(Query, Key)) · Value
```

**Components:**
- **Query**: "What am I looking for?" (from decoder)
- **Key**: "What information do I have?" (from encoder)
- **Value**: "What should I aggregate?" (from encoder)
- **Score Function**: Determines relevance

### Intuitive Example

```
Task: Machine Translation (English → French)

English: "The cat sat on the mat"
          [e_1 e_2 e_3 e_4 e_5 e_6]

French:  "Le chat était assis sur le tapis"

When generating "chat" (cat):
Query = current decoder state
Key/Value = all English encoder states

Attention scores:
"The": 0.02    (not relevant)
"cat": 0.85    ← highest (relevant!)
"sat": 0.08
"on": 0.03
"the": 0.01
"mat": 0.01

context = 0.85 * [cat embedding] + 0.08 * [sat embedding] + ...
```

---

## 1️⃣ Additive Attention (Bahdanau)

### Formula

```
score(s_t, h_i) = v^T · tanh(W · [s_t; h_i])

where:
s_t = decoder state
h_i = encoder state i
W = weight matrix
v = weight vector
[·; ·] = concatenation
```

### Computation Steps

```
1. Concatenate: [s_t; h_i]  → (2*d_h)
2. Linear: W @ [s_t; h_i] → d_a
3. Tanh: tanh(...)          → d_a (non-linearity)
4. Dot: v^T @ ...           → scalar (score for h_i)
5. Repeat for all i         → [scores] of length T
6. Softmax: normalize       → [weights] summing to 1
7. Sum: Σ weights_i * h_i   → context vector
```

### Python Implementation

```python
class AdditiveAttention(nn.Module):
    def __init__(self, d_h, d_a):
        super().__init__()
        self.W = nn.Linear(2 * d_h, d_a)
        self.v = nn.Linear(d_a, 1)

    def forward(self, query, keys, values):
        # query: (batch, d_h)
        # keys, values: (batch, T, d_h)

        batch_size, T, _ = keys.size()

        # Expand query
        query = query.unsqueeze(1)  # (batch, 1, d_h)
        query = query.expand(-1, T, -1)  # (batch, T, d_h)

        # Concatenate
        combined = torch.cat([query, keys], dim=-1)  # (batch, T, 2*d_h)

        # Score
        scores = self.W(combined)  # (batch, T, d_a)
        scores = torch.tanh(scores)
        scores = self.v(scores)  # (batch, T, 1)
        scores = scores.squeeze(-1)  # (batch, T)

        # Softmax
        weights = F.softmax(scores, dim=-1)  # (batch, T)

        # Context
        context = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, T)
            values  # (batch, T, d_h)
        ).squeeze(1)  # (batch, d_h)

        return context, weights
```

### Advantages
✅ General, works for any dimensions
✅ Non-linear scoring function
✅ Proven effective in practice

### Disadvantages
❌ More parameters (W and v)
❌ Slower than multiplicative
❌ Extra tanh computation

---

## 2️⃣ Multiplicative Attention (Luong)

### Formula

```
score(s_t, h_i) = s_t^T · W · h_i

where:
s_t = decoder state (d_h)
W = weight matrix (d_h × d_h)
h_i = encoder state i (d_h)
result = scalar
```

### Simplified Version

```
score(s_t, h_i) = s_t^T · h_i  (when W = I)
= dot product
```

### Computation Steps

```
1. Matrix multiply: s_t @ W    → (d_h)
2. Dot product: ... @ h_i^T     → scalar
3. Repeat for all i             → [scores]
4. Softmax                       → [weights]
5. Weighted sum                  → context
```

### Python Implementation

```python
class MultiplicativeAttention(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.W = nn.Linear(d_h, d_h)

    def forward(self, query, keys, values):
        # query: (batch, d_h)
        # keys, values: (batch, T, d_h)

        # Project query
        query = self.W(query)  # (batch, d_h)
        query = query.unsqueeze(1)  # (batch, 1, d_h)

        # Dot product
        scores = torch.bmm(query, keys.transpose(1, 2))  # (batch, 1, T)
        scores = scores.squeeze(1)  # (batch, T)

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # Context
        context = torch.bmm(
            weights.unsqueeze(1),
            values
        ).squeeze(1)

        return context, weights
```

### Advantages
✅ Simpler than additive
✅ Fewer parameters
✅ Faster computation
✅ Used in Transformers

### Disadvantages
❌ Requires matching dimensions
❌ Less expressive than additive

---

## 3️⃣ Scaled Dot-Product Attention

### Formula

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

where:
Q = queries (batch, T_q, d_k)
K = keys (batch, T_k, d_k)
V = values (batch, T_k, d_v)
√d_k = scaling factor
```

### Why Scale by √d_k?

```
Without scaling:
- Dot products can be very large
- Softmax becomes too peaked
- Gradients become small (saturation)

With scaling:
- Dot products normalized
- Gradients more stable
- Better convergence
```

### Multi-Head Version

```
for h in range(num_heads):
    Q_h = Q @ W_Q^h
    K_h = K @ W_K^h
    V_h = V @ W_V^h

    head_h = Attention(Q_h, K_h, V_h)

output = Concat(head_0, ..., head_{h-1}) @ W_O
```

**Motivation:**
- Different subspaces attend to different features
- Like multiple filters in CNNs
- Empirically improves performance

### Python Implementation

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, T, d_k)

        # Compute scores
        scores = torch.bmm(Q, K.transpose(1, 2))  # (batch, T_q, T_k)
        scores = scores / math.sqrt(self.d_k)

        # Apply mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # Apply attention
        output = torch.bmm(weights, V)  # (batch, T_q, d_v)

        return output, weights
```

### Advantages
✅ Foundation of Transformers
✅ Highly parallelizable
✅ Excellent empirical results
✅ Standard in industry

---

## 4️⃣ Self-Attention

### Concept

```
Sequence attends to itself
h_i = Σ_j softmax(score(h_i, h_j)) · h_j

Each position attends to all positions
```

### Use Cases

```
# Find relevant positions within same sequence
Input: "The cat sat on the mat"
       [w_1 w_2 w_3 w_4 w_5 w_6]

For "cat" (position 2):
├─ Attend to "The" (0.1)
├─ Attend to "cat" (0.8) ← itself
├─ Attend to "sat" (0.05)
└─ Attend to others (0.05)

→ Gathers information from whole sentence
```

### Architecture

```
Input X: (batch, T, d_model)

Q = X @ W_Q  # Self-query
K = X @ W_K  # Self-key
V = X @ W_V  # Self-value

Output = Attention(Q, K, V)
```

### Applications

- Transformers (encoder layers)
- Long-range dependencies
- Graph neural networks
- 3D vision models

---

## 🔄 Attention in Seq2Seq

### Without Attention

```
Encoder:
x_1 → LSTM → h_1
x_2 → LSTM → h_2
x_3 → LSTM → h_3
      ↓
   [h_3 only]  ← bottleneck
      ↓
Decoder:
   [h_3] → LSTM → y_1
   [h_3] → LSTM → y_2
   [h_3] → LSTM → y_3
```

Problem: All information compressed to h_3

### With Attention

```
Encoder:
x_1 → LSTM → h_1
x_2 → LSTM → h_2
x_3 → LSTM → h_3
      ↓ ↓ ↓ (all available)

Decoder:
             attention([h_1, h_2, h_3])
                    ↓
   y_1 ← LSTM ← context_1
   y_2 ← LSTM ← context_2
   y_3 ← LSTM ← context_3
```

Benefit: Each output step can focus on relevant inputs

---

## 📊 Comparison

| Type | Complexity | Speed | Parameters | When to Use |
|------|-----------|-------|-----------|------------|
| **Additive** | O(T·d_a) | Slow | Medium | General use |
| **Multiplicative** | O(T·d) | Medium | Few | Standard |
| **Scaled Dot-Prod** | O(T²) | Fast | None | Transformers |
| **Self** | O(T²) | Fast | None | Modern NNs |

---

## 💡 Key Insights

### 1. Attention is Interpretable
```python
# Visualize what model attends to
weights = attention_weights[0]  # First sample
visualization = weights.unsqueeze(-1) * input_images
# See which input regions affect output
```

### 2. Attention Solves Information Bottleneck
```
Without: [x_1, x_2, x_3] → fixed vector → output
With:    [x_1, x_2, x_3] → dynamic selection → output
```

### 3. Multiple Heads = Multiple Perspectives
```
Head 1: Attends to adjacent words (local)
Head 2: Attends to same subject (semantic)
Head 3: Attends to verbs (syntactic)
...
Combined: Rich representation
```

---

## 🚀 Implementation Tips

### Masking for Decoder

```python
# Prevent attending to future positions
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)

# In attention
scores = scores.masked_fill(mask == 0, -1e9)
```

### Attention Dropout

```python
# Regularization: randomly zero attention weights
attention_weights = F.dropout(attention_weights, p=0.1)
```

### Temperature Scaling

```python
# Control sharpness of attention
scores = scores / temperature
# Lower temp → sharper (more focused)
# Higher temp → softer (more distributed)
```

---

## 📚 Resources

### Key Papers
- **Additive**: "Neural Machine Translation by Jointly Learning to Align" (Bahdanau et al., 2015)
- **Multiplicative**: "Effective Approaches to Attention" (Luong et al., 2015)
- **Transformer**: "Attention Is All You Need" (Vaswani et al., 2017)

---

## ⚠️ Common Issues

1. **Attention All Ones**: Loss too high, gradients unstable
   - Solution: Gradient clipping, warmup

2. **Attention Collapse**: Attends uniformly to all positions
   - Solution: Check initialization, learning rate

3. **Memory Issues**: Computing T² attention matrix for large T
   - Solution: Use sparse attention, local attention

---

**Last Updated:** December 2024
**Status:** ✅ Complete with 4 attention mechanism implementations
