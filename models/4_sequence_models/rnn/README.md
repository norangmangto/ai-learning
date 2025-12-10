# RNN Variants: LSTM, GRU, Bidirectional, and Attention

Recurrent Neural Networks (RNNs) process sequential data by maintaining hidden state across time steps. This module covers the main RNN architectures and variants.

## 📋 Overview

| Type | Release | Gates | Best For |
|------|---------|-------|----------|
| **Vanilla RNN** | 1997 | None | Educational |
| **LSTM** | 1997 | 3 (input, forget, output) | Long-term dependencies ✅ |
| **GRU** | 2014 | 2 (reset, update) | Efficient LSTM |
| **Bidirectional** | 1997 | Bi-directional processing | Understanding tasks |
| **Attention** | 2015 | Attention mechanism | Focus on relevant steps |

## 🏗️ Common Architecture

All RNNs follow the same principle:

```
h_t = f(x_t, h_{t-1})
y_t = g(h_t)
```

Where:
- `x_t`: Input at time step t
- `h_t`: Hidden state (memory)
- `y_t`: Output at time step t
- `f`: RNN function (different for each variant)

## 1️⃣ LSTM (Long Short-Term Memory)

### Problem with Vanilla RNN

```
Gradient Flow Issue:
    ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
h_0 → h_1 → h_2 → ... → h_T

Vanishing Gradients:
∂L/∂h_0 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_1/∂h_0
          ↑ product of small values (< 1) → vanishes
```

### LSTM Solution

**Three Gates Control Information Flow:**

```
forget_gate = σ(W_f · [h_{t-1}, x_t] + b_f)
input_gate = σ(W_i · [h_{t-1}, x_t] + b_i)
output_gate = σ(W_o · [h_{t-1}, x_t] + b_o)

candidate = tanh(W_c · [h_{t-1}, x_t] + b_c)

C_t = forget_gate · C_{t-1} + input_gate · candidate
h_t = output_gate · tanh(C_t)
```

**Key Components:**

| Gate | Purpose | Formula | Range |
|------|---------|---------|-------|
| **Forget** | Erase old info | σ(...) | [0,1] |
| **Input** | Add new info | σ(...) | [0,1] |
| **Output** | Select what to reveal | σ(...) | [0,1] |
| **Candidate** | What to add | tanh(...) | [-1,1] |

### Cell State vs Hidden State

```
Cell State (C_t):
├─ Memory: accumulates information
├─ Additive updates (forget + input)
├─ Constant error flow (multiplication by forget gate)
└─ Prevents vanishing gradients

Hidden State (h_t):
├─ Output: what's visible to next layer
├─ Modulated by output gate
└─ Passed to next time step and final layer
```

### Unrolled LSTM Over Time

```
t=0          t=1          t=2
x_0          x_1          x_2
 ↓           ↓            ↓
[LSTM]      [LSTM]       [LSTM]
 ↓ ↓          ↓ ↓          ↓ ↓
h_0 C_0     h_1 C_1      h_2 C_2
 ↓           ↓            ↓
y_0          y_1          y_2
```

### Advantages
✅ Solves vanishing gradient problem
✅ Long-term dependency learning
✅ Industry standard
✅ Well-understood

### Disadvantages
❌ Complex (7 matrix multiplications)
❌ Computationally expensive
❌ More parameters to tune

---

## 2️⃣ GRU (Gated Recurrent Unit)

### Simplified LSTM

**Two Gates Instead of Three:**

```
reset_gate = σ(W_r · [h_{t-1}, x_t] + b_r)
update_gate = σ(W_u · [h_{t-1}, x_t] + b_u)

candidate = tanh(W_h · [reset_gate · h_{t-1}, x_t] + b_h)
h_t = (1 - update_gate) · h_{t-1} + update_gate · candidate
```

### GRU vs LSTM

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 | 2 |
| Cell State | Yes (separate) | No |
| Parameters | More | Fewer (3/4 of LSTM) |
| Training Speed | Slower | Faster ✅ |
| Performance | Better on long sequences | Comparable on medium |
| Interpretability | Output gate | Simpler |

### Gate Intuition

```
Reset Gate:
├─ Controls how much previous state to remember
├─ If 0: ignore previous state
└─ If 1: use all of previous state

Update Gate:
├─ Controls how much to update state
├─ If 0: keep previous state
└─ If 1: use candidate (like LSTM input gate)
```

### Advantages
✅ Fewer parameters than LSTM
✅ Faster training
✅ Comparable performance
✅ Simpler to understand

### Disadvantages
❌ Slightly less expressive than LSTM
❌ No separate cell state (less flexibility)
❌ Not always better than LSTM

---

## 3️⃣ Bidirectional RNNs

### Processing Direction

**Unidirectional (Left-to-Right):**
```
x_0 → h_0 →
x_1 → h_1 →
x_2 → h_2 → ...
```
Only past context

**Bidirectional (Both Directions):**
```
         → h_f_0 →
x_0 ⟲
         ← h_b_0 ←

    concat(h_f, h_b) = context from both directions
```

### Architecture

```
Input Sequence: [x_0, x_1, x_2, x_3]

Forward Pass:
x_0 → LSTM → h_f_0 ↘
x_1 → LSTM → h_f_1 ↘
x_2 → LSTM → h_f_2 ↘
x_3 → LSTM → h_f_3 ↘

Backward Pass:
          ↙ h_b_0 ← LSTM ← x_0
          ↙ h_b_1 ← LSTM ← x_1
          ↙ h_b_2 ← LSTM ← x_2
          ↙ h_b_3 ← LSTM ← x_3

Output: [h_f_0 ⊕ h_b_0, h_f_1 ⊕ h_b_1, h_f_2 ⊕ h_b_2, h_f_3 ⊕ h_b_3]
where ⊕ = concatenation
```

### Information Aggregation

**Different pooling strategies for final representation:**

```python
# Last hidden state
output = hidden_states[-1]

# Mean pooling
output = hidden_states.mean(dim=0)

# Max pooling
output = hidden_states.max(dim=0)

# Attention pooling
output = sum(attention_weights * hidden_states)
```

### Use Cases

**Understanding vs Generation:**
- **Bidirectional**: Classification, tagging, understanding ✅
- **Unidirectional**: Generation, sequence prediction ✅

### Advantages
✅ Full context (past and future)
✅ Better for classification
✅ Works for understanding tasks
✅ Improved accuracy vs unidirectional

### Disadvantages
❌ Can't generate (needs future)
❌ Slower inference (need full sequence)
❌ Double parameters
❌ 2× memory usage

---

## 4️⃣ Attention Mechanism in RNNs

### Attention Problem

**Without Attention:**
```
Encoder compresses sequence into single vector
x_1, x_2, x_3, x_4 → LSTM → [final hidden state]
                          ↑ information bottleneck
                    loses info about x_1, x_2
```

**With Attention:**
```
Encoder produces sequence of states
x_1, x_2, x_3, x_4 → LSTM → [h_1, h_2, h_3, h_4]
                             ↑ maintain all information
Decoder can focus on relevant states at each step
```

### Attention Types

**1. Additive Attention (Bahdanau):**
```
score(s_t, h_i) = v^T · tanh(W · [s_t; h_i])
```
General purpose, slower

**2. Multiplicative Attention (Luong):**
```
score(s_t, h_i) = s_t^T · W · h_i
```
Faster, requires matching dimensions

**3. Scaled Dot-Product:**
```
score(s_t, h_i) = (s_t^T · h_i) / sqrt(d_k)
```
Used in Transformers, numerically stable

**4. Self-Attention:**
```
Sequence attends to itself
h_i = sum_j attention(h_i, h_j) · h_j
```

### Computation Steps

```
1. Query: Current state s_t (what am I looking for?)
2. Keys: Encoder states h_i (what info is available?)
3. Values: Encoder states h_i (what info to aggregate?)

scores = Attention(Query, Keys) → [length]
         ↓
weights = softmax(scores) → probabilities over positions
         ↓
context = sum(weights * Values) → weighted information
         ↓
output = combine(state, context) → updated representation
```

### Attention Visualization

```
                    weights for position t:
Decoder at t_3:     [0.02, 0.85, 0.10, 0.03]
                     ↓    ↓    ↓    ↓
Encoder:     [h_0  h_1  h_2  h_3]
                     ↑ focus here

context = 0.02*h_0 + 0.85*h_1 + 0.10*h_2 + 0.03*h_3
```

### Use Cases

- Seq2Seq (Machine Translation)
- Question Answering
- Attention-based Caption Generation
- Visual Question Answering

### Advantages
✅ Interpretability (see what model attends to)
✅ Better handling of long sequences
✅ Improves accuracy significantly
✅ Foundation for Transformers

### Disadvantages
❌ Added complexity
❌ More parameters
❌ Computation: O(seq_len²)

---

## 📊 Comparison: LSTM vs GRU vs Attention-based

| Metric | LSTM | GRU | Attention |
|--------|------|-----|-----------|
| Parameters | Many | Medium | Many |
| Speed | Medium | Fast ✅ | Slow |
| Accuracy | High ✅ | Medium | High ✅ |
| Long Sequences | Good ✅ | Good | Best ✅ |
| Interpretability | Medium | Medium | High ✅ |
| Complexity | High | Medium | Very High |

---

## 🎯 When to Use What

```
Task: Classification
├─ Use: Bidirectional LSTM
├─ Why: Full context, standard approach
└─ Example: Sentiment analysis

Task: Sequence Generation
├─ Use: LSTM or GRU (unidirectional)
├─ Why: Can only use past context
└─ Example: Text generation

Task: Long Sequences (> 100 steps)
├─ Use: Attention-based or Transformer
├─ Why: Solves vanishing gradient better
└─ Example: Machine translation

Task: Fast Training, Limited Data
├─ Use: GRU
├─ Why: Fewer parameters, faster training
└─ Example: Small dataset NLP
```

---

## 💡 Implementation Tricks

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevents exploding gradients in RNNs

### Dropout
```python
# Varies per timestep, consistent across RNN
rnn = nn.LSTM(..., dropout=0.5)
```

### Learning Rate
```python
# RNNs are sensitive to learning rate
optimizer = Adam(model.parameters(), lr=0.001)  # Often need smaller LR
```

### Weight Initialization
```python
# Orthogonal initialization helps gradient flow
nn.init.orthogonal_(rnn.weight_hh_l0)
```

---

## 🚀 Training Tips

1. **Start with GRU**: Faster, comparable performance
2. **Use Bidirectional for Classification**: Better accuracy
3. **Add Attention for Long Sequences**: Improves results
4. **Monitor Gradients**: RNNs gradient flow is critical
5. **Use Gradient Clipping**: Essential for stability

---

## 📚 Resources

### Key Papers
- **LSTM**: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- **GRU**: "Learning Phrase Representations with RNNs" (Cho et al., 2014)
- **Attention**: "Neural Machine Translation with Attention" (Bahdanau et al., 2015)
- **Self-Attention**: "Attention Is All You Need" (Vaswani et al., 2017)

### Intuitions
- **Forget Gate**: "Do I need to remember this?"
- **Input Gate**: "Is this information important?"
- **Output Gate**: "Should I reveal this state?"

---

## ⚠️ Common Issues

1. **Vanishing Gradients**: Use LSTM/GRU
2. **Exploding Gradients**: Use gradient clipping
3. **Slow on Long Sequences**: Add attention
4. **Poor Convergence**: Reduce learning rate, use warm-up
5. **Overfitting**: Add dropout

---

**Last Updated:** December 2024
**Status:** ✅ Complete with 5 RNN variant implementations
