# LSTM (Long Short-Term Memory) Networks

LSTM is a type of recurrent neural network designed to learn long-term dependencies by using memory cells and gating mechanisms.

## 📋 Overview

**Problem Solved:** Vanishing gradients in vanilla RNNs
**Solution:** Memory cells (C_t) with multiplicative gates

## 🏗️ Architecture

### LSTM Cell

```
Input: x_t, h_{t-1}, C_{t-1}

Forget Gate:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell Candidate: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Output Gate:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

Cell State:     C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Hidden State:   h_t = o_t ⊙ tanh(C_t)
```

### Variants Implemented

1. **Vanilla LSTM**: Standard architecture
2. **Bidirectional LSTM**: Process sequence both ways
3. **Stacked LSTM**: Multiple layers

## 🎯 Use Cases

- Time series prediction
- Sequence classification
- Text generation
- Machine translation
- Video analysis

## 📊 Performance Comparison

| Architecture | Parameters | Speed | Long-term |
|-------------|-----------|-------|-----------|
| Vanilla RNN | Few | Fast | ❌ Vanishing |
| LSTM | More | Medium | ✅ Good |
| GRU | Medium | Faster | ✅ Good |
| Transformer | Most | Slowest | ✅ Best |

## 🚀 Quick Start

```python
import torch
import torch.nn as nn

# Create LSTM
lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=2, batch_first=True)

# Forward pass
x = torch.randn(32, 50, 100)  # [batch, seq_len, input_size]
output, (h_n, c_n) = lstm(x)

# output: [batch, seq_len, hidden_size]
# h_n: [num_layers, batch, hidden_size] (final hidden state)
# c_n: [num_layers, batch, hidden_size] (final cell state)
```

## 💡 Key Insights

1. **Cell State is Memory**: C_t accumulates information across time
2. **Forget Gate Controls Clearing**: Can discard irrelevant information
3. **Input Gate Controls Writing**: Can selectively update memory
4. **Output Gate Controls Reading**: Can selectively expose memory

## ⚠️ Common Issues

- Exploding gradients: Use gradient clipping
- Slow convergence: LSTM training can be slow
- Overfitting: Add dropout between layers
- Vanishing Objective: Not a problem for LSTM (designed to solve it)

## 📈 Training Tips

1. Use gradient clipping (`max_norm=1.0`)
2. Reduce learning rate (LSTMs are sensitive)
3. Use LSTM dropout (varies per timestep)
4. Initialize orthogonally for better gradient flow

## 📚 References

- **Original Paper**: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- **Tutorial**: Understanding LSTM Networks (Colah's blog)

---

**Last Updated:** December 2024
**Status:** ✅ Complete with PyTorch implementation
