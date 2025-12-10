# GRU (Gated Recurrent Unit)

GRU is a simplified variant of LSTM with fewer gates, offering better efficiency while maintaining comparable performance.

## 📋 Overview

**Simplification of:** LSTM architecture
**Trade-off:** Fewer parameters, faster training, slightly less expressive

## 🏗️ Architecture

### GRU Cell

```
Input: x_t, h_{t-1}

Reset Gate:    r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
Update Gate:   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
Candidate:     h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
Hidden State:  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### GRU vs LSTM

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (input, forget, output) | 2 (reset, update) |
| Cell State | Separate | None |
| Parameters | More (3×) | Fewer (3/4×) |
| Training Time | Slower | Faster ✅ |
| Gradient Flow | Better | Good |
| Complexity | Higher | Lower ✅ |

## 🎯 When to Use

```
Use GRU when:
✅ Limited computational budget
✅ Training time is critical
✅ Medium-length sequences (< 500)
✅ Dataset size is small-medium

Use LSTM when:
✅ Long sequences (> 1000)
✅ Complex dependencies
✅ Computation budget is high
✅ Extra expressiveness needed
```

## 🚀 Quick Start

```python
import torch
import torch.nn as nn

# Create GRU
gru = nn.GRU(input_size=100, hidden_size=256, num_layers=2, batch_first=True)

# Forward pass
x = torch.randn(32, 50, 100)  # [batch, seq_len, input_size]
output, h_n = gru(x)

# output: [batch, seq_len, hidden_size]
# h_n: [num_layers, batch, hidden_size] (final hidden state)
```

## 📊 Empirical Comparison

| Dataset | GRU | LSTM | Winner |
|---------|-----|------|--------|
| Machine Translation | 95% | 96% | LSTM |
| Sentiment Analysis | 92% | 92% | Tie |
| POS Tagging | 97% | 97.5% | LSTM |
| Machine Comprehension | 91% | 92% | LSTM |

**Conclusion**: Performance often similar, but GRU is faster

## 💡 Key Insights

### Reset Gate
- Controls how much of previous state to remember
- When 0: Start fresh (forget everything)
- When 1: Keep all of previous state

### Update Gate
- Controls how much to update state
- When 0: Keep previous state unchanged
- When 1: Use candidate completely

### No Separate Cell State
- GRU mixes memory and output
- LSTM keeps them separate
- GRU simpler but slightly less flexible

## ⚠️ Potential Issues

1. **Worse on very long sequences**: Use LSTM if seq_len > 1000
2. **May underfit**: Fewer parameters = less capacity
3. **Update gate saturation**: Can get stuck on gradual values

## 📈 Training Tips

1. Start with GRU (faster iteration)
2. Switch to LSTM only if performance plateaus
3. Same tricks apply (gradient clipping, dropout)
4. Slightly lower learning rates than feedforward

## 🔄 Comparison with Alternatives

| Model | Speed | Performance | Use Case |
|-------|-------|-------------|----------|
| Vanilla RNN | Fast | Poor | Educational |
| GRU | Medium | Good ✅ | Practical |
| LSTM | Slow | Better | Complex |
| Transformer | Slowest | Best | State-of-art |

## 📚 References

- **Paper**: "Learning Phrase Representations with RNNs" (Cho et al., 2014)
- **Comparison**: GRU vs LSTM empirical studies show mixed results

## 🎓 Learning Outcomes

- [x] Understand GRU gates and computation
- [x] Know when to use GRU vs LSTM
- [x] Implement GRU in PyTorch
- [x] Compare with alternatives
- [x] Training best practices

---

**Last Updated:** December 2024
**Status:** ✅ Complete with PyTorch implementation
