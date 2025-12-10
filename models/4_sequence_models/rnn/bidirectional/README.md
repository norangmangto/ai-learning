# Bidirectional RNNs

Process sequences in both directions (forward and backward) to capture context from both sides.

## 📋 Overview

**Key Advantage:** Full contextual understanding (both past and future)
**Trade-off:** Can't use for generation, double parameters and computation

## 🏗️ Architecture

### Forward and Backward Processing

```
Sentence: "The cat sat on the mat"

Forward:
The → LSTM → h_f_1
cat → LSTM → h_f_2
sat → LSTM → h_f_3
...

Backward:
mat ← LSTM ← h_b_6
the ← LSTM ← h_b_5
...

Combined (for "cat"):
concat(h_f_2, h_b_2) = final representation
```

### Output Aggregation

**Different strategies for final representation:**

1. **Last Hidden State**
   ```python
   output = concat(h_f[-1], h_b[0])
   # Only uses final/initial states
   ```

2. **Mean Pooling**
   ```python
   output = mean(concat(h_f, h_b))
   # Averages all positions
   ```

3. **Max Pooling**
   ```python
   output = max(concat(h_f, h_b))
   # Takes maximum activation
   ```

4. **Attention Pooling**
   ```python
   weights = attention_mechanism(h_f, h_b)
   output = sum(weights * concat(h_f, h_b))
   # Learns which positions to attend to
   ```

## 🎯 Use Cases

### Perfect For:
- ✅ Classification (sentiment, topic, intent)
- ✅ Sequence tagging (NER, POS tagging)
- ✅ Question answering (finding answer span)
- ✅ Semantic similarity
- ✅ Any task where full context helps

### Not For:
- ❌ Generation (can't look into future)
- ❌ Real-time processing (need full sequence)
- ❌ Streaming data (must process all first)

## 🚀 Quick Start

```python
import torch
import torch.nn as nn

# Create bidirectional LSTM
bilstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True  # Key parameter!
)

# Forward pass
x = torch.randn(32, 50, 100)  # [batch, seq_len, input_size]
output, (h_n, c_n) = bilstm(x)

# Output shape: [batch, seq_len, hidden_size * 2]  (doubled!)
# h_n shape: [num_layers * 2, batch, hidden_size]  (forward + backward)
```

## 📊 Efficiency Considerations

| Metric | Unidirectional | Bidirectional |
|--------|----------------|---------------|
| Parameters | N | 2N |
| Computation | T | 2T |
| Memory | M | 2M |
| Context | Past only | Both |
| Accuracy | Baseline | +2-5% typically |

## 💡 Key Insights

### Information Dependency
```
Forward LSTM:
x_0 → h_1 → h_2 → h_3
      ↑ depends on past

Backward LSTM:
x_3 ← h_2 ← h_1 ← h_0
      ↑ depends on future

Bidirectional:
x_1 ← → full context from both directions
```

### Why It Helps
```
Sentence: "The movie was bad"
Position: "bad"

Unidirectional (left→):
- Sees "The", "movie", "was", "bad"
- Must predict from these

Unidirectional (right←):
- Can't use (backward not typical for generation)

Bidirectional:
- Sees full context: "The movie was bad"
- Knows "was" and "bad" are critical
- Better understanding!
```

## 🔄 Different RNN Architectures

### UniLSTM (Unidirectional)
```python
lstm = nn.LSTM(100, 256, bidirectional=False)
```
- Input → Output: left to right
- Output shape: [batch, seq, 256]

### BiLSTM (Bidirectional)
```python
bilstm = nn.LSTM(100, 256, bidirectional=True)
```
- Input → Output: both directions
- Output shape: [batch, seq, 512]

### Bidirectional with Attention
```python
# Custom attention over bidirectional outputs
bioutput, _ = bilstm(x)  # [batch, seq, 512]
attention_weights = attention(bioutput)
context = sum(attention_weights * bioutput)
```

## 📈 Performance Impact

### Typical Improvements (BiLSTM vs UniLSTM)
- **Text Classification**: +3-5% accuracy
- **NER Tagging**: +2-4% F1
- **Semantic Similarity**: +5-10% accuracy
- **QA Systems**: +2-3% EM

## ⚠️ Common Issues

1. **Double Parameters**: Need more memory
2. **Training Time**: ~2× slower
3. **Can't Generate**: Can't produce outputs in real-time
4. **Sequence Dependency**: Need full sequence before outputting

## 🎓 Learning Outcomes

- [x] Understand bidirectional processing
- [x] Know output aggregation strategies
- [x] Implement BiLSTM/BiGRU in PyTorch
- [x] Know when bidirectional helps
- [x] Attention-weighted pooling

## 📚 References

- **Original**: Neural Machine Translation (Bahdanau et al., 2015)
- **BiLSTM Tagging**: BiLSTM-CRF (Ma & Hovy, 2016)
- **Attention**: Show, Attend, Tell (Xu et al., 2015)

---

**Last Updated:** December 2024
**Status:** ✅ Complete with multiple variants and pooling strategies
