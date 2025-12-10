# Encoder-Only Transformer (BERT-style)

Bidirectional encoder transformers for understanding tasks like classification and tagging.

## 📋 Overview

**Architecture:** Transformer encoder only
**Masking:** No masking (bidirectional attention)
**Best For:** Classification, understanding, tagging

## 🏗️ Architecture

```
[CLS] token + Input Tokens
        ↓
Positional Encoding
        ↓
Transformer Encoder Layers × N
  ├─ Multi-head Self-Attention
  ├─ Feed-Forward
  └─ Layer Normalization & Residuals
        ↓
[CLS] Token Representation
        ↓
Classification Head
        ↓
Output Logits
```

## 🎯 Key Features

### [CLS] Token
- Special token prepended to sequence
- Final [CLS] embedding used for classification
- Learns to aggregate sequence information
- From BERT paper: "The first token of every sequence is [CLS]"

### Bidirectional Attention
- Can attend to both left and right context
- Process entire sequence at once
- No autoregressive constraint
- Perfect for understanding

## 💡 BERT vs GPT

| Aspect | BERT (Encoder) | GPT (Decoder) |
|--------|---|---|
| Masking | None | Causal |
| Training | Masked LM | Next token prediction |
| Use | Classification, understanding | Generation, completion |
| Context | Bidirectional | Left only |
| Speed | Parallel all layers | Sequential generation |

## 🚀 Quick Start

```python
from train_pytorch import TransformerEncoder

# Create model
model = TransformerEncoder(
    vocab_size=100,
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=1024,
    num_classes=3  # Classification
)

# Forward pass
logits, _ = model(input_ids)
# logits: [batch, num_classes]
```

## 📊 Applications

| Task | Input | Output |
|------|-------|--------|
| Text Classification | Text | Class label |
| Sentiment Analysis | Review | Positive/Negative/Neutral |
| Intent Detection | User query | Intent class |
| Topic Classification | Document | Topic |
| Similarity | Sentence pair | Similar/Not similar |

## 📈 Performance Notes

| Metric | Value |
|--------|-------|
| Typical Accuracy | 85-95% |
| Training Time | Fast (parallel) |
| Inference | Very fast |
| Parameters | Medium |

## 🎓 Learning Outcomes

- [x] Encoder-only architecture
- [x] [CLS] token usage
- [x] Bidirectional attention
- [x] Classification head
- [x] Transfer learning with BERT-style models

## 📚 Key Papers

- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **Attention**: "Attention Is All You Need" (Vaswani et al., 2017)

---

**Last Updated:** December 2024
**Status:** ✅ Complete
