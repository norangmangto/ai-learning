# Encoder-Decoder Transformer (Seq2Seq)

Original transformer architecture for sequence-to-sequence tasks like translation.

## 📋 Overview

**Architecture:** Encoder + Decoder with cross-attention
**Masking:** Encoder (none), Decoder (causal)
**Best For:** Translation, summarization, seq2seq

## 🏗️ Architecture

```
Source Language (English):
[e_1 e_2 e_3 e_4] → Encoder → [c_1 c_2 c_3 c_4]
                     (bidirectional)

Target Language (French):
Decoder
[<START>] → Cross-Attention to encoder → f_1
[<START> f_1] → Cross-Attention to encoder → f_2
...
```

## 🎯 Key Features

### Encoder
- Processes entire source sequence
- Bidirectional self-attention
- Outputs context vectors for each position

### Decoder
- Generates target token-by-token
- Causal self-attention (can't see future)
- Cross-attention to encoder outputs
- Attends to relevant source positions

### Cross-Attention
```
Query: Decoder state (what am I generating?)
Key/Value: Encoder outputs (what source info is available?)

→ Allows decoder to focus on relevant parts of input
```

## 📊 Information Flow

```
English: "The cat sat on the mat"
          [h_1 h_2 h_3 h_4 h_5 h_6]

French: "Le"  → Attend to [The], [cat] → "chat"
        "chat" → Attend to [cat], [sat] → "était"
        "était" → Attend to [sat], [on] → "assis"
        "assis" → Attend to [on], [the], [mat] → "sur"
        ...

Cross-attention weights visualize alignment!
```

## 🚀 Quick Start

```python
from train_pytorch import Transformer

# Create model
model = Transformer(
    src_vocab_size=100,
    tgt_vocab_size=80,
    d_model=256,
    num_encoder_layers=3,
    num_decoder_layers=3,
    num_heads=8,
    d_ff=1024
)

# Training
src, tgt = batch
logits, _, _ = model(src, tgt)
loss = CrossEntropyLoss(logits.view(-1, tgt_vocab_size),
                        targets.view(-1))

# Inference: greedy decoding
def translate(source):
    encoder_output = model.encode(source)
    current = [<START>]
    while len(current) < max_len:
        logits = model.decode(current, encoder_output)
        next_token = logits[-1, :].argmax()
        current.append(next_token)
        if next_token == <END>:
            break
    return current
```

## 📈 Applications

| Task | Example |
|------|---------|
| **Machine Translation** | English → French |
| **Summarization** | Long text → Short summary |
| **Paraphrase** | Text → Rephrase |
| **Simplification** | Complex → Simple |
| **Code Generation** | Natural language → Code |

## 💡 Encoder-Decoder Intuition

```
ENCODER: "What's in the input?"
         ↓ Process and understand
         [Context vectors for each position]

DECODER: "What should I generate next?"
         ↓ Cross-attention to encoder
         [Focus on relevant input parts]
         ↓
         [Generate one token at a time]
```

## 🔄 Training vs Inference

### Training (Teacher Forcing)
```python
# Use ground truth targets as decoder input
decoder_input = target_tokens[:-1]  # All but last
decoder_output = model.decode(decoder_input, encoder_output)
loss = criterion(decoder_output, target_tokens[1:])  # All but first
```

**Advantage:** Faster training (parallel)
**Disadvantage:** Exposure bias (model trained on gold, tests on own outputs)

### Inference (Autoregressive)
```python
# Use model's own outputs as next input
output = [<START>]
while output[-1] != <END>:
    decoder_output = model.decode(output, encoder_output)
    next_token = decoder_output[-1, :].argmax()
    output.append(next_token)
return output
```

**Advantage:** Real test scenario
**Disadvantage:** Slower (sequential)

## ⚠️ Common Issues

1. **Exposure Bias**: Train with gold, test with predictions
   - Solution: Scheduled sampling
2. **Length Mismatch**: Model over/under-generates
   - Solution: Length penalty in decoding
3. **Repetition**: Generates same tokens repeatedly
   - Solution: Coverage mechanism
4. **Slow Inference**: Greedy decoding is slow
   - Solution: Beam search, caching

## 🎓 Learning Outcomes

- [x] Encoder-decoder architecture
- [x] Cross-attention mechanism
- [x] Teacher forcing training
- [x] Autoregressive decoding
- [x] Translation quality metrics

## 📚 Key Papers

- **Original**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Beam Search**: "Effective Approaches to Attention" (Luong et al., 2015)
- **Coverage**: "Addressing the Rare Word Problem" (Luong et al., 2014)

## 📊 Improvements Over Simpler Seq2Seq

| Aspect | Simple Seq2Seq | Transformer Seq2Seq |
|--------|---|---|
| Parallelization | Limited | Full ✅ |
| Long Sequences | Poor | Excellent ✅ |
| Interpretability | Limited | Good (attention) ✅ |
| Speed | Slow | Fast ✅ |
| Accuracy | Medium | High ✅ |

---

**Last Updated:** December 2024
**Status:** ✅ Complete
