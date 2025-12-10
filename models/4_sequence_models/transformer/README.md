# Transformers: Complete Architecture Comparison

Transformers are the foundation of modern deep learning. This module covers all four main transformer architectures for different tasks.

## 📋 Overview

Four transformer variants for different use cases:

| Type | Architecture | Best For | Key Feature |
|------|--------------|----------|------------|
| **Encoder-Only** | Bidirectional self-attention | Understanding | BERT-style |
| **Decoder-Only** | Causal self-attention | Generation | GPT-style |
| **Encoder-Decoder** | Bidirectional + Causal + Cross-attention | Translation | Seq2Seq |
| **Vision** | Patch embeddings + Transformer | Image Classification | ViT-style |

## 🏗️ Shared Components

All transformers use common building blocks:

### 1. Positional Encoding

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Purpose:**
- Injects position information (transformers are permutation-invariant)
- Sine/cosine ensures smooth interpolation
- Absolute position encoding (alternative: relative position biases)

### 2. Multi-Head Self-Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Key Properties:**
- Parallel attention heads with different subspaces
- Scaled by √d_k for numerical stability
- Query-Key interaction determines attention weights

### 3. Position-wise Feedforward

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**Design:**
- Two linear layers with ReLU activation
- Expands to d_ff (usually 4×d_model) then contracts
- Applied identically to all positions

### 4. Layer Normalization & Residual Connections

```
output = LayerNorm(input + Sublayer(input))
```

**Benefits:**
- Stable gradients during backpropagation
- Enables very deep networks (12-100+ layers)
- Pre-norm vs post-norm variants available

---

## 1️⃣ Encoder-Only Transformer (BERT-style)

### Architecture

```
Input Tokens → Embedding + Positional Encoding
    ↓
[Self-Attention → Feed-Forward] × 4 layers (BIDIRECTIONAL)
    ↓
Output Embeddings
    ↓
[CLS] Token → Classification Head
```

### Key Features

**Bidirectional Attention:**
- Can attend to both previous AND future tokens
- Process entire sequence at once
- Enables deep contextual understanding

**[CLS] Token:**
- Special token prepended to sequence
- Final [CLS] representation used for classification
- Aggregates sequence information

### Use Cases

```
Text Classification
    ↓ Input: "This movie is great!" → Output: Positive

Sentiment Analysis
    ↓ Input: Review text → Output: Star rating

Named Entity Recognition
    ↓ Input: "John works at Google" → Output: Person, Company

Question Answering
    ↓ Input: Document + Question → Output: Answer span
```

### Training

```python
model = TransformerEncoder(
    vocab_size=100,
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=1024
)

# Input: sequences of token IDs
# Output: classification logits

loss = CrossEntropyLoss(logits, labels)
```

### Advantages
✅ Bidirectional context
✅ Fast (parallel processing)
✅ Best for understanding tasks
✅ Strong transfer learning

### Disadvantages
❌ Can't generate efficiently (no causal masking)
❌ Less interpretable for generation
❌ Requires classification head

---

## 2️⃣ Decoder-Only Transformer (GPT-style)

### Architecture

```
Input Tokens → Embedding + Positional Encoding
    ↓
[Masked Self-Attention → Feed-Forward] × 4 layers (CAUSAL)
    ↓
Output Logits
    ↓
Next Token Prediction
```

### Key Features

**Causal Masking:**
- Position i can only attend to positions < i
- Implemented with lower triangular mask
- Enables autoregressive generation

**Autoregressive Generation:**
- Predict one token at a time
- Use previous tokens as context
- Temperature and top-k sampling for diversity

### Use Cases

```
Text Generation
    ↓ Input: "Once upon a" → Output: Continue story

Machine Translation
    ↓ Input: Source tokens → Output: Target tokens

Code Completion
    ↓ Input: Incomplete function → Output: Complete function

Dialogue Systems
    ↓ Input: Question → Output: Response
```

### Training

```python
model = GPTDecoder(
    vocab_size=100,
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=1024
)

# Language modeling loss: predict next token
loss = CrossEntropyLoss(logits[:, :-1, :], targets[:, 1:])

# Inference: autoregressive generation
generated = model.generate(start_tokens, max_length=50)
```

### Sampling Strategies

**Greedy:**
```python
next_token = logits.argmax()  # Always pick highest probability
```

**Temperature Sampling:**
```python
logits = logits / temperature
probs = softmax(logits)
next_token = multinomial(probs)
# Low temp → more confident, High temp → more random
```

**Top-K Sampling:**
```python
# Only consider top K most likely tokens
top_k_logits[logits < top_k[K-1]] = -inf
probs = softmax(top_k_logits)
next_token = multinomial(probs)
```

### Advantages
✅ Natural generation capability
✅ Simple single decoder
✅ Efficient inference
✅ Excellent for open-ended tasks

### Disadvantages
❌ Unidirectional (only left context)
❌ Slower training (token-by-token generation)
❌ Hallucination tendency

---

## 3️⃣ Encoder-Decoder Transformer (Original)

### Architecture

```
Source Tokens → Embedding + Positional Encoding
    ↓
ENCODER [Self-Attention ↔ Feed-Forward] × 3 (BIDIRECTIONAL)
    ↓
Context Vectors
    ↓
Target Tokens → Embedding + Positional Encoding
    ↓
DECODER [Masked Self-Attention ↔ Cross-Attention → Feed-Forward] × 3 (CAUSAL)
    ↓
Output Logits
```

### Key Features

**Cross-Attention:**
- Decoder attends to encoder outputs
- Separate Q (from decoder), K/V (from encoder)
- Essential for attending to input while generating

**Two Masking Strategies:**
1. **Source Mask**: No masking (bidirectional understanding)
2. **Target Mask**: Causal masking (autoregressive generation)

### Use Cases

```
Machine Translation
    ↓ EN → FR / DE / ZH

Abstractive Summarization
    ↓ Long document → Short summary

Paraphrase Generation
    ↓ Input text → Rephrase

Text Simplification
    ↓ Complex → Simple
```

### Architecture Detail

```
Decoder output at position i depends on:
- Previous decoder outputs (causal)
- All encoder outputs (cross-attention)
```

### Training

```python
model = Transformer(
    src_vocab_size=100,
    tgt_vocab_size=80,
    d_model=256,
    num_encoder_layers=3,
    num_decoder_layers=3
)

# Forward pass needs both source and target
logits, _, _ = model(src_sequences, tgt_sequences)

# Loss on target predictions
loss = CrossEntropyLoss(logits.view(-1, vocab_size),
                        targets.view(-1))
```

### Inference

```python
# Encoder: process input once
encoder_output = model.encode(source)

# Decoder: generate target token-by-token
for t in range(max_length):
    logits = model.decode(current_target, encoder_output)
    next_token = logits[:, -1, :].argmax()
    current_target = append(next_token)
    if next_token == EOS_TOKEN:
        break
```

### Advantages
✅ Combines best of encoder and decoder
✅ Strong for sequence-to-sequence
✅ Interpretable cross-attention
✅ Flexible input/output lengths

### Disadvantages
❌ More complex architecture
❌ Slower inference (no caching initially)
❌ More parameters

---

## 4️⃣ Vision Transformer (ViT)

### Architecture

```
Image (64×64) → Patch Embedding (8×8 patches = 64 tokens)
    ↓
[CLS] Token + Positional Embedding
    ↓
[Self-Attention → Feed-Forward] × 6 layers
    ↓
[CLS] Embedding
    ↓
Classification Head → Class Logits
```

### Key Features

**Patch Embedding:**
- Divide image into non-overlapping patches (e.g., 8×8 from 64×64)
- Flatten patch pixels and project to embedding dimension
- Equivalent to conv with kernel_size = patch_size

**Learnable Positional Embeddings:**
- Unlike sinusoidal (text), learn position embeddings
- Crucial for spatial information in images
- Can be interpolated for different input sizes

### Image Patches

```
Original: 64×64 RGB image
    ↓
Divide into 8×8 = 64 patches
Each patch: 8×8×3 = 192 values
    ↓
Linear projection: 192 → 256
    ↓
Sequence of 64 tokens (like words in NLP)
```

### Use Cases

```
Image Classification
    ↓ Input: Image → Output: Class label

Medical Image Analysis
    ↓ X-ray/CT → Disease detection

Scene Understanding
    ↓ Image → Objects and relationships

Fine-grained Recognition
    ↓ Bird species / Car model identification
```

### Training

```python
model = VisionTransformer(
    img_size=64,
    patch_size=8,
    in_channels=3,
    num_classes=10,
    embed_dim=256,
    depth=6,
    num_heads=8
)

# Input: images [batch, 3, 64, 64]
# Output: logits [batch, num_classes]

logits = model(images)
loss = CrossEntropyLoss(logits, labels)
```

### Advantages
✅ No convolutional inductive bias (learns from data)
✅ Scales well with large datasets
✅ Strong transfer learning
✅ Interpretable attention patterns

### Disadvantages
❌ Requires large datasets (ImageNet)
❌ Slower training than CNNs
❌ Position interpolation needed for size changes

---

## 📊 Comparison Table

| Aspect | Encoder-Only | Decoder-Only | Encoder-Decoder | Vision |
|--------|--------|----------|------------|--------|
| **Masking** | None | Causal | Causal (decoder) | None |
| **Attention** | Self | Self | Self + Cross | Self |
| **Generation** | ❌ | ✅ | ✅ | ❌ |
| **Understanding** | ✅ | ⚠️ | ✅ | ✅ |
| **Inference** | Fast | Medium | Slow | Fast |
| **Training** | Fast | Medium | Slow | Medium |
| **Parameters** | Fewer | Medium | More | Medium |
| **Use Cases** | Classification | Generation | Translation | Vision |

## 🔑 Key Insights

### 1. Attention Is Everything
- No convolutions, no recurrence
- Pure attention-based architecture
- Each layer allows global information flow

### 2. Position Matters
- Sinusoidal encoding (text): works across sequence lengths
- Learned encoding (vision): data-dependent positions

### 3. Masking Controls Information Flow
- No mask: Bidirectional (encoder)
- Causal mask: Unidirectional (decoder)
- Cross-mask: Between encoder-decoder

### 4. Scaling Laws
- More data → Better performance
- More parameters → Better with large data
- Larger models transfer better

## 🚀 Implementation Tips

### General

```python
# Layer normalization placement
Pre-norm:  norm → attention/ffn → add
Post-norm: attention/ffn → norm → add
# Pre-norm typically trains faster
```

### Attention

```python
# Scaled dot-product
scores = QK^T / sqrt(d_k)
# Prevents attention weights from being too small
```

### Generation

```python
# Key concept: token-by-token
prev_tokens = [start_token]
for _ in range(max_length):
    logits = model(prev_tokens)
    next_token = sample(logits[-1])
    prev_tokens.append(next_token)
    if next_token == eos_token: break
```

## 💡 Hyperparameter Tuning

| Parameter | Range | Notes |
|-----------|-------|-------|
| d_model | 256-1024 | Larger = more expressive |
| num_heads | 4-16 | Should divide d_model |
| num_layers | 2-12 | More = more complex |
| d_ff | 4×d_model | Usually 1024-4096 |
| dropout | 0.1-0.3 | Regularization |

## 📚 Resources

### Key Papers
- **Original Transformer**: "Attention Is All You Need" (Vaswani et al., 2017)
- **BERT**: "Bidirectional Encoder Representations from Transformers" (Devlin et al., 2018)
- **GPT**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- **ViT**: "An Image is Worth 16×16 Words" (Dosovitskiy et al., 2020)

### Concepts
- Query: What am I looking for?
- Key: What information do I have?
- Value: What information should I aggregate?

## ⚠️ Common Pitfalls

1. **Forgetting Positional Encoding**: Transformers are permutation-invariant
2. **Wrong Masking**: Critical for correct behavior
3. **Large Batch Sizes**: Needed for stable contrastive-style training
4. **Gradient Clipping**: Important for stability
5. **Learning Rate Warmup**: Essential for convergence

## 🧪 Testing Your Implementation

```python
# Test 1: Check attention mask shapes
logits, weights = model(input)
assert weights.shape == (batch, heads, seq, seq)

# Test 2: Verify causality in decoder
# logits[t] should not depend on input[t+1]

# Test 3: Check cross-attention in encoder-decoder
# decoder should attend to encoder correctly

# Test 4: Verify generation produces valid sequences
generated = model.generate(start)
assert generated.shape[1] <= max_length
```

---

**Last Updated:** December 2024
**Status:** ✅ Complete with 4 architecture implementations
