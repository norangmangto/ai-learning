# Vision-Language Models: VQA & Image Captioning

Vision-language models bridge visual perception and natural language understanding for tasks requiring joint reasoning over images and text.

## 📋 Overview

This module implements:
- **Visual Question Answering (VQA)**: Answer questions about image content
- **Image Captioning**: Generate natural language descriptions of images
- **Cross-modal Attention**: Learn connections between visual regions and language

## 🏗️ Architecture

### Visual Question Answering (VQA)

```
Image → CNN Feature Extractor (16 spatial regions)
          ↓
Question → LSTM Text Encoder
          ↓
[Cross-Modal Attention] ← Attends to relevant image regions
          ↓
Fusion Network → Answer Classification
```

**Components:**
- **Image Encoder**: CNN with spatial feature maps (16 regions)
- **Question Encoder**: Bi-directional LSTM
- **Attention Mechanism**: Query-conditional attention over image regions
- **Fusion Module**: Concatenate attended visual features + question features
- **Classifier**: 3-layer FC network for answer prediction

### Image Captioning

```
Image → CNN Feature Extractor (16 spatial regions)
          ↓
        LSTM Decoder with Attention
          ↓
[Cross-Modal Attention] ← Generate captions token-by-token
          ↓
Vocabulary Projection → Caption tokens
```

**Components:**
- **Image Encoder**: Spatial CNN features
- **Word Embedding**: Learnable embeddings for vocabulary
- **LSTM Decoder**: Generates captions autoregressively
- **Attention Module**: Focuses on relevant image regions at each step
- **Generation**: Start token → end token with greedy/sampling strategies

## 📊 Task Details

### VQA Task

**Input:**
- Image: RGB image (64×64)
- Question: "What pattern?" (encoded as token sequence)

**Output:**
- Answer: Class index (0=vertical, 1=horizontal, 2=checkerboard)

**Loss Function:** Cross-entropy loss

**Metrics:**
- Accuracy: Exact match percentage
- Per-class accuracy

### Image Captioning Task

**Input:**
- Image: RGB image (64×64)

**Output:**
- Caption: Sequence of tokens describing the image

**Loss Function:** Cross-entropy with padding ignored

**Metrics:**
- BLEU Score (approximate)
- Perplexity
- Vocabulary coverage

## 🚀 Usage

### Training VQA

```python
from train_vqa_captioning import VQAModel, generate_vqa_data, train_vqa

# Generate or load data
images, questions, answers = generate_vqa_data(n_samples=1500)

# Create model
model = VQAModel(
    vocab_size=10,
    num_answers=3,
    embed_dim=256,
    hidden_dim=512
)

# Train
history = train_vqa(model, train_loader, val_loader, epochs=40)
```

### Training Image Captioning

```python
from train_vqa_captioning import ImageCaptioningModel, generate_captioning_data, train_captioning

# Generate or load data
images, captions = generate_captioning_data(n_samples=1500)

# Create model
model = ImageCaptioningModel(
    vocab_size=10,
    embed_dim=256,
    hidden_dim=512
)

# Train
history = train_captioning(model, train_loader, val_loader, epochs=40)
```

### Caption Generation

```python
# Load trained model
model.load_state_dict(torch.load('best_captioning.pth'))
model.eval()

# Generate captions
generated_captions = model.generate(
    images,
    start_token=1,
    end_token=2,
    max_length=20
)
```

## 🔑 Key Features

### VQA
- **Query-Conditional Attention**: Attention weights depend on question context
- **Spatial Reasoning**: Attends to specific image regions
- **End-to-end Training**: Joint optimization of all components
- **Interpretability**: Visualization of attention maps

### Image Captioning
- **Autoregressive Generation**: Generates one token at a time
- **Spatial Attention**: Different regions attended at each step
- **Teacher Forcing**: During training, uses ground truth previous tokens
- **Flexible Generation**: Greedy or sampling-based strategies

## 📈 Performance Considerations

### VQA Model
| Aspect | Details |
|--------|---------|
| Parameters | ~1.5M |
| Training Time | 30-40 minutes (CPU), 2-5 minutes (GPU) |
| Typical Accuracy | 85-95% (synthetic data) |
| Inference Time | ~10ms per sample |

### Captioning Model
| Aspect | Details |
|--------|---------|
| Parameters | ~1.2M |
| Training Time | 40-50 minutes (CPU), 3-6 minutes (GPU) |
| Typical Perplexity | 1.2-1.5 |
| Generation Time | ~50ms per caption |

## 🎯 Architecture Decisions

### Why Attention?
- Enables model to focus on relevant image regions
- Provides interpretability through attention weights
- Improves performance on complex scenes
- Mimics human visual attention patterns

### Cross-modal Fusion Strategies
1. **Early Fusion**: Concatenate before processing ❌ (loses modality-specific info)
2. **Late Fusion**: Separate processing + combine (✓ current approach)
3. **Attention-based Fusion**: ✓ (used in our implementation)

### Question Encoding Choices
- **LSTM**: Captures sequential dependencies ✓
- **Transformer**: Better for longer questions (alternative)
- **CNN**: Treats text as image (less suitable)

## 🔄 Training Pipeline

### Data Preparation
```python
# Synthetic data generation
- Images: Patterned images (vertical/horizontal/checkerboard)
- Questions: Token sequences
- Captions: Description sequences
- Vocabulary: Fixed set of tokens
```

### Training Loop
1. **Forward Pass**: Image + text → predictions
2. **Loss Computation**: Compare with ground truth
3. **Backpropagation**: Update all parameters
4. **Attention Computation**: Track attention maps

### Validation
- Check loss on held-out validation set
- Save best model based on validation metrics
- Monitor per-class performance

## 💡 Hyperparameter Tuning Guide

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| embed_dim | 256-512 | Larger for more complex scenes |
| hidden_dim | 512-1024 | LSTM hidden state size |
| batch_size | 32-64 | Larger for stable gradients |
| learning_rate | 0.0001-0.001 | Use decay schedule |
| num_layers (LSTM) | 1-2 | More layers for complexity |

## 🎓 Learning Outcomes

After implementing this module, you understand:
- [x] Cross-modal attention mechanisms
- [x] Joint vision-language processing
- [x] Sequence-to-sequence with visual context
- [x] Autoregressive generation
- [x] Attention visualization and interpretation
- [x] Multi-task learning (VQA + Captioning)

## 🔗 Related Models

**Vision-Only:** [CNN Models](../../3_computer_vision/image_classification)

**Language-Only:** [Language Models](../../2_nlp_models/language_models)

**Attention:** [Attention Mechanisms](../../4_sequence_models/attention_mechanisms)

**Transformers:** [Vision Transformer](../../4_sequence_models/transformer/vision_transformer)

## 📚 Resources

### Key Papers
- **VQA**: "VQA: Visual Question Answering" (Antol et al., 2015)
- **Attention**: "Show, Attend and Tell" (Xu et al., 2015)
- **Captioning**: "Knowing When to Look" (Lu et al., 2017)

### Implementation Details
- Attention computation: Softmax over image regions
- Fusion: Concatenation + FC layers
- Generation: Argmax (greedy) or sampling

## ⚠️ Limitations & Future Work

### Current Limitations
- Fixed vocabulary (no OOV handling)
- Simple synthetic data patterns
- Single attention head (no multi-head)
- No beam search decoding

### Future Improvements
- [ ] Multi-head attention
- [ ] Beam search for caption generation
- [ ] Word embeddings (Word2Vec/GloVe)
- [ ] Faster RCNN for better region features
- [ ] Transformer-based decoders
- [ ] Multi-task learning framework
- [ ] Real image datasets (COCO, Flickr)

## 🧪 Evaluation Metrics

### VQA
```python
# Accuracy
accuracy = (predictions == answers).mean()

# Per-class accuracy
for class_id in range(num_classes):
    mask = answers == class_id
    class_acc = (predictions[mask] == answers[mask]).mean()
```

### Captioning
```python
# Perplexity
perplexity = exp(mean(loss))

# BLEU score (approximated)
from nltk.translate.bleu_score import corpus_bleu
bleu = corpus_bleu(references, hypotheses)
```

## 📝 Notes

- Attention weights sum to 1 across all image regions
- Temperature in softmax can be adjusted for sharper/softer attention
- Gradient clipping prevents exploding gradients in LSTM
- Layer normalization not included (consider adding)

---

**Last Updated:** December 2024
**Status:** ✅ Complete with training examples and documentation
