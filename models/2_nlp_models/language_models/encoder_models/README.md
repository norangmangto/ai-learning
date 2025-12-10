# Encoder-Only Transformer Models (BERT-style)

This directory contains implementations of encoder-only transformer models for natural language understanding tasks.

## Overview

Encoder-only models use bidirectional attention to understand text context from both directions. Unlike decoder models (GPT), they cannot generate text autoregressively but excel at understanding and classification tasks.

## Models Implemented

### BERT (Bidirectional Encoder Representations from Transformers)
- **Parameters**: 110M (base), 340M (large)
- **Pre-training**: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- **Strengths**: Strong baseline, widely supported
- **Use Cases**: General NLP, fine-tuning for downstream tasks

### RoBERTa (Robustly Optimized BERT)
- **Parameters**: 125M (base), 355M (large)
- **Pre-training**: MLM only (no NSP), dynamic masking, more data
- **Strengths**: Better performance than BERT
- **Use Cases**: When accuracy is critical

### ALBERT (A Lite BERT)
- **Parameters**: 12M (base), 18M (large)
- **Innovation**: Parameter sharing across layers
- **Pre-training**: MLM + Sentence Order Prediction (SOP)
- **Strengths**: Much smaller model size, efficient
- **Use Cases**: Resource-constrained environments

### DistilBERT
- **Parameters**: 66M (40% smaller than BERT)
- **Innovation**: Knowledge distillation from BERT
- **Strengths**: 60% faster, retains 97% of BERT's performance
- **Use Cases**: Fast inference, production deployment

## Key Concepts

### Bidirectional Attention
```
Input: "The cat sits on the mat"
       ↓
[CLS] The cat sits on the mat [SEP]
       ↓
Each token attends to ALL other tokens (bidirectional)
```

### Pre-training Tasks

1. **Masked Language Modeling (MLM)**
   - Mask 15% of tokens
   - Predict masked tokens from context
   - Example: "The [MASK] sits on the mat" → predict "cat"

2. **Next Sentence Prediction (NSP)** [BERT only]
   - Given two sentences A and B
   - Predict if B actually follows A

## Usage

```bash
# Train BERT classifier on IMDB dataset
python train_pytorch.py

# Use different model variants
python train_pytorch.py --model_type roberta
python train_pytorch.py --model_type albert
python train_pytorch.py --model_type distilbert
```

## Configuration

Edit `CONFIG` in `train_pytorch.py`:

```python
CONFIG = {
    'model_type': 'bert',  # 'bert', 'roberta', 'albert', 'distilbert'
    'model_name': 'bert-base-uncased',
    'task': 'classification',  # 'mlm', 'classification'
    'dataset': 'imdb',
    'max_length': 512,
    'batch_size': 16,
    'epochs': 3,
    'learning_rate': 2e-5,
}
```

## Common Applications

- **Text Classification**: Sentiment analysis, topic classification
- **Named Entity Recognition (NER)**: Extract entities from text
- **Question Answering**: Answer questions based on context
- **Sentence Similarity**: Determine semantic similarity
- **Information Extraction**: Extract structured data from text
- **Semantic Search**: Find semantically similar documents

## Best Practices

1. **Use Pre-trained Models**: Don't train from scratch unless necessary
2. **Small Learning Rate**: Use 2e-5 for fine-tuning (smaller than training from scratch)
3. **Warmup Steps**: Add warmup for training stability
4. **Gradient Clipping**: Prevent exploding gradients
5. **Monitor Validation**: Watch for overfitting
6. **Task-Specific Heads**: Add appropriate layers for your task

## Architecture Comparison

| Feature | BERT | RoBERTa | ALBERT | DistilBERT |
|---------|------|---------|--------|------------|
| Size | Large | Large | Small | Medium |
| Speed | Medium | Medium | Fast | Very Fast |
| Accuracy | Good | Better | Good | Good |
| Training | NSP+MLM | MLM only | SOP+MLM | Distilled |
| Best For | General | Accuracy | Efficiency | Production |

## Differences from Decoder Models (GPT)

| Aspect | Encoder (BERT) | Decoder (GPT) |
|--------|----------------|---------------|
| Attention | Bidirectional | Causal (left-to-right) |
| Task | Understanding | Generation |
| Use Case | Classification, NER | Text generation |
| Pre-training | MLM | Next token prediction |

## Model Outputs

- **[CLS] token**: Used for classification tasks
- **Token representations**: Used for token-level tasks (NER)
- **Pooled output**: Aggregate representation for sentence-level tasks

## Requirements

```bash
pip install transformers torch datasets
```

## References

- BERT: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- RoBERTa: [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- ALBERT: [Lan et al., 2019](https://arxiv.org/abs/1909.11942)
- DistilBERT: [Sanh et al., 2019](https://arxiv.org/abs/1910.01108)
