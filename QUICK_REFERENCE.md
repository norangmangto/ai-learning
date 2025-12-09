# Quick Reference - All Model Implementations

## File Structure Overview

```
models/
├── basics/                          # Classical ML with multiple frameworks
│   ├── linear_regression/
│   │   ├── train_pytorch.py        # PyTorch custom loop
│   │   ├── train_sklearn.py        # Scikit-Learn native
│   │   ├── train_jax.py            # JAX with functional approach
│   │   └── train_tensorflow_v2.py  # TensorFlow/Keras
│   ├── logistic_regression/
│   │   ├── train_pytorch.py
│   │   ├── train_pytorch_v2.py     # Enhanced with normalization
│   │   ├── train_sklearn.py
│   │   ├── train_jax.py
│   │   └── train_tensorflow_v2.py
│   ├── mlp/
│   │   ├── train_pytorch.py
│   │   ├── train_pytorch_v2.py     # BatchNorm + Dropout
│   │   └── train_sklearn.py
│   ├── random_forest/
│   │   ├── train_sklearn.py
│   │   └── train_pytorch.py        # Neural ensemble approximation
│   ├── svm/
│   │   ├── train_sklearn.py
│   │   └── train_pytorch.py        # RBF kernel approximation
│   └── xgboost/
│       ├── train.py                # XGBoost native
│       └── train_pytorch.py        # Sequential boosting approx
│
├── advanced/                        # Deep Learning with alternatives
│   ├── cnn/
│   │   ├── train_pytorch.py        # ResNet
│   │   └── train_tensorflow.py     # MobileNet
│   ├── rnn/
│   │   ├── train_pytorch.py        # LSTM original
│   │   └── train_pytorch_v2.py     # LSTM v2 (text classification)
│   ├── transformer/
│   │   ├── train_pytorch.py        # BERT
│   │   └── train_tensorflow.py     # GPT-2
│   ├── gan/
│   │   ├── train_pytorch.py        # DCGAN
│   │   └── train_tensorflow.py     # Conditional GAN
│   ├── autoencoder/
│   │   ├── train_pytorch.py        # VAE
│   │   └── train_tensorflow.py     # Denoising AE
│   ├── dnn/
│   │   ├── train_pytorch.py
│   │   └── train_tensorflow.py
│   ├── image_classification/
│   │   ├── train_pytorch.py        # ResNet-50
│   │   └── train_pytorch_v2.py     # EfficientNet
│   ├── object_detection/
│   │   ├── train_pytorch.py        # Faster R-CNN
│   │   └── train_pytorch_v2.py     # YOLOv5
│   ├── sentiment_analysis/
│   │   ├── train_pytorch.py        # BERT-based
│   │   ├── train_tensorflow.py     # DistilBERT
│   │   └── train_sklearn.py        # TF-IDF + LogReg
│   ├── text_classification/
│   │   ├── train_pytorch.py        # BERT
│   │   ├── train_tensorflow.py     # RoBERTa
│   │   └── train_sklearn.py        # FastText
│   ├── generative/                 # Stable Diffusion, DALL-E
│   │   ├── text_to_image/
│   │   └── image_to_image/
│   └── video_*                     # Video classification/detection
│
├── text_summarization/
│   ├── bart/
│   │   ├── train_pytorch.py        # BART, T5, PEGASUS
│   │   └── train_tensorflow.py     # TF variants
│   └── extractive/
│       ├── train_pytorch.py        # SciBERT + TF-IDF
│       └── train_tensorflow.py     # DistilBERT + TF-IDF
│
└── langchain/
    ├── train_embeddings.py         # Hugging Face embeddings
    ├── train_embeddings_v2.py      # Sentence-Transformers
    ├── train_llm.py                # OpenAI/Hugging Face LLM
    ├── train_llm_v2.py             # Local models (Ollama, quantized)
    ├── train_retriever.py          # Vector similarity search
    └── README.md
```

## Quick Command Reference

### Run Specific Models:

**Basics - Compare all approaches:**
```bash
# Linear Regression
cd /Users/norangmangto/works/ai-learning
python models/basics/linear_regression/train_pytorch.py
python models/basics/linear_regression/train_sklearn.py
python models/basics/linear_regression/train_jax.py

# Logistic Regression
python models/basics/logistic_regression/train_pytorch.py
python models/basics/logistic_regression/train_pytorch_v2.py
python models/basics/logistic_regression/train_sklearn.py
```

**Advanced - Text Models:**
```bash
# Text Summarization (Abstractive)
python models/text_summarization/bart/train_pytorch.py

# Text Summarization (Extractive)
python models/text_summarization/extractive/train_pytorch.py

# Sentiment Analysis
python models/advanced/sentiment_analysis/train_pytorch.py
python models/advanced/sentiment_analysis/train_sklearn.py

# Text Classification
python models/advanced/text_classification/train_pytorch.py
python models/advanced/text_classification/train_sklearn.py
```

**Advanced - Vision Models:**
```bash
# Image Classification
python models/advanced/image_classification/train_pytorch.py
python models/advanced/image_classification/train_pytorch_v2.py

# Object Detection
python models/advanced/object_detection/train_pytorch.py
python models/advanced/object_detection/train_pytorch_v2.py

# RNN Text Classification
python models/advanced/rnn/train_pytorch_v2.py
```

**LangChain Models:**
```bash
# Embeddings
python models/langchain/train_embeddings.py
python models/langchain/train_embeddings_v2.py

# LLM
python models/langchain/train_llm.py
python models/langchain/train_llm_v2.py
```

---

## Models Comparison Table

### Basics Models

| Model | PyTorch | TensorFlow | Scikit-Learn | JAX | Best For |
|-------|---------|-----------|--------------|-----|----------|
| Linear Regression | ✓ | ✓ | ✓ | ✓ | Quick baseline |
| Logistic Regression | ✓ | ✓ | ✓ | ✓ | Binary classification |
| MLP | ✓ | ✓ | ✓ | - | Multi-layer networks |
| Random Forest | - | - | ✓ | - | Tabular data |
| SVM | - | - | ✓ | - | Small datasets |
| XGBoost | ✓ | - | - | - | Structured data |

### Advanced Models

| Model | PyTorch | TensorFlow | Alternatives | Best For |
|-------|---------|-----------|--------------|----------|
| CNN | ✓ | ✓ | EfficientNet | Image classification |
| RNN | ✓ | ✓ | LSTM v2 | Sequence modeling |
| Transformer | ✓ | ✓ | BERT, GPT | NLP tasks |
| GAN | ✓ | ✓ | StyleGAN | Image generation |
| Autoencoder | ✓ | ✓ | VAE, Sparse | Feature learning |
| Text Summarization | ✓ | ✓ | Extractive | Document summarization |
| Sentiment Analysis | ✓ | ✓ | TF-IDF | Opinion mining |
| Text Classification | ✓ | ✓ | FastText | Document categorization |
| Object Detection | ✓ | ✓ | YOLOv5 | Real-time detection |
| Image Classification | ✓ | ✓ | EfficientNet | Category prediction |

### LangChain Models

| Component | Framework | Alternatives | Best For |
|-----------|-----------|--------------|----------|
| Embeddings | Hugging Face | Sentence-Transformers | Document similarity |
| LLM | OpenAI | Local (Ollama), Quantized | Text generation |
| Retriever | Vector Search | BM25, FAISS | Information retrieval |

---

## Architecture Patterns

### Approach 1: Basic Implementation
```
Simple → PyTorch/TensorFlow/Scikit-Learn
Cost: Low
Speed: Moderate
Quality: Good for baseline
```

### Approach 2: Enhanced Implementation
```
Simple → Add regularization/optimization → Better quality
Cost: Low-Medium
Speed: Moderate
Quality: Good production
```

### Approach 3: Ensemble/Multiple Approaches
```
Model A + Model B + Model C → Combined predictions
Cost: Medium
Speed: Slower
Quality: Excellent (often best)
```

### Approach 4: Transfer Learning
```
Pre-trained model → Fine-tune on specific data
Cost: Low (pre-training done)
Speed: Fast training
Quality: Excellent (leverages pre-training)
```

---

## Framework Selection Guide

### Choose PyTorch if:
- ✓ Doing research or experimentation
- ✓ Need maximum flexibility
- ✓ Custom architectures required
- ✓ Academic/research setting

### Choose TensorFlow if:
- ✓ Building production systems
- ✓ Need deployment at scale
- ✓ Mobile/edge deployment
- ✓ Team already using TensorFlow

### Choose Scikit-Learn if:
- ✓ Classical ML (RF, SVM, XGBoost)
- ✓ Quick prototyping needed
- ✓ Tabular/structured data
- ✓ Interpretability important

### Choose JAX if:
- ✓ Numerical computing research
- ✓ Complex derivatives needed
- ✓ Performance critical
- ✓ Comfortable with functional programming

---

## Performance Expectations

### Training Time (Approximate)
- Linear Regression: < 1 second
- Logistic Regression: 1-5 seconds
- MLP: 5-30 seconds
- CNN: 1-5 minutes
- RNN: 5-15 minutes
- Transformer Fine-tuning: 30 minutes - 2 hours

### Accuracy (on standard datasets)
- Random Forest: 80-90%
- Linear Models: 70-85%
- Neural Networks: 85-95%
- Transformers: 90-98%
- Ensembles: 92-99%

---

## Dependencies by Framework

### PyTorch Stack
```
torch>=1.9.0
torchvision>=0.10.0
pytorch-lightning (optional)
```

### TensorFlow Stack
```
tensorflow>=2.8.0
keras (included in TF 2.x)
tf-hub (optional)
```

### Scikit-Learn Stack
```
scikit-learn>=1.0.0
numpy>=1.20.0
scipy>=1.7.0
```

### JAX Stack
```
jax>=0.3.0
jaxlib>=0.3.0
optax (optimizers)
```

### NLP Stack
```
transformers>=4.20.0
datasets>=2.0.0
tokenizers>=0.12.0
sentence-transformers>=2.2.0
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solutions:**
1. Reduce batch size: `--batch_size 16` → `--batch_size 8`
2. Use smaller model: EfficientNet instead of ResNet
3. Enable mixed precision: PyTorch `torch.cuda.amp`
4. Use CPU: Remove CUDA, slower but works

### Issue: "Low accuracy"
**Solutions:**
1. Increase epochs: 10 → 50
2. Better hyperparameters: learning rate, regularization
3. Use better model: Transformer > CNN > Classical ML
4. More data: Collect or augment training data

### Issue: "Module not found"
**Solutions:**
1. Install missing package: `pip install package_name`
2. Check Python version: Python 3.8+
3. Virtual environment: Isolate dependencies

---

## Next Steps

1. **Run baseline models** on your data
2. **Compare different approaches** for your use case
3. **Choose best framework** based on performance and resources
4. **Fine-tune hyperparameters** for better results
5. **Deploy selected model** to production

---

## Additional Resources

- Full implementation details: See `IMPLEMENTATION_SUMMARY.md`
- Framework docs:
  - PyTorch: https://pytorch.org/docs
  - TensorFlow: https://tensorflow.org
  - Scikit-Learn: https://scikit-learn.org
  - JAX: https://jax.readthedocs.io

---

**Created**: December 2025
**Updated**: Latest
**Total Implementations**: 50+
**Frameworks**: 6
**Models**: 20+
