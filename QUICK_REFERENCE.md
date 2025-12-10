# Quick Reference - Model Navigation Guide

## 📂 Current Repository Structure

```
models/
├── 1_supervised_learning/           # Classification, Regression, Ensembles
│   ├── classification/
│   ├── regression/
│   └── ensemble_methods/
│
├── 2_nlp_models/                    # NLP implementations (12+ files)
│   ├── embeddings/
│   │   ├── word_embeddings/        # Word2Vec, GloVe, FastText
│   │   ├── sentence_embeddings/    # Sentence-BERT, SimCSE
│   │   └── document_embeddings/
│   ├── text_classification/
│   ├── text_summarization/
│   │   ├── abstractive/
│   │   └── extractive/
│   ├── sentiment_analysis/
│   ├── retrieval_systems/
│   │   ├── semantic_search/
│   │   ├── vector_search/
│   │   └── hybrid_search/
│   └── language_models/
│       ├── gpt_variants/
│       ├── llama_variants/
│       └── encoder_models/
│
├── 3_computer_vision/               # Vision models (8+ files)
│   ├── classification/
│   │   ├── single_label/         # ResNet, EfficientNet, ViT
│   │   └── multi_label/          # Multi-label classification
│   ├── object_detection/           # YOLO, Faster R-CNN
│   ├── semantic_segmentation/      # U-Net, FCN, DeepLabV3
│   ├── instance_segmentation/
│   ├── image_to_image/
│   └── video_analysis/
│
├── 4_sequence_models/               # RNNs and Transformers (10+ files)
│   ├── rnn/
│   │   ├── lstm/                   # LSTM architecture
│   │   ├── gru/                    # GRU architecture
│   │   └── bidirectional/          # Bidirectional RNNs
│   ├── attention_mechanisms/       # 4 attention types
│   └── transformer/
│       ├── encoder_only/           # BERT-style
│       ├── decoder_only/           # GPT-style
│       ├── encoder_decoder/        # Seq2Seq
│       └── vision_transformer/     # ViT
│
├── 5_generative_models/             # GANs and Diffusion
│   ├── gan/                        # Generative Adversarial Networks
│   ├── diffusion_models/
│   │   └── text_to_image/
│   ├── autoencoder/
│   ├── flow_models/
│   └── text_generation/
│
├── 6_unsupervised_learning/         # Clustering & Dimensionality (10+ files)
│   ├── clustering/
│   │   ├── kmeans/                 # K-Means clustering
│   │   ├── hierarchical/           # Agglomerative clustering
│   │   ├── gmm/                    # Gaussian Mixture Models
│   │   └── dbscan/                 # Density-based clustering
│   ├── dimensionality_reduction/
│   │   ├── pca/                    # Principal Component Analysis
│   │   ├── tsne/                   # t-SNE visualization
│   │   └── umap/                   # UMAP projection
│   └── anomaly_detection/
│
└── 7_multimodal_learning/           # Vision + Language (3+ files)
    ├── text_image_matching/        # CLIP-style models
    ├── vision_language/            # VQA, image captioning
    └── audio_visual/
        └── speech_to_text/
            └── whisper/            # Speech recognition
```

---

## 🚀 Quick Commands

### Supervised Learning
```bash
# Classification
python models/1_supervised_learning/classification/train_pytorch.py
python models/1_supervised_learning/classification/train_sklearn.py

# Ensemble methods
python models/1_supervised_learning/ensemble_methods/train_sklearn.py
python models/1_supervised_learning/ensemble_methods/train_pytorch.py
```

### NLP Models
```bash
# Text classification
python models/2_nlp_models/text_classification/train_pytorch.py

# Text summarization
python models/2_nlp_models/text_summarization/abstractive/train_pytorch.py
python models/2_nlp_models/text_summarization/extractive/train_pytorch.py

# Sentiment analysis
python models/2_nlp_models/sentiment_analysis/train_pytorch.py

# Semantic search (example notebooks or scripts)
# See README in models/2_nlp_models/retrieval_systems/semantic_search/
```

### Computer Vision
```bash
# Image classification
python models/3_computer_vision/classification/single_label/train_pytorch.py

# Object detection (check available implementations)
# See README in models/3_computer_vision/object_detection/

# Semantic segmentation
# See README in models/3_computer_vision/semantic_segmentation/
```

### Sequence Models
```bash
# RNN examples
python models/4_sequence_models/rnn/train_pytorch.py

# Transformer examples (check subdirectories)
# Each transformer variant has dedicated README with examples
```

### Unsupervised Learning
```bash
# K-Means clustering
python models/6_unsupervised_learning/clustering/kmeans/train_sklearn.py

# DBSCAN
python models/6_unsupervised_learning/clustering/dbscan/train_sklearn.py

# PCA
python models/6_unsupervised_learning/dimensionality_reduction/pca/train_sklearn.py
```

### Generative Models
```bash
# GAN training
python models/5_generative_models/gan/dcgan/train_pytorch.py

# Autoencoder
python models/5_generative_models/autoencoder/standard_ae/train_pytorch.py
```

---

## 📖 Documentation Quick Links

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
