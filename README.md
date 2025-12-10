# AI Learning Repository

A comprehensive collection of machine learning and deep learning implementations with detailed documentation.

## 📚 What's Inside

- **86 Python implementations** across 7 major categories
- **39 comprehensive README guides** covering architectures, use cases, and best practices
- **Multiple frameworks**: PyTorch, TensorFlow, Scikit-learn, JAX, Gensim
- **Production-ready code** with examples and documentation

## 🗂️ Repository Structure

```
models/
├── 1_supervised_learning/     # Classification, regression, ensemble methods
├── 2_nlp_models/              # Text classification, summarization, embeddings, retrieval
├── 3_computer_vision/         # Image classification, object detection, segmentation
├── 4_sequence_models/         # RNNs, transformers, attention mechanisms
├── 5_generative_models/       # GANs, diffusion models, autoencoders
├── 6_unsupervised_learning/   # Clustering, dimensionality reduction, anomaly detection
└── 7_multimodal_learning/     # Vision-language, text-image matching, speech-to-text

theory/                        # ML theory notes and cheat sheets
```

## 🚀 Quick Start

### Installation
```bash
# Core dependencies
pip install torch tensorflow scikit-learn jax numpy scipy

# NLP and computer vision
pip install transformers datasets sentence-transformers torchvision

# Specific models
pip install gensim umap-learn faiss-cpu
```

### Run a Model
```bash
# Text classification
python models/2_nlp_models/text_classification/train_pytorch.py

# Image classification
python models/3_computer_vision/classification/single_label/train_pytorch.py

# Clustering
python models/6_unsupervised_learning/clustering/kmeans/train_sklearn.py
```

## 📖 Documentation

Each model category includes:
- **Architecture explanations** with diagrams
- **Mathematical foundations** with formulas
- **Quick start code examples**
- **Hyperparameter tuning guides**
- **Performance comparisons**
- **Common pitfalls and solutions**
- **Real-world applications**
- **Key research paper references**

Start with these guides:
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Quick lookup reference
- **[PROJECT_INDEX.md](./PROJECT_INDEX.md)** - Detailed project overview
- **[VISUAL_OVERVIEW.md](./VISUAL_OVERVIEW.md)** - Visual guide with diagrams

## 🎯 Learning Paths

### Beginner
1. Supervised learning basics (classification, regression)
2. Classical ML (K-means, PCA)
3. Simple neural networks (MLP, CNN)

### Intermediate
1. NLP models (text classification, sentiment analysis)
2. Computer vision (object detection, segmentation)
3. Sequence models (RNN, LSTM, GRU)

### Advanced
1. Transformers (BERT, GPT architectures)
2. Generative models (GANs, diffusion)
3. Multimodal learning (CLIP, VQA)

## 🌟 Key Features

- ✅ **Comprehensive documentation** - Every model has detailed README
- ✅ **Multiple approaches** - Compare different architectures/frameworks
- ✅ **Production-ready** - Clean, well-documented code
- ✅ **Educational** - Learning outcomes and key papers included
- ✅ **Active development** - Regular updates and improvements

## 📊 Categories Covered

| Category | Models | READMEs | Examples |
|----------|--------|---------|----------|
| Supervised Learning | 3 | 3 | Classification, Regression, Ensembles |
| NLP Models | 12+ | 8 | Embeddings, Summarization, Classification |
| Computer Vision | 8+ | 3 | Image classification, Detection, Segmentation |
| Sequence Models | 10+ | 9 | RNN, LSTM, Transformers (all variants) |
| Unsupervised | 10+ | 11 | Clustering, PCA, t-SNE, UMAP |
| Generative | 5+ | 2 | GANs, Diffusion models |
| Multimodal | 3+ | 3 | CLIP, VQA, Whisper |

## 🛠️ Technologies

- **Deep Learning**: PyTorch, TensorFlow, JAX
- **Classical ML**: Scikit-learn
- **NLP**: Transformers, Sentence-BERT, Gensim
- **Computer Vision**: Torchvision, OpenCV
- **Tools**: NumPy, SciPy, Matplotlib

## 📝 License

MIT License - feel free to use for learning and projects!

## 🤝 Contributing

Contributions welcome! Please check existing implementations and documentation style.
