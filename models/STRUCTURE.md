# ML Models Directory Structure

This document describes the new domain-based organization of the `./models` directory, which consolidates all machine learning implementations into 7 main categories for better discoverability and maintainability.

## Overview

The restructuring moves from a framework/implementation-focused structure (basics, advanced, langchain, text_summarization) to a **domain-based structure** that groups models by their primary use case and discipline.

### New Structure (Domain-Based)

```
models/
├── 1_supervised_learning/      # Classical ML and supervised deep learning
├── 2_nlp_models/               # Natural Language Processing models
├── 3_computer_vision/          # Computer vision and image models
├── 4_sequence_models/          # RNN, Transformer, and sequence-based architectures
├── 5_generative_models/        # GAN, Autoencoders, Diffusion, Flow models
├── 6_unsupervised_learning/    # Clustering, dimensionality reduction, anomaly detection
└── 7_multimodal_learning/      # Cross-modal models (vision-language, audio-visual)
```

---

## 1. Supervised Learning

**Path:** `models/1_supervised_learning/`

Contains classical machine learning and supervised deep learning models for both regression and classification tasks.

### Subcategories:

- **regression/** - Regression models
  - Linear Regression (sklearn, TensorFlow, PyTorch, JAX)
  - Polynomial Regression
  - Ridge/Lasso Regression

- **classification/** - Classification models
  - Logistic Regression (sklearn, TensorFlow, PyTorch, JAX)
  - Multi-layer Perceptron / DNN (TensorFlow, PyTorch)
  - Support Vector Machine (sklearn, PyTorch)

- **ensemble_methods/** - Ensemble learning methods
  - Random Forest (sklearn, PyTorch)
  - XGBoost (PyTorch, native)
  - Gradient Boosting variants

### Key Files:
- `1_supervised_learning/regression/train_*.py` - Regression implementations
- `1_supervised_learning/classification/train_*.py` - Classification implementations
- `1_supervised_learning/ensemble_methods/train_*.py` - Ensemble implementations

---

## 2. NLP Models

**Path:** `models/2_nlp_models/`

Comprehensive NLP models consolidated from `advanced/`, `langchain/`, and `text_summarization/` directories.

### Subcategories:

- **sentiment_analysis/** - Sentiment classification
  - BERT-based models (PyTorch, TensorFlow)
  - Sklearn TF-IDF approach
  - DistilBERT fine-tuning

- **text_classification/** - General text classification
  - BERT variants (PyTorch, TensorFlow)
  - RoBERTa models
  - FastText approach (sklearn)

- **text_summarization/** - Text summarization models
  - `abstractive/` - Seq2Seq based (BART, T5, PEGASUS)
    - BART (PyTorch, TensorFlow)
    - T5 variants
  - `extractive/` - Extractive summarization
    - TF-IDF based extraction
    - SciBERT fine-tuning
    - DistilBERT approach

- **embeddings/** - Word and sentence embeddings
  - `word_embeddings/` - Word-level embeddings
    - Word2Vec
    - GloVe vectors
  - `sentence_embeddings/` - Sentence-level embeddings
    - Sentence-Transformers (HF)
    - BERT pooling approaches
    - LangChain embeddings
  - `document_embeddings/` - Document-level embeddings
    - Doc2Vec
    - Average pooling methods

- **language_models/** - Large language models and variants
  - `gpt_variants/` - GPT family models
    - GPT-2 (OpenAI, DistilGPT2)
    - Fine-tuning approaches
    - LangChain LLM wrappers
  - `llama_variants/` - Llama family models
    - Llama 2
    - Ollama integration
    - Mistral models
  - `encoder_models/` - Encoder-only models
    - BERT variants
    - RoBERTa
    - DistilBERT

- **retrieval_systems/** - Document retrieval and RAG
  - `vector_search/` - Vector similarity search
    - LangChain retrievers
    - FAISS integration
    - Embedding-based search
  - `hybrid_search/` - Hybrid retrieval (lexical + semantic)
    - BM25 + embeddings
    - Ensemble methods
  - `semantic_search/` - Semantic search approaches

### Key Files:
- `2_nlp_models/sentiment_analysis/train_*.py` - Sentiment models
- `2_nlp_models/text_classification/train_*.py` - Classification models
- `2_nlp_models/text_summarization/{abstractive,extractive}/train_*.py` - Summarization
- `2_nlp_models/embeddings/*/train_*.py` - Embedding models
- `2_nlp_models/language_models/*/train_*.py` - LLM implementations
- `2_nlp_models/retrieval_systems/*/train_*.py` - Retrieval models

---

## 3. Computer Vision

**Path:** `models/3_computer_vision/`

Consolidates all image and video models from `advanced/` directory with clear task-based organization.

### Subcategories:

- **classification/** - Image classification models
  - `single_label/` - Single-label classification
    - ResNet (PyTorch, TensorFlow)
    - EfficientNet
    - Vision Transformer (ViT)
    - Inception, VGG variants
  - `multi_label/` - Multi-label classification
    - Multi-label BERT
    - Multi-head architectures

- **object_detection/** - Object detection models
  - `single_stage/` - Single-stage detectors
    - YOLO family (PyTorch, TensorFlow)
    - SSD models
    - RetinaNet
  - `two_stage/` - Two-stage detectors
    - Faster R-CNN
    - Mask R-CNN
  - `anchor_free/` - Anchor-free methods
    - FCOS
    - CenterNet

- **semantic_segmentation/** - Semantic segmentation models
  - `fcn/` - Fully Convolutional Networks
  - `unet/` - U-Net architecture
  - `deeplabv3/` - DeepLabV3+ models

- **instance_segmentation/** - Instance segmentation
  - Mask R-CNN
  - YOLACT variants

- **image_to_image/** - Image transformation models
  - `super_resolution/` - Super-resolution
    - ESPCN
    - Real-ESRGAN
  - `style_transfer/` - Neural style transfer
    - AdaIN
    - Patch-based methods
  - `inpainting/` - Image inpainting/completion
  - `unet/` - UNet-based transformations
    - Image2Image with UNet

- **video_analysis/** - Video-based models
  - `video_classification/` - Video action classification
    - 3D CNN (PyTorch, TensorFlow)
    - SlowFast networks
  - `video_object_detection/` - Video object tracking
    - Temporal R-CNN
    - Tracking-by-detection

### Key Files:
- `3_computer_vision/classification/single_label/train_*.py` - Image classification
- `3_computer_vision/classification/multi_label/train_*.py` - Multi-label classification
- `3_computer_vision/object_detection/*/train_*.py` - Object detection
- `3_computer_vision/semantic_segmentation/*/train_*.py` - Segmentation
- `3_computer_vision/image_to_image/*/train_*.py` - Image transformation
- `3_computer_vision/video_analysis/*/train_*.py` - Video models

---

## 4. Sequence Models

**Path:** `models/4_sequence_models/`

Specializes in sequential and temporal architectures, including RNNs and Transformers.

### Subcategories:

- **rnn/** - Recurrent Neural Networks
  - `lstm/` - Long Short-Term Memory networks
    - Unidirectional LSTM (PyTorch, TensorFlow)
    - Stacked LSTM
    - Attention-augmented LSTM
  - `gru/` - Gated Recurrent Units
    - Simple GRU
    - Stacked GRU
  - `bidirectional/` - Bidirectional RNNs
    - BiLSTM
    - BiGRU
    - Encoder-Decoder with BiLSTM

- **transformer/** - Transformer architectures
  - `encoder_only/` - Encoder-only models
    - BERT (covered in NLP, but can have seq models)
    - RoBERTa
  - `decoder_only/` - Decoder-only models (auto-regressive)
    - GPT variants (see NLP for full coverage)
    - Decoder-only implementations
  - `encoder_decoder/` - Encoder-Decoder models
    - BART (see NLP text_summarization for full coverage)
    - T5
    - Transformer base implementation
  - `vision_transformer/` - Vision Transformers
    - ViT for image classification
    - ViT for object detection

- **attention_mechanisms/** - Attention mechanism implementations
  - Self-attention
  - Multi-head attention
  - Cross-attention
  - Sparse attention patterns

### Key Files:
- `4_sequence_models/rnn/lstm/train_*.py` - LSTM implementations
- `4_sequence_models/rnn/gru/train_*.py` - GRU implementations
- `4_sequence_models/transformer/*/train_*.py` - Transformer variants

---

## 5. Generative Models

**Path:** `models/5_generative_models/`

Consolidates all models designed for generation tasks.

### Subcategories:

- **gan/** - Generative Adversarial Networks
  - `dcgan/` - Deep Convolutional GAN
    - DCGAN implementations (PyTorch, TensorFlow)
    - Image generation
  - `conditional_gan/` - Conditional GAN
    - Class-conditional generation
    - Attribute-controlled generation
  - `stylegan/` - StyleGAN architecture
    - StyleGAN variants
    - High-quality image synthesis
  - `progressive_gan/` - Progressive GAN
    - Progressive training
    - Multi-scale generation

- **autoencoder/** - Autoencoders and variants
  - `standard_ae/` - Standard autoencoders
    - Basic AE (PyTorch, TensorFlow)
    - Reconstruction learning
  - `denoising_ae/` - Denoising autoencoders
    - Noise injection and removal
  - `sparse_ae/` - Sparse autoencoders
    - Sparsity constraints
  - `variational_ae/` - Variational autoencoders
    - VAE implementation
    - Latent space learning

- **diffusion_models/** - Diffusion and score-based models
  - `ddpm/` - Denoising Diffusion Probabilistic Models
    - Basic DDPM
    - Training and sampling
  - `stable_diffusion/` - Stable Diffusion
    - Text-to-image generation
    - Fine-tuning approaches
    - DreamBooth integration
  - `dalle/` - DALL-E variants
    - Vision-language models
  - `dreambooth_lora/` - DreamBooth with LoRA
    - Efficient fine-tuning
    - Personalization
  - `dreambooth_lora_flux/` - DreamBooth for Flux model
    - Flux-specific tuning
  - `dreambooth_lora_sd3/` - DreamBooth for SD3
    - SD3-specific implementations
  - `text_to_image/` - Text-to-image generation
    - Flux text-to-image
    - SD3 text-to-image

- **flow_models/** - Flow-based generative models
  - Normalizing flows
  - Invertible neural networks
  - Continuous normalizing flows

- **text_generation/** - Sequence generation models
  - `gpt_based/` - GPT-based text generation
    - Causal language modeling
    - Fine-tuning for generation
  - `seq2seq/` - Sequence-to-sequence generation
    - Encoder-Decoder generation
    - Attention-based decoding

### Key Files:
- `5_generative_models/gan/*/train_*.py` - GAN implementations
- `5_generative_models/autoencoder/*/train_*.py` - Autoencoder variants
- `5_generative_models/diffusion_models/*/train_*.py` - Diffusion model training
- `5_generative_models/diffusion_models/*/generate.py` - Generation scripts

---

## 6. Unsupervised Learning

**Path:** `models/6_unsupervised_learning/`

(Currently placeholder for future expansion)

### Subcategories (planned):

- **clustering/** - Clustering algorithms
  - `kmeans/` - K-Means clustering
  - `dbscan/` - Density-based clustering
  - `hierarchical/` - Hierarchical clustering

- **dimensionality_reduction/** - Dimension reduction techniques
  - `pca/` - Principal Component Analysis
  - `tsne/` - t-SNE visualization
  - `umap/` - UMAP embeddings

- **anomaly_detection/** - Anomaly detection models
  - `isolation_forest/` - Isolation Forest
  - `autoencoder_based/` - AE-based anomaly detection

---

## 7. Multimodal Learning

**Path:** `models/7_multimodal_learning/`

(Currently placeholder for future expansion)

### Subcategories (planned):

- **vision_language/** - Vision-language models
  - CLIP variants
  - Vision Transformers with language
  - Contrastive learning approaches

- **audio_visual/** - Audio-visual models
  - Speech-to-text (Whisper)
  - Audio-visual synchronization
  - Cross-modal learning

- **text_image_matching/** - Text-image matching
  - Image-text retrieval
  - Visual question answering
  - Scene graphs

### Key Files:
- `7_multimodal_learning/audio_visual/speech_to_text/whisper/transcribe.py` - Speech recognition

---

## Migration Reference

### Old Structure → New Structure Mapping

| Old Path | New Path |
|----------|----------|
| `basics/linear_regression/` | `1_supervised_learning/regression/` |
| `basics/logistic_regression/` | `1_supervised_learning/classification/` |
| `basics/mlp/` | `1_supervised_learning/classification/` |
| `basics/svm/` | `1_supervised_learning/classification/` |
| `basics/random_forest/` | `1_supervised_learning/ensemble_methods/` |
| `basics/xgboost/` | `1_supervised_learning/ensemble_methods/` |
| `advanced/sentiment_analysis/` | `2_nlp_models/sentiment_analysis/` |
| `advanced/text_classification/` | `2_nlp_models/text_classification/` |
| `advanced/image_classification/` | `3_computer_vision/classification/image_classification/` |
| `advanced/object_detection/` | `3_computer_vision/object_detection/single_stage/` |
| `advanced/cnn/` | `3_computer_vision/classification/image_classification/` |
| `advanced/image_to_image/` | `3_computer_vision/image_to_image/unet/` |
| `advanced/video_classification/` | `3_computer_vision/video_analysis/video_classification/` |
| `advanced/video_object_detection/` | `3_computer_vision/video_analysis/video_object_detection/` |
| `advanced/rnn/` | `4_sequence_models/rnn/` |
| `advanced/transformer/` | `4_sequence_models/transformer/` |
| `advanced/dnn/` | `1_supervised_learning/classification/` |
| `advanced/autoencoder/` | `5_generative_models/autoencoder/standard_ae/` |
| `advanced/gan/` | `5_generative_models/gan/dcgan/` |
| `advanced/generative/` | `5_generative_models/diffusion_models/` |
| `advanced/audio/` | `7_multimodal_learning/audio_visual/` |
| `langchain/train_embeddings.py` | `2_nlp_models/embeddings/sentence_embeddings/` |
| `langchain/train_llm.py` | `2_nlp_models/language_models/gpt_variants/` |
| `langchain/train_retriever.py` | `2_nlp_models/retrieval_systems/vector_search/` |
| `text_summarization/bart/` | `2_nlp_models/text_summarization/abstractive/` |
| `text_summarization/extractive/` | `2_nlp_models/text_summarization/extractive/` |

---

## Framework Coverage

Models are implemented across multiple frameworks for flexibility and comparison:

- **PyTorch** - Primary deep learning framework
- **TensorFlow/Keras** - Keras-based implementations
- **Scikit-Learn** - Classical ML and preprocessing
- **JAX** - Numerical computing (selected models)
- **Hugging Face Transformers** - NLP and vision models
- **LangChain** - LLM integrations and RAG
- **OpenAI/Anthropic APIs** - Cloud-based LLMs

---

## Usage Examples

### Finding Models by Category

```bash
# View all supervised learning models
ls -R models/1_supervised_learning/

# View all NLP models
ls -R models/2_nlp_models/

# View image classification specifically
ls -R models/3_computer_vision/classification/single_label/

# View all transformer implementations
ls -R models/4_sequence_models/transformer/

# View all generative models
ls -R models/5_generative_models/
```

### Running Models

Each model typically has:
- `train_*.py` - Training scripts
- `generate.py` - Generation/inference scripts (for generative models)
- Implementation files with specific framework (PyTorch, TensorFlow, etc.)

Example:
```bash
# Train sentiment analysis with PyTorch
python models/2_nlp_models/sentiment_analysis/train_pytorch.py

# Run image classification with TensorFlow
python models/3_computer_vision/classification/single_label/train_tensorflow.py

# Generate with stable diffusion
python models/5_generative_models/diffusion_models/stable_diffusion/generate.py
```

---

## Benefits of This Structure

1. **Discoverability** - Models grouped by use case, not implementation detail
2. **Scalability** - Easy to add new models within existing categories
3. **Clarity** - Clear separation of concerns (NLP vs Vision vs Generative)
4. **Consistency** - Standardized naming and file organization
5. **Comparison** - Easy to compare multiple approaches for the same task
6. **Multi-framework** - Each category supports multiple framework implementations

---

## Future Expansion

### Planned Categories
- Enhanced 6_unsupervised_learning with actual implementations
- Enhanced 7_multimodal_learning with more models
- Time series forecasting
- Graph neural networks
- Reinforcement learning agents

### Structure Stability

This structure is stable and designed to accommodate significant growth while maintaining clarity and organization.
