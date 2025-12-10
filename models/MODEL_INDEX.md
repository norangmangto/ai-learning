# Model Index - Quick Reference

This is a quick reference guide to locate any model in the repository.

## 1. Supervised Learning Models
**Directory:** `models/1_supervised_learning/`

### Regression
- **Linear Regression**: `regression/`
  - train_sklearn.py, train_pytorch.py, train_tensorflow_v2.py, train_jax.py

### Classification
- **Logistic Regression**: `classification/`
- **Multi-Layer Perceptron (MLP)**: `classification/`
- **Support Vector Machine (SVM)**: `classification/`

### Ensemble Methods
- **Random Forest**: `ensemble_methods/`
- **XGBoost**: `ensemble_methods/`

**Total Files:** 14+ Python training scripts

---

## 2. NLP Models
**Directory:** `models/2_nlp_models/`

### Sentiment Analysis
- **BERT**: `sentiment_analysis/train_pytorch.py`
- **TensorFlow BERT**: `sentiment_analysis/train_tensorflow.py`
- **Scikit-Learn TF-IDF**: `sentiment_analysis/train_sklearn.py`

### Text Classification
- **BERT/RoBERTa**: `text_classification/train_pytorch.py`
- **TensorFlow**: `text_classification/train_tensorflow.py`
- **Scikit-Learn**: `text_classification/train_sklearn.py`

### Text Summarization
**Abstractive:**
- **BART**: `text_summarization/abstractive/train_pytorch.py`, `train_tensorflow.py`
- **T5**: Included in BART PyTorch implementation

**Extractive:**
- **TF-IDF**: `text_summarization/extractive/train_tensorflow.py`
- **DistilBERT**: `text_summarization/extractive/train_pytorch.py`

### Embeddings
- **Sentence Embeddings**: `embeddings/sentence_embeddings/train_embeddings.py` (v1, v2)
- **Word Embeddings**: `embeddings/word_embeddings/` (GloVe, Word2Vec)

### Language Models
- **GPT-2**: `language_models/gpt_variants/train_llm.py`
- **LLaMA/Mistral**: Covered via LangChain integration
- **OpenAI/Anthropic**: `language_models/gpt_variants/train_llm_v2.py`

### Retrieval Systems
- **Vector Search**: `retrieval_systems/vector_search/train_retriever.py`
- **RAG/LangChain**: Integration with LLMs

**Total Files:** 14+ Python training scripts

---

## 3. Computer Vision Models
**Directory:** `models/3_computer_vision/`

### Image Classification
- **ResNet/EfficientNet**: `classification/single_label/train_pytorch.py`, `train_pytorch_v2.py`, `train_tensorflow.py`
- **Vision Transformer**: Included in classification training

### Object Detection
- **YOLO**: `object_detection/single_stage/train_pytorch.py`, `train_pytorch_v2.py`
- **Faster R-CNN**: `object_detection/single_stage/train_tensorflow.py`

### Semantic Segmentation
- **FCN, UNet, DeepLabV3**: `semantic_segmentation/`

### Image-to-Image
- **Super-Resolution**: `image_to_image/unet/unet/train_pytorch.py`
- **Style Transfer**: `image_to_image/unet/unet/train_tensorflow.py`
- **Inpainting**: UNet-based implementation

### Video Analysis
- **Video Classification**: `video_analysis/video_classification/train_pytorch.py`, `train_tensorflow.py`
- **Video Object Detection**: `video_analysis/video_object_detection/train_pytorch.py`, `train_tensorflow.py`

**Total Files:** 15+ Python training scripts

---

## 4. Sequence Models
**Directory:** `models/4_sequence_models/`

### RNN Models
- **LSTM**: `rnn/train_pytorch.py`, `train_pytorch_v2.py`, `train_tensorflow.py`
- **GRU**: Variant in RNN implementations
- **Bidirectional RNN**: Bidirectional LSTM/GRU

### Transformers
- **Base Transformer**: `transformer/train_pytorch.py`
- **Vision Transformer**: `transformer/train_tensorflow.py`
- **GPT (Decoder-only)**: See NLP Models
- **BART (Encoder-Decoder)**: See NLP Models

### Attention Mechanisms
- **Self-Attention**: Core in transformer implementations
- **Multi-Head Attention**: Transformer attention mechanism

**Total Files:** 5+ Python training scripts

---

## 5. Generative Models
**Directory:** `models/5_generative_models/`

### GANs
- **DCGAN**: `gan/dcgan/train_pytorch.py`, `train_tensorflow.py`
- **Conditional GAN**: StyleGAN, Progressive GAN variants

### Autoencoders
- **Standard AE**: `autoencoder/standard_ae/train_pytorch.py`, `train_tensorflow.py`
- **Variational AE**: VAE implementation
- **Denoising AE**: Noise-robust variant

### Diffusion Models
- **Stable Diffusion**: `diffusion_models/stable_diffusion/` (text-to-image)
- **DALLE**: DALLE variants
- **DreamBooth**: Fine-tuning for personalization
  - `diffusion_models/dreambooth_lora/train.py`, `generate.py`
  - `diffusion_models/dreambooth_lora_flux/train.py`, `generate.py`
  - `diffusion_models/dreambooth_lora_sd3/train.py`, `generate.py`
- **Flux**: `diffusion_models/text_to_image/train_flux.py`
- **SD3**: `diffusion_models/text_to_image/train_sd3.py`

### Flow Models
- Normalizing Flows (placeholder for implementation)

### Text Generation
- **GPT-based**: See NLP Models
- **Seq2Seq**: Transformer-based generation

**Total Files:** 24+ Python training/generation scripts

---

## 6. Unsupervised Learning
**Directory:** `models/6_unsupervised_learning/`

*Currently placeholder structure. Ready for:*
- Clustering (K-Means, DBSCAN, Hierarchical)
- Dimensionality Reduction (PCA, t-SNE, UMAP)
- Anomaly Detection (Isolation Forest, AE-based)

---

## 7. Multimodal Learning
**Directory:** `models/7_multimodal_learning/`

### Audio-Visual
- **Speech-to-Text (Whisper)**: `audio_visual/speech_to_text/whisper/transcribe.py`

*Ready for expansion with:*
- Vision-Language models (CLIP variants)
- Audio-Visual synchronization
- Image-Text matching and VQA

---

## Finding Models by Framework

### PyTorch
Most models have `train_pytorch.py` implementations:
```
grep -r "train_pytorch.py" models/1_supervised_learning/
grep -r "train_pytorch.py" models/2_nlp_models/
grep -r "train_pytorch.py" models/3_computer_vision/
```

### TensorFlow/Keras
Most models have `train_tensorflow.py` implementations:
```
grep -r "train_tensorflow.py" models/
grep -r "train_tensorflow_v2.py" models/
```

### Scikit-Learn
Classical ML implementations:
```
grep -r "train_sklearn.py" models/
```

### JAX
Numerical computing (selected models):
```
grep -r "train_jax.py" models/
```

---

## Finding Models by Task

| Task | Location |
|------|----------|
| Image Classification (Single-Label) | `3_computer_vision/classification/single_label/` |
| Image Classification (Multi-Label) | `3_computer_vision/classification/multi_label/` |
| Object Detection | `3_computer_vision/object_detection/` |
| Image Segmentation | `3_computer_vision/semantic_segmentation/` |
| Sentiment Analysis | `2_nlp_models/sentiment_analysis/` |
| Text Classification | `2_nlp_models/text_classification/` |
| Text Summarization | `2_nlp_models/text_summarization/` |
| Text-to-Image | `5_generative_models/diffusion_models/` |
| Image-to-Image | `3_computer_vision/image_to_image/` |
| Video Analysis | `3_computer_vision/video_analysis/` |
| Sequence Modeling | `4_sequence_models/` |
| Language Models | `2_nlp_models/language_models/` |
| Embeddings | `2_nlp_models/embeddings/` |
| Clustering | `6_unsupervised_learning/clustering/` |
| GANs | `5_generative_models/gan/` |
| Autoencoders | `5_generative_models/autoencoder/` |

---

## Statistics

- **Total Categories:** 7
- **Total Subcategories:** 40+
- **Total Python Files:** 59+
- **Frameworks:** PyTorch, TensorFlow/Keras, Scikit-Learn, JAX, HuggingFace, LangChain
- **Models Implemented:** 20+

---

## Common Commands

```bash
# List all models in a category
ls -R models/1_supervised_learning/

# Find all PyTorch implementations
find models -name "train_pytorch.py"

# Find all TensorFlow implementations
find models -name "train_tensorflow*.py"

# Count files per category
find models/1_supervised_learning -name "*.py" | wc -l

# View directory tree
tree models -L 3 -d

# Find a specific model
find models -type d -name "*bert*"
find models -type d -name "*transformer*"
```

---

## Next Steps

1. **Explore a Category**: Start with any numbered directory (e.g., `models/1_supervised_learning/`)
2. **Run a Model**: Choose a Python training script and execute it
3. **Compare Implementations**: Look at the same model across different frameworks
4. **Extend**: Add new models following the directory pattern
5. **Document**: Update this index when adding new models

---

*Last Updated: December 9, 2024*
*Structure: Domain-based (7 main categories with 40+ subcategories)*
