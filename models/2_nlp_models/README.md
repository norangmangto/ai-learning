# LangChain Model Training

Train and fine-tune models that integrate seamlessly with LangChain for building production-ready AI applications.

## 📚 Overview

This directory contains training scripts for three core LangChain model types:

1. **Embedding Models** - For semantic search and retrieval
2. **Language Models (LLMs)** - For text generation and reasoning
3. **Retrieval Models** - For RAG systems and document search

## 🚀 Quick Start

### Install Dependencies

```bash
# Core dependencies
pip install torch transformers sentence-transformers datasets

# LangChain
pip install langchain langchain-community langchain-openai

# Vector stores and retrieval
pip install faiss-cpu chromadb rank-bm25

# Training utilities
pip install peft accelerate bitsandbytes
```

### Run Training Scripts

```bash
# Train embedding models
python train_embeddings.py

# Train language models
python train_llm.py

# Train retrieval models
python train_retriever.py
```

## 📖 Training Scripts

### 1. Embedding Models (`train_embeddings.py`)

Fine-tune embedding models for domain-specific semantic search.

**What it trains:**
- Sentence Transformers (bi-encoders)
- Domain-specific embeddings
- Models compatible with LangChain's `HuggingFaceEmbeddings`

**Use cases:**
- Semantic search
- Document similarity
- RAG retrieval layer
- Clustering and classification

**LangChain integration:**
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="models/finetuned_embeddings"
)
```

**Best practices:**
- Use cosine similarity loss for semantic matching
- Train on query-document pairs from your domain
- Typical dimensions: 384 (fast), 768 (accurate), 1536 (high quality)
- Fine-tune on 1K-10K examples for good results

### 2. Language Models (`train_llm.py`)

Fine-tune LLMs for instruction following and generation tasks.

**What it trains:**
- GPT-2 / GPT-Neo (full fine-tuning)
- LLaMA / Mistral with LoRA (efficient fine-tuning)
- Instruction-tuned models

**Use cases:**
- Custom chatbots
- Domain-specific QA
- Text generation
- Agent reasoning

**LangChain integration:**
```python
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline("text-generation", model="models/finetuned_gpt2")
llm = HuggingFacePipeline(pipeline=pipe)
```

**Training approaches:**

**Full Fine-tuning:**
- Pros: Best quality, full model adaptation
- Cons: Requires more memory and time
- Use for: Small models (<3B), plenty of GPU memory

**LoRA (Low-Rank Adaptation):**
- Pros: Memory efficient, fast, modular
- Cons: Slightly lower quality than full fine-tuning
- Use for: Large models (7B+), limited GPU memory

**Best practices:**
- Use instruction format: "### Instruction: ... ### Response: ..."
- Train on 100+ high-quality examples minimum
- Use 4-bit quantization for models >7B parameters
- Monitor generation quality, not just loss

### 3. Retrieval Models (`train_retriever.py`)

Train specialized models for document retrieval in RAG systems.

**What it trains:**
- Dense retrievers (bi-encoders for semantic search)
- Cross-encoders (re-rankers for accuracy)
- Hybrid retrieval systems

**Use cases:**
- RAG pipelines
- Question answering
- Document search
- Multi-stage retrieval

**LangChain integration:**
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="models/finetuned_retriever")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
```

**Two-stage retrieval:**
1. **Dense retriever** (fast): Get top 50-100 candidates
2. **Cross-encoder** (accurate): Re-rank to top 5-10

**Best practices:**
- Train on query-document-relevance triplets
- Use hard negatives (similar but irrelevant docs)
- Combine with BM25 for hybrid search
- Cache document embeddings

## 🎯 Model Recommendations

### For Embeddings

| Model | Dimension | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ⭐⭐⭐ | General purpose, fast |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ⭐⭐⭐⭐ | Higher accuracy |
| e5-large-v2 | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | Best quality |
| text-embedding-3-small | 1536 | ⚡⚡ | ⭐⭐⭐⭐ | OpenAI (paid) |

### For LLMs

| Model | Size | Speed | Quality | License |
|-------|------|-------|---------|---------|
| GPT-2 | 124M | ⚡⚡⚡ | ⭐⭐ | MIT |
| GPT-Neo-2.7B | 2.7B | ⚡⚡ | ⭐⭐⭐ | MIT |
| Mistral-7B | 7B | ⚡⚡ | ⭐⭐⭐⭐⭐ | Apache 2.0 |
| LLaMA-2-7B | 7B | ⚡⚡ | ⭐⭐⭐⭐ | LLaMA 2 |
| Falcon-7B | 7B | ⚡⚡ | ⭐⭐⭐⭐ | Apache 2.0 |

### For Retrieval

| Model | Type | Use Case |
|-------|------|----------|
| all-MiniLM-L6-v2 | Bi-encoder | Fast initial retrieval |
| ms-marco-MiniLM-L6 | Cross-encoder | Accurate re-ranking |
| e5-base-v2 | Bi-encoder | Better retrieval quality |
| bge-large-en | Bi-encoder | State-of-the-art retrieval |

## 💡 Complete RAG Example

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# 1. Load fine-tuned embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="models/finetuned_embeddings"
)

# 2. Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Load fine-tuned LLM
pipe = pipeline("text-generation", model="models/finetuned_gpt2")
llm = HuggingFacePipeline(pipeline=pipe)

# 4. Create RAG chain
template = """Use the following context to answer the question.

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# 5. Query
response = qa_chain.invoke({"query": "What is RAG?"})
print(response["result"])
```

## 🔧 Fine-tuning Tips

### Data Requirements

**Embeddings:**
- Minimum: 100 query-document pairs
- Recommended: 1K-10K pairs
- Format: (query, positive_doc, negative_doc)

**LLMs:**
- Minimum: 100 instruction-response pairs
- Recommended: 1K-100K pairs
- Format: Instruction + Response

**Retrievers:**
- Minimum: 500 query-document-relevance triplets
- Recommended: 10K-100K triplets
- Include hard negatives for better quality

### Training Hyperparameters

**For embeddings:**
```python
epochs = 3-10
batch_size = 16-32
learning_rate = 2e-5
loss = CosineSimilarityLoss or MultipleNegativesRankingLoss
```

**For LLMs (full):**
```python
epochs = 3-5
batch_size = 4-8 (with gradient accumulation)
learning_rate = 2e-5 to 5e-5
```

**For LLMs (LoRA):**
```python
epochs = 3-10
batch_size = 8-16
learning_rate = 1e-4 to 3e-4
lora_r = 8-16
lora_alpha = 16-32
```

### GPU Requirements

| Task | Model Size | Minimum VRAM | Recommended |
|------|-----------|--------------|-------------|
| Embeddings | 100M-400M | 4GB | 8GB |
| LLM (full) | <3B | 12GB | 24GB |
| LLM (LoRA) | 7B | 12GB | 16GB |
| LLM (QLoRA) | 7B | 8GB | 12GB |
| Retriever | 100M-400M | 4GB | 8GB |

## 📊 Evaluation Metrics

### Embeddings
- **Cosine similarity**: Measure query-document similarity
- **Retrieval accuracy**: Correct matches in top-K
- **Embedding visualization**: t-SNE plots

### LLMs
- **Perplexity**: Language modeling quality
- **BLEU/ROUGE**: Generation quality
- **Human evaluation**: Response quality
- **Task accuracy**: Correct answers on test set

### Retrievers
- **Recall@K**: Relevant docs in top K results
- **MRR**: Mean reciprocal rank
- **NDCG**: Normalized discounted cumulative gain
- **Precision@K**: Precision in top K results

## 🚀 Production Deployment

### Optimization Techniques

1. **Quantization**: 4-bit/8-bit for lower memory
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

2. **Model caching**: Cache embeddings for frequent docs
```python
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

store = LocalFileStore("./cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace="my_docs"
)
```

3. **Batch processing**: Process multiple queries together
```python
embeddings = model.encode(texts, batch_size=32)
```

4. **Inference servers**: Use vLLM or Text Generation Inference
```bash
# vLLM for fast LLM inference
pip install vllm
vllm serve models/finetuned_llm --dtype auto
```

### Monitoring

Track these metrics in production:
- Latency (p50, p95, p99)
- Throughput (requests/second)
- Memory usage
- Cache hit rate
- Quality metrics (user feedback)

## 📚 Additional Resources

### Datasets for Training

**Embeddings:**
- MS MARCO: 8.8M query-doc pairs
- Natural Questions: 307K QA pairs
- SQuAD: 100K+ QA pairs

**LLMs:**
- Alpaca: 52K instruction-following
- Dolly-15K: High-quality instruction data
- ShareGPT: Conversation data

**Retrieval:**
- MS MARCO Passage: Large-scale retrieval
- Natural Questions: Wikipedia QA
- BEIR: Diverse retrieval benchmark

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT (LoRA)](https://huggingface.co/docs/peft)

### Papers
- Sentence-BERT: [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
- LoRA: [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- RAG: [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
- Dense Passage Retrieval: [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)

## 🤝 Contributing

To add new training scripts:
1. Follow the existing structure with QA validation
2. Include LangChain integration examples
3. Add usage documentation
4. Test with sample data

## ⚠️ Notes

- Always validate models before production deployment
- Monitor for model drift over time
- Update models as your domain evolves
- Consider privacy and security for sensitive data
- Respect model licenses for commercial use

Happy training! 🚀
