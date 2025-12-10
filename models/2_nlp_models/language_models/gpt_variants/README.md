# Alternative LLM Approaches

This directory contains implementations of various approaches to using Large Language Models (LLMs) without relying solely on commercial APIs.

## 📁 Files Overview

| File | Description | Key Features |
|------|-------------|--------------|
| `train_huggingface.py` | Hugging Face Transformers | DistilGPT2, GPT-2, BERT, BART |
| `train_ollama.py` | Ollama integration | Local LLM server, API usage |
| `train_langchain.py` | LangChain framework | Multi-provider, chains, RAG |
| `train_quantized.py` | Quantized models | GGUF format, CPU-optimized |
| `train_all_approaches.py` | Comprehensive demo | All approaches with benchmarks |

## 🚀 Quick Start

### 1. Hugging Face Transformers (Simplest)

```bash
# Install
pip install transformers torch

# Run
python train_huggingface.py
```

**Features:**
- No external services required
- Multiple pre-trained models
- Text generation, classification, QA, summarization
- Works on CPU or GPU

### 2. Ollama (Recommended for Development)

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama2
ollama pull mistral

# Start server
ollama serve

# Run
python train_ollama.py
```

**Features:**
- Easy local deployment
- High-quality models (Llama2, Mistral, CodeLlama)
- Simple API
- No API keys required

### 3. LangChain (Best for Complex Apps)

```bash
# Install
pip install langchain langchain-community transformers torch

# Run
python train_langchain.py
```

**Features:**
- Multi-provider support
- Chain composition
- Conversation memory
- RAG (Retrieval Augmented Generation)
- Agent capabilities

### 4. Quantized Models (Best for CPUs)

```bash
# Install
pip install gpt4all
# or
pip install llama-cpp-python

# Run
python train_quantized.py
```

**Features:**
- CPU-optimized inference
- Small file sizes (4-bit quantization)
- No GPU required
- Good performance on laptops

## 📊 Comparison

| Approach | Setup | Quality | Speed | Memory | Best For |
|----------|-------|---------|-------|--------|----------|
| **Hugging Face** | Easy | Good | Medium | High | Experimentation |
| **Ollama** | Easy | Excellent | Fast | Medium | Development |
| **LangChain** | Medium | Varies | Varies | Medium | Complex apps |
| **Quantized** | Medium | Good | Fast | Low | Edge devices |

## 🎯 Use Case Recommendations

### For Learning & Experimentation
```bash
python train_huggingface.py
```
- No setup required
- Many models available
- Good documentation

### For Development & Prototyping
```bash
# Start Ollama
ollama serve

# Run
python train_ollama.py
```
- High quality models
- Fast inference
- Local control

### For Complex Applications
```bash
python train_langchain.py
```
- Multi-step workflows
- Conversational AI
- Document Q&A systems

### For Production on Edge Devices
```bash
python train_quantized.py
```
- Runs on CPU
- Small memory footprint
- Fast inference

### For Comprehensive Testing
```bash
python train_all_approaches.py
```
- Tests all approaches
- Performance benchmarks
- Side-by-side comparison

## 📦 Installation

### Minimal Setup (Hugging Face only)
```bash
pip install transformers torch
```

### Full Setup (All approaches)
```bash
# Core libraries
pip install transformers torch

# Ollama (separate installation)
# macOS/Linux: curl https://ollama.ai/install.sh | sh
# Windows: Download from ollama.ai

# LangChain
pip install langchain langchain-community langchain-openai
pip install faiss-cpu sentence-transformers

# Quantized models
pip install gpt4all llama-cpp-python

# Optional: Ollama Python client
pip install ollama
```

## 🔧 Configuration

### Environment Variables (Optional)

```bash
# For OpenAI (if using LangChain with OpenAI)
export OPENAI_API_KEY="your-key-here"

# For Anthropic Claude (if using LangChain with Claude)
export ANTHROPIC_API_KEY="your-key-here"
```

### Ollama Configuration

```bash
# List available models
ollama list

# Pull additional models
ollama pull codellama    # For code generation
ollama pull neural-chat  # For conversation
ollama pull mistral      # Fast and efficient

# Remove models
ollama rm model-name
```

## 💡 Examples

### Simple Text Generation
```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
result = generator("Machine learning is", max_length=50)
print(result[0]['generated_text'])
```

### Ollama API Call
```python
import requests

response = requests.post('http://localhost:11434/api/generate',
    json={"model": "llama2", "prompt": "What is AI?", "stream": False})
print(response.json()['response'])
```

### LangChain with Memory
```python
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = Ollama(model="llama2")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

response1 = conversation.predict(input="My name is Alice")
response2 = conversation.predict(input="What is my name?")
```

### Quantized Model
```python
from gpt4all import GPT4All

model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
response = model.generate("What is Python?", max_tokens=100)
print(response)
```

## 🐛 Troubleshooting

### Hugging Face Issues
```bash
# Clear cache if models fail to load
rm -rf ~/.cache/huggingface/

# Re-install transformers
pip install --upgrade transformers torch
```

### Ollama Issues
```bash
# Check if server is running
curl http://localhost:11434/api/tags

# Restart Ollama
killall ollama
ollama serve

# Check logs
ollama logs
```

### Memory Issues
- Use smaller models (DistilGPT2 instead of GPT-2)
- Use quantized models (Q4_K_M)
- Reduce batch size
- Use CPU instead of GPU for small tasks

## 📚 Resources

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [LangChain Documentation](https://python.langchain.com/)
- [GPT4All Documentation](https://docs.gpt4all.io/)

### Model Resources
- [Hugging Face Models](https://huggingface.co/models)
- [TheBloke's Quantized Models](https://huggingface.co/TheBloke)
- [Ollama Model Library](https://ollama.ai/library)
- [GPT4All Models](https://gpt4all.io/models/gguf/)

## 🎓 Learning Path

1. **Start Here:** `train_huggingface.py`
   - Understand basic text generation
   - Learn about different tasks (classification, QA, summarization)

2. **Next Step:** `train_ollama.py`
   - Set up local LLM server
   - Learn API integration

3. **Advanced:** `train_langchain.py`
   - Build complex chains
   - Implement RAG systems
   - Add conversation memory

4. **Optimization:** `train_quantized.py`
   - Learn about quantization
   - Optimize for production

5. **Integration:** `train_all_approaches.py`
   - Compare all approaches
   - Choose the best for your use case

## 🚀 Next Steps

After completing these implementations, consider:

1. **Fine-tuning**: Adapt models to your specific domain
2. **RAG Systems**: Build document Q&A applications
3. **Agents**: Create autonomous AI agents with LangChain
4. **Deployment**: Deploy models in production environments
5. **Evaluation**: Implement proper model evaluation metrics

## 📝 License

These implementations are for educational purposes. Please check the licenses of individual models and libraries before commercial use.
