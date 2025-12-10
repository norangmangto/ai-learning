# Llama Variants - Meta's Open Source LLM Family

This directory contains comprehensive implementations of Meta's Llama family of large language models, including Llama 2, Llama 3, Code Llama, and fine-tuning approaches.

## 📁 Files Overview

| File | Description | Key Features |
|------|-------------|--------------|
| `train_llama2.py` | Llama 2 implementation | 7B/13B/70B variants, chat format |
| `train_llama3.py` | Llama 3 implementation | 8B/70B variants, improved performance |
| `train_codellama.py` | Code Llama implementation | Code generation, 17+ languages |
| `train_llama_finetuning.py` | Fine-tuning with LoRA/QLoRA | Efficient fine-tuning techniques |
| `train_all_llamas.py` | Comprehensive comparison | All variants with benchmarks |

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install core dependencies
pip install transformers torch accelerate

# For fine-tuning
pip install peft bitsandbytes datasets trl

# Login to HuggingFace
huggingface-cli login
```

### 2. Accept Model License

Before using Llama models, you must accept the license:
1. Visit: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
2. Accept the license agreement
3. For Llama 3: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

### 3. Run Examples

```bash
# Llama 2
python train_llama2.py

# Llama 3
python train_llama3.py

# Code Llama
python train_codellama.py

# Fine-tuning guide
python train_llama_finetuning.py

# Comprehensive comparison
python train_all_llamas.py
```

## 📊 Model Comparison

### Llama 2 Family

| Model | Parameters | Context | Memory (FP16) | Best For |
|-------|-----------|---------|---------------|----------|
| Llama-2-7b | 7B | 4K | ~14 GB | Development, testing |
| Llama-2-7b-chat | 7B | 4K | ~14 GB | Chatbots |
| Llama-2-13b | 13B | 4K | ~26 GB | Better quality |
| Llama-2-13b-chat | 13B | 4K | ~26 GB | Production chatbots |
| Llama-2-70b | 70B | 4K | ~140 GB | Highest quality |
| Llama-2-70b-chat | 70B | 4K | ~140 GB | Production (premium) |

### Llama 3 Family

| Model | Parameters | Context | Vocabulary | Best For |
|-------|-----------|---------|------------|----------|
| Meta-Llama-3-8B | 8B | 8K | 128K | Fine-tuning, custom apps |
| Meta-Llama-3-8B-Instruct | 8B | 8K | 128K | Chat, instruction following |
| Meta-Llama-3-70B | 70B | 8K | 128K | Research |
| Meta-Llama-3-70B-Instruct | 70B | 8K | 128K | Production (high quality) |

**Llama 3 Improvements:**
- 4x larger vocabulary (128K vs 32K)
- 2x longer context (8K vs 4K)
- 15-20% better performance
- Trained on 15T+ tokens
- Better multilingual support

### Code Llama Family

| Model | Parameters | Context | Specialization |
|-------|-----------|---------|----------------|
| CodeLlama-7b | 7B | 100K | General code |
| CodeLlama-7b-Python | 7B | 100K | Python code |
| CodeLlama-7b-Instruct | 7B | 100K | NL to code |
| CodeLlama-13b | 13B | 100K | Better quality |
| CodeLlama-34b | 34B | 100K | Highest quality |

**Supported Languages:** Python, C++, Java, JavaScript, TypeScript, C#, Go, Rust, PHP, Ruby, Swift, Kotlin, Scala, Perl, Lua, R, SQL

## 🎯 Use Case Guide

### For Different Scenarios

| Use Case | Recommended Model |
|----------|------------------|
| **Chatbot (Development)** | Llama 2 7B Chat |
| **Chatbot (Production)** | Llama 3 8B Instruct |
| **Chatbot (Premium)** | Llama 3 70B Instruct |
| **Code Generation** | Code Llama 7B Instruct |
| **Code Completion** | Code Llama 7B Base |
| **Python Specific** | Code Llama 7B Python |
| **Research** | Llama 3 70B Base |
| **Fine-tuning** | Llama 2/3 7B/8B Base |
| **Low Memory** | Any model with 4-bit quantization |
| **Long Context** | Code Llama (100K tokens) |

## 💾 Memory Requirements

### Standard Loading (FP16)

| Model Size | Memory Required |
|-----------|----------------|
| 7B/8B | ~14-16 GB |
| 13B | ~26 GB |
| 34B | ~68 GB |
| 70B | ~140 GB |

### With Quantization

| Quantization | 7B Model | 13B Model | 70B Model |
|-------------|----------|-----------|-----------|
| **4-bit (QLoRA)** | ~5 GB | ~9 GB | ~40 GB |
| **8-bit** | ~7 GB | ~13 GB | ~70 GB |
| **FP16** | ~14 GB | ~26 GB | ~140 GB |

## 🔧 Fine-tuning

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,          # LoRA scaling
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# Trainable params: ~0.1% of total
```

### QLoRA (4-bit Quantization + LoRA)

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
# Memory: ~5GB instead of ~14GB
```

### Fine-tuning Comparison

| Method | Memory (7B) | Trainable % | Adapter Size |
|--------|------------|-------------|--------------|
| **Full Fine-tuning** | ~28 GB | 100% | Full model |
| **LoRA** | ~16 GB | 0.1% | ~10 MB |
| **QLoRA** | ~5 GB | 0.1% | ~10 MB |

## 📝 Usage Examples

### Basic Text Generation (Llama 2)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Chat Format (Llama 2)

```python
system_prompt = "You are a helpful AI assistant."
user_message = "Explain neural networks briefly."

chat = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""

inputs = tokenizer(chat, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150)
```

### Chat Format (Llama 3)

```python
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is deep learning?"}
]

# Llama 3 uses apply_chat_template
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
```

### Code Generation (Code Llama)

```python
model_name = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "[INST] Write a Python function to calculate fibonacci [/INST]"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.2,  # Lower for code
    top_p=0.95
)
```

## 🎓 Generation Parameters

### Recommended Settings

| Parameter | Llama 2 | Llama 3 | Code Llama |
|-----------|---------|---------|------------|
| **Temperature** | 0.7 | 0.6 | 0.1-0.2 |
| **Top-p** | 0.9 | 0.9 | 0.95 |
| **Top-k** | 50 | 50 | 50 |
| **Max tokens** | 512-2048 | 512-2048 | 256-1024 |
| **Repetition penalty** | 1.1 | 1.1 | 1.0 |

### Parameter Effects

- **Temperature**: Controls randomness (0.1 = deterministic, 1.0 = creative)
- **Top-p**: Nucleus sampling (0.9 recommended for balanced output)
- **Top-k**: Limits vocabulary per step (50 is good default)
- **Repetition penalty**: Reduces repetition (1.1-1.2 recommended)

## 📚 Popular Fine-tuning Datasets

| Dataset | Size | Type | License |
|---------|------|------|---------|
| **Alpaca** | 52K | Instruction | Research only |
| **Dolly-15k** | 15K | Instruction | Commercial OK |
| **OpenAssistant** | 161K | Conversation | Apache 2.0 |
| **Code Alpaca** | 20K | Code | Research only |
| **ShareGPT** | 90K | Conversation | Community |

## 🔍 Benchmark Performance

### General Benchmarks

| Model | MMLU | GSM8K | HumanEval |
|-------|------|-------|-----------|
| Llama 2 (7B) | 45% | 15% | 13% |
| Llama 2 (13B) | 55% | 30% | 18% |
| Llama 2 (70B) | 68% | 55% | 30% |
| Llama 3 (8B) | 68% | 75% | 62% |
| Llama 3 (70B) | 82% | 93% | 81% |
| Code Llama (7B) | 35% | 25% | 35% |
| Code Llama (34B) | 55% | 50% | 48% |

- **MMLU**: General knowledge (57 subjects)
- **GSM8K**: Grade school math
- **HumanEval**: Python code generation

## 🛠️ Advanced Features

### Flash Attention 2

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # 2x faster
)
```

### Gradient Checkpointing (Fine-tuning)

```python
model.gradient_checkpointing_enable()
# Reduces memory at cost of ~20% slower training
```

### CPU Offloading

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload",  # Offload to disk if needed
    max_memory={0: "10GB", "cpu": "30GB"}
)
```

## 🐛 Troubleshooting

### Out of Memory

```python
# Use 4-bit quantization
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=config
)
```

### Slow Generation

```python
# Enable flash attention and compile
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"
)
model = torch.compile(model)  # PyTorch 2.0+
```

### Authentication Errors

```bash
# Login and accept license
huggingface-cli login
# Then visit model page and accept license
```

## 📖 Resources

### Official Documentation
- [Llama 2 Paper](https://arxiv.org/abs/2307.09288)
- [Llama 3 Announcement](https://ai.meta.com/blog/meta-llama-3/)
- [Code Llama Paper](https://arxiv.org/abs/2308.12950)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)

### Model Cards
- [Llama 2 Models](https://huggingface.co/meta-llama)
- [Llama 3 Models](https://huggingface.co/meta-llama)
- [Code Llama Models](https://huggingface.co/codellama)

### Community
- [Llama Discord](https://discord.gg/llama)
- [HuggingFace Forums](https://discuss.huggingface.co/)
- [Reddit r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

## 🎯 Best Practices

### Model Selection
- Start with 7B/8B models for experimentation
- Use Chat/Instruct variants for conversations
- Use Base models for fine-tuning
- Use Code Llama for programming tasks
- Scale to larger models only when needed

### Performance Optimization
- Use 4-bit quantization for limited GPU memory
- Enable flash-attention-2 for 2x faster inference
- Batch multiple requests when possible
- Cache model weights to avoid re-downloading
- Use appropriate context length (don't waste tokens)

### Fine-tuning Tips
- Use QLoRA for consumer GPUs (12GB+)
- Start with learning rate 2e-4
- Train for 3-5 epochs, monitor validation loss
- Use high-quality datasets (1K+ examples minimum)
- Evaluate on held-out test set

### Production Deployment
- Use vLLM or TGI for optimized serving
- Implement request batching
- Set up monitoring and logging
- Use smaller models when quality permits
- Consider model distillation for edge deployment

## 📝 License

Llama models are released under custom licenses:
- **Llama 2**: Llama 2 Community License (commercial use allowed)
- **Llama 3**: Llama 3 Community License (commercial use allowed)
- **Code Llama**: Llama 2 Community License (commercial use allowed)

See individual model cards for complete license terms.

## 🚀 Next Steps

1. **Start Simple**: Begin with `train_llama3.py`
2. **Experiment**: Try different models and parameters
3. **Fine-tune**: Use `train_llama_finetuning.py` for custom tasks
4. **Deploy**: Set up production inference
5. **Optimize**: Implement quantization and batching
6. **Evaluate**: Measure performance on your tasks

For comprehensive comparison, run:
```bash
python train_all_llamas.py
```
