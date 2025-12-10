"""
Fine-tuning Language Models for LangChain
This script demonstrates how to fine-tune LLMs that work with LangChain:
- GPT-2 / GPT-Neo (open source)
- LLaMA / Mistral (with LoRA/QLoRA)
- Instruction-tuning for chat/QA tasks
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os
from peft import LoraConfig, get_peft_model
import json

# For LangChain integration
try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠ LangChain not installed. Install with: pip install langchain langchain-community")
    LANGCHAIN_AVAILABLE = False


def create_instruction_dataset():
    """
    Create instruction-following dataset for fine-tuning.
    Format: Instruction -> Response pairs
    """

    # Sample instruction-response pairs
    # In production, use larger datasets like Alpaca, Dolly, or domain-specific data
    data = [
        {
            "instruction": "Explain what a vector database is.",
            "response": "A vector database is a specialized database designed to store and query high-dimensional vector embeddings. It enables semantic search by finding similar vectors using distance metrics like cosine similarity. Vector databases are essential for RAG systems, recommendation engines, and similarity search applications."
        },
        {
            "instruction": "What is LangChain and what is it used for?",
            "response": "LangChain is a framework for developing applications powered by language models. It provides tools for chaining LLM calls, managing prompts, integrating with vector stores, and building complex AI applications. LangChain is commonly used for chatbots, question-answering systems, document analysis, and retrieval-augmented generation (RAG) pipelines."
        },
        {
            "instruction": "How does RAG (Retrieval Augmented Generation) work?",
            "response": "RAG combines information retrieval with language model generation. First, relevant documents are retrieved from a knowledge base using semantic search. Then, these documents are provided as context to a language model, which generates a response based on both the retrieved information and its training. This approach reduces hallucinations and grounds responses in factual information."
        },
        {
            "instruction": "What are the benefits of using LoRA for fine-tuning?",
            "response": "LoRA (Low-Rank Adaptation) offers several benefits: 1) Memory efficiency - only trains a small subset of parameters, 2) Faster training - requires less computational resources, 3) Modular - LoRA adapters can be swapped without changing the base model, 4) Preserves base model - original model weights remain frozen, 5) Cost-effective - enables fine-tuning large models on consumer hardware."
        },
        {
            "instruction": "Explain the difference between embeddings and language models.",
            "response": "Embeddings convert text into fixed-size numerical vectors that capture semantic meaning, enabling similarity comparisons. They're typically smaller models (100M-1B parameters) used for search and retrieval. Language models generate text and understand complex instructions, typically being much larger (7B-70B+ parameters). In a RAG system, embeddings find relevant information while language models generate coherent responses."
        },
        {
            "instruction": "What is prompt engineering?",
            "response": "Prompt engineering is the practice of designing effective prompts to get desired outputs from language models. It involves crafting instructions, providing examples (few-shot learning), structuring context, and using templates. Good prompt engineering can significantly improve model performance without fine-tuning, making it a cost-effective way to adapt LLMs to specific tasks."
        },
        {
            "instruction": "How do you evaluate a language model's performance?",
            "response": "Language models are evaluated using multiple methods: 1) Perplexity - measures how well the model predicts text, 2) Task-specific metrics - accuracy for classification, BLEU/ROUGE for generation, 3) Human evaluation - rating quality, coherence, and factuality, 4) Benchmark datasets - MMLU, HellaSwag, TruthfulQA, 5) Domain-specific tests - accuracy on your use case."
        },
        {
            "instruction": "What are the key components of a LangChain application?",
            "response": "Key LangChain components include: 1) LLMs - the language models for generation, 2) Prompts - templates for formatting inputs, 3) Chains - sequences of LLM calls and logic, 4) Memory - conversation history management, 5) Embeddings - for semantic search, 6) Vector stores - document storage and retrieval, 7) Agents - autonomous decision-making systems, 8) Tools - external APIs and functions the LLM can use."
        },
    ]

    # Format as instruction-following prompts
    formatted_data = []
    for item in data:
        # Alpaca-style format
        text = f"""### Instruction:
{item['instruction']}

### Response:
{item['response']}"""
        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)


def train_gpt2_model():
    """
    Fine-tune GPT-2 (small, fast, good for learning)
    Can be used with LangChain's HuggingFacePipeline
    """

    print("=" * 60)
    print("Fine-tuning GPT-2 for LangChain Applications")
    print("=" * 60)

    model_name = "gpt2"  # or "gpt2-medium" for better quality
    print(f"\n1. Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    print(f"✓ Model loaded")
    print(f"  Parameters: {model.num_parameters() / 1e6:.1f}M")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # 2. Prepare dataset
    print("\n2. Preparing training dataset...")
    dataset = create_instruction_dataset()

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    print(f"✓ Dataset prepared: {len(tokenized_dataset)} examples")

    # 3. Set up training
    output_dir = "models/finetuned_gpt2"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Disable wandb
    )

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # 4. Train
    print("\n3. Fine-tuning model...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 5. Save model
    print("\n4. Saving fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to: {output_dir}")

    # 6. QA Validation
    print("\n" + "=" * 60)
    print("QA Validation")
    print("=" * 60)

    # Test generation
    test_prompts = [
        "### Instruction:\nWhat is LangChain?\n\n### Response:",
        "### Instruction:\nExplain vector databases.\n\n### Response:",
    ]

    print("\n--- Generation Tests ---")
    model.eval()

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt.split('Instruction:')[1].split('Response:')[0].strip()}")

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.split("### Response:")[-1].strip()

        print(f"Response: {response[:200]}...")

    # Sanity checks
    print("\n--- Sanity Checks ---")

    # Check 1: Model files exist
    config_file = os.path.join(output_dir, "config.json")

    if os.path.exists(config_file):
        print("✓ Model configuration saved")
    else:
        print("✗ WARNING: Model config missing")

    # Check 2: Can load model
    try:
        AutoModelForCausalLM.from_pretrained(output_dir)
        print("✓ Model can be reloaded successfully")
    except Exception as e:
        print(f"✗ WARNING: Cannot reload model: {e}")

    # Check 3: Tokenizer works
    test_text = "Hello world"
    tokens = tokenizer(test_text, return_tensors="pt")
    if tokens['input_ids'].shape[1] > 0:
        print("✓ Tokenizer working correctly")
    else:
        print("✗ WARNING: Tokenizer issue")

    print("\n=== Overall Validation Result ===")
    validation_passed = (
        os.path.exists(config_file) and
        len(response) > 10
    )

    if validation_passed:
        print("✓ Model validation PASSED - Ready for LangChain integration")
    else:
        print("✗ Model validation FAILED - Review training process")

    return output_dir


def train_with_lora():
    """
    Fine-tune a model using LoRA (Low-Rank Adaptation)
    More memory efficient, suitable for larger models like LLaMA/Mistral
    """

    print("\n" + "=" * 60)
    print("Fine-tuning with LoRA (Efficient Approach)")
    print("=" * 60)

    # Use a small model for demonstration
    # For production, use: "meta-llama/Llama-2-7b-hf" or "mistralai/Mistral-7B-v0.1"
    model_name = "gpt2"

    print(f"\n1. Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    print(f"✓ Base model loaded: {model.num_parameters() / 1e6:.1f}M parameters")

    # 2. Configure LoRA
    print("\n2. Configuring LoRA...")

    lora_config = LoraConfig(
        r=8,  # Rank - higher = more parameters but better quality
        lora_alpha=32,  # Scaling factor
        target_modules=["c_attn"],  # For GPT-2; adjust for other models
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"✓ LoRA configured")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")

    # 3. Prepare dataset
    print("\n3. Preparing dataset...")
    dataset = create_instruction_dataset()

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(f"✓ Dataset ready: {len(tokenized_dataset)} examples")

    # 4. Training
    output_dir = "models/finetuned_lora"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("\n4. Training with LoRA...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 5. Save LoRA adapters
    print("\n5. Saving LoRA adapters...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config
    config = {
        "base_model": model_name,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": lora_config.target_modules,
    }

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ LoRA adapters saved to: {output_dir}")
    print(f"  Note: Only adapter weights saved (~few MB), not full model")

    print("\n=== Validation ===")

    # Check files
    adapter_file = os.path.join(output_dir, "adapter_model.bin")
    if os.path.exists(adapter_file):
        size_mb = os.path.getsize(adapter_file) / (1024 * 1024)
        print(f"✓ LoRA adapter saved: {size_mb:.2f} MB")
    else:
        print("✗ WARNING: Adapter file not found")

    print("✓ LoRA fine-tuning complete")

    return output_dir


def test_langchain_integration():
    """
    Test fine-tuned models with LangChain
    """

    if not LANGCHAIN_AVAILABLE:
        print("\nSkipping LangChain tests - package not installed")
        return

    print("\n" + "=" * 60)
    print("Testing LangChain Integration")
    print("=" * 60)

    from transformers import pipeline

    model_path = "models/finetuned_gpt2"

    if not os.path.exists(model_path):
        print(f"⚠ Model not found: {model_path}")
        print("  Run training first to create the model")
        return

    print(f"\n1. Loading fine-tuned model: {model_path}")

    # Create HuggingFace pipeline
    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )

    # Create LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    print("✓ Model loaded into LangChain")

    # 2. Test with simple prompts
    print("\n2. Testing direct prompts...")

    test_prompts = [
        "What is RAG?",
        "Explain vector databases in simple terms.",
    ]

    for prompt in test_prompts:
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:"
        print(f"\nPrompt: {prompt}")
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response[:150]}...")

    # 3. Test with LangChain chain
    print("\n3. Testing LangChain Chains...")

    template = """### Instruction:
{question}

### Response:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = LLMChain(llm=llm, prompt=prompt)

    questions = [
        "What are the benefits of using LangChain?",
        "How do embeddings work?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        response = chain.invoke({"question": question})
        print(f"Response: {response['text'][:150]}...")

    print("\n✓ LangChain integration test complete")


def main():
    """Main training pipeline"""

    print("\n🤖 Language Model Fine-tuning for LangChain")
    print("=" * 60)

    print("\nChoose training approach:")
    print("1. Full fine-tuning (GPT-2) - Good for learning")
    print("2. LoRA fine-tuning - Memory efficient for larger models")
    print("3. Both + LangChain integration test")

    # For automation, run option 1
    choice = "1"

    if choice == "1":
        train_gpt2_model()
    elif choice == "2":
        train_with_lora()
    else:
        # Run both
        train_gpt2_model()
        train_with_lora()

    # Test LangChain integration
    test_langchain_integration()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Test models with LangChain:")
    print("   from langchain_community.llms import HuggingFacePipeline")
    print("   llm = HuggingFacePipeline.from_model_id(...)")
    print("\n2. Build applications:")
    print("   • Question-answering systems")
    print("   • Chatbots with memory")
    print("   • RAG pipelines")
    print("   • Agent systems")
    print("\n3. Recommended Models for LangChain:")
    print("   • GPT-2/Neo: Fast, good for prototyping")
    print("   • Mistral-7B: High quality, efficient")
    print("   • LLaMA-2: Instruction-tuned, chat optimized")
    print("   • Falcon: Strong performance, permissive license")
    print("\n4. Production Tips:")
    print("   • Use LoRA for large models (>7B parameters)")
    print("   • Quantize models (4-bit/8-bit) for lower memory")
    print("   • Use vLLM or TGI for faster inference")
    print("   • Cache embeddings for repeated queries")


if __name__ == "__main__":
    main()
