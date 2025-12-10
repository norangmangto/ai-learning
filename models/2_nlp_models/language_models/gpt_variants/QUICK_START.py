"""
Quick Reference - Alternative LLM Approaches
Run this to see all available implementations
"""


def show_implementations():
    """Display all available implementations"""

    implementations = {
        "1. Hugging Face Transformers": {
            "file": "train_huggingface.py",
            "description": "Local models (DistilGPT2, GPT-2, BERT, BART)",
            "install": "pip install transformers torch",
            "run": "python train_huggingface.py",
            "use_case": "Experimentation and development",
            "difficulty": "⭐ Easy"
        },
        "2. Ollama": {
            "file": "train_ollama.py",
            "description": "Local LLM server (Llama2, Mistral, CodeLlama)",
            "install": "curl https://ollama.ai/install.sh | sh",
            "run": "ollama serve && python train_ollama.py",
            "use_case": "Development and prototyping",
            "difficulty": "⭐ Easy"
        },
        "3. LangChain": {
            "file": "train_langchain.py",
            "description": "Multi-provider framework with chains and RAG",
            "install": "pip install langchain langchain-community",
            "run": "python train_langchain.py",
            "use_case": "Complex LLM applications",
            "difficulty": "⭐⭐ Medium"
        },
        "4. Quantized Models": {
            "file": "train_quantized.py",
            "description": "CPU-optimized GGUF models",
            "install": "pip install gpt4all",
            "run": "python train_quantized.py",
            "use_case": "Edge devices and laptops",
            "difficulty": "⭐⭐ Medium"
        },
        "5. All Approaches": {
            "file": "train_all_approaches.py",
            "description": "Comprehensive demo with benchmarks",
            "install": "See individual approaches above",
            "run": "python train_all_approaches.py",
            "use_case": "Comparison and evaluation",
            "difficulty": "⭐ Easy"
        }
    }

    print("=" * 70)
    print("ALTERNATIVE LLM APPROACHES - QUICK REFERENCE")
    print("=" * 70)

    for name, details in implementations.items():
        print(f"\n{name}")
        print("-" * 70)
        print(f"File:        {details['file']}")
        print(f"Description: {details['description']}")
        print(f"Install:     {details['install']}")
        print(f"Run:         {details['run']}")
        print(f"Use Case:    {details['use_case']}")
        print(f"Difficulty:  {details['difficulty']}")

    print("\n" + "=" * 70)
    print("QUICK START GUIDE")
    print("=" * 70)

    print("\n1️⃣  Fastest Way to Start (No Setup):")
    print("   python train_huggingface.py")

    print("\n2️⃣  Best Quality (Requires Ollama):")
    print("   # Install Ollama from https://ollama.ai")
    print("   ollama pull llama2")
    print("   ollama serve")
    print("   python train_ollama.py")

    print("\n3️⃣  For Laptops/CPUs:")
    print("   pip install gpt4all")
    print("   python train_quantized.py")

    print("\n4️⃣  For Complex Apps:")
    print("   pip install langchain langchain-community")
    print("   python train_langchain.py")

    print("\n5️⃣  Compare All Approaches:")
    print("   python train_all_approaches.py")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS BY SCENARIO")
    print("=" * 70)

    scenarios = {
        "🎓 Learning ML/AI": "train_huggingface.py",
        "💻 Building a Prototype": "train_ollama.py",
        "🚀 Production Application": "train_langchain.py",
        "📱 Running on Laptop": "train_quantized.py",
        "🔬 Research & Comparison": "train_all_approaches.py",
        "🏢 Enterprise (Privacy)": "train_ollama.py",
        "⚡ Quick Demo": "train_huggingface.py"
    }

    for scenario, file in scenarios.items():
        print(f"{scenario:30} → {file}")

    print("\n" + "=" * 70)
    print("FEATURES COMPARISON")
    print("=" * 70)

    features = """
┌───────────────────┬──────────┬─────────┬────────┬────────────┐
│     Feature       │   HF     │ Ollama  │ Chain  │ Quantized  │
├───────────────────┼──────────┼─────────┼────────┼────────────┤
│ No Setup Required │    ✓     │    ✗    │   ✗    │     ✗      │
│ Best Quality      │    ~     │    ✓    │   ✓    │     ~      │
│ CPU Friendly      │    ~     │    ~    │   ~    │     ✓      │
│ Privacy (Local)   │    ✓     │    ✓    │   ~    │     ✓      │
│ Memory Efficient  │    ✗     │    ~    │   ~    │     ✓      │
│ Complex Workflows │    ✗     │    ~    │   ✓    │     ✗      │
│ Easy Integration  │    ✓     │    ✓    │   ~    │     ✓      │
│ Multiple Models   │    ✓     │    ✓    │   ✓    │     ✓      │
└───────────────────┴──────────┴─────────┴────────┴────────────┘

Legend: ✓ = Excellent, ~ = Good, ✗ = Limited
"""
    print(features)

    print("\n" + "=" * 70)
    print("INSTALLATION SUMMARY")
    print("=" * 70)

    print("\n# Minimal (Hugging Face only)")
    print("pip install transformers torch")

    print("\n# Complete (All approaches)")
    print("pip install transformers torch langchain langchain-community "
          "gpt4all")

    print("\n# Ollama (separate installation)")
    print("# macOS/Linux:")
    print("curl https://ollama.ai/install.sh | sh")
    print("# Windows:")
    print("# Download from https://ollama.ai")

    print("\n" + "=" * 70)
    print("For detailed documentation, see README.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    show_implementations()
