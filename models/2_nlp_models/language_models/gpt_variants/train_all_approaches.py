"""
Comprehensive LLM Approaches Demo
Demonstrating all alternative LLM approaches with comparisons
"""

import warnings
import time

warnings.filterwarnings("ignore")


def test_huggingface():
    """Test Hugging Face Transformers approach"""
    print("=" * 60)
    print("APPROACH 1: Hugging Face Transformers (Local)")
    print("=" * 60)

    try:
        from transformers import pipeline

        start_time = time.time()

        generator = pipeline("text-generation", model="distilgpt2", device=-1)
        result = generator(
            "Artificial intelligence is",
            max_length=40,
            num_return_sequences=1
        )

        elapsed = time.time() - start_time

        print(f"✓ Model: DistilGPT2 (82M parameters)")
        print(f"✓ Generated: {result[0]['generated_text']}")
        print(f"✓ Time: {elapsed:.2f}s")
        print(f"✓ Device: CPU")

        return {
            "status": "success",
            "time": elapsed,
            "approach": "transformers"
        }

    except ImportError:
        print("⚠ Install: pip install transformers torch")
        return {"status": "missing_package"}
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"status": "error"}


def test_ollama():
    """Test Ollama approach"""
    print("\n" + "=" * 60)
    print("APPROACH 2: Ollama (Local Server)")
    print("=" * 60)

    try:
        import requests

        # Check if server is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = response.json().get("models", [])

        if not models:
            print("⚠ Ollama running but no models installed")
            print("  Install with: ollama pull llama2")
            return {"status": "no_models"}

        model_name = models[0]['name']
        print(f"✓ Using model: {model_name}")

        start_time = time.time()

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Artificial intelligence is",
                "stream": False,
                "options": {"num_predict": 30}
            },
            timeout=30
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Generated: {result['response']}")
            print(f"✓ Time: {elapsed:.2f}s")
            print(f"✓ Device: Local")

            return {
                "status": "success",
                "time": elapsed,
                "approach": "ollama"
            }

    except requests.exceptions.ConnectionError:
        print("⚠ Ollama server not running")
        print("  Start with: ollama serve")
        return {"status": "server_not_running"}
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"status": "error"}


def test_langchain():
    """Test LangChain approach"""
    print("\n" + "=" * 60)
    print("APPROACH 3: LangChain (Multi-Provider)")
    print("=" * 60)

    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=30
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        start_time = time.time()
        response = llm.invoke("Artificial intelligence is")
        elapsed = time.time() - start_time

        print(f"✓ Framework: LangChain")
        print(f"✓ Backend: HuggingFace (GPT-2)")
        print(f"✓ Generated: {response}")
        print(f"✓ Time: {elapsed:.2f}s")

        return {
            "status": "success",
            "time": elapsed,
            "approach": "langchain"
        }

    except ImportError:
        print("⚠ Install: pip install langchain langchain-community "
              "transformers torch")
        return {"status": "missing_package"}
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"status": "error"}


def test_quantized():
    """Test Quantized Models approach"""
    print("\n" + "=" * 60)
    print("APPROACH 4: Quantized Models (GGUF)")
    print("=" * 60)

    try:
        from gpt4all import GPT4All

        print("✓ Using GPT4All (auto-downloads model)")

        start_time = time.time()

        model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
        response = model.generate(
            "Artificial intelligence is",
            max_tokens=30,
            temp=0.7
        )

        elapsed = time.time() - start_time

        print(f"✓ Model: Orca Mini 3B (Q4 quantized)")
        print(f"✓ Generated: {response}")
        print(f"✓ Time: {elapsed:.2f}s")
        print(f"✓ Device: CPU")

        return {
            "status": "success",
            "time": elapsed,
            "approach": "quantized"
        }

    except ImportError:
        print("⚠ Install: pip install gpt4all")
        return {"status": "missing_package"}
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"status": "error"}


def print_comparison(results):
    """Print comparison of all approaches"""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r.get("status") == "success"]

    if successful:
        print("\n┌─────────────────────┬──────────┬─────────────────┐")
        print("│      Approach       │   Time   │     Status      │")
        print("├─────────────────────┼──────────┼─────────────────┤")

        for result in results:
            approach = result.get("approach", "Unknown").title()
            status = result.get("status", "error")
            time_taken = result.get("time", 0)

            if status == "success":
                print(f"│ {approach:19} │ {time_taken:6.2f}s │ "
                      f"{'✓ Success':^15} │")
            else:
                status_msg = status.replace("_", " ").title()
                print(f"│ {approach:19} │    -     │ "
                      f"{status_msg:^15} │")

        print("└─────────────────────┴──────────┴─────────────────┘")

        # Find fastest
        if successful:
            fastest = min(successful, key=lambda x: x["time"])
            print(f"\n✓ Fastest approach: "
                  f"{fastest['approach'].title()} "
                  f"({fastest['time']:.2f}s)")

    print("\n" + "=" * 60)
    print("DETAILED COMPARISON")
    print("=" * 60)

    comparisons = {
        "Hugging Face": {
            "Pros": ["Free", "Many models", "Fast on GPU", "No server needed"],
            "Cons": ["Large memory", "CPU slower", "Downloads needed"],
            "Best for": "Experimentation and development"
        },
        "Ollama": {
            "Pros": ["Easy setup", "Good models", "Local control", "Fast"],
            "Cons": ["Requires server", "Limited models", "Extra process"],
            "Best for": "Development and prototyping"
        },
        "LangChain": {
            "Pros": ["Multi-provider", "Chains", "Memory", "RAG support"],
            "Cons": ["Extra abstraction", "Learning curve", "Dependencies"],
            "Best for": "Complex LLM applications"
        },
        "Quantized": {
            "Pros": ["CPU-friendly", "Small size", "Fast", "No GPU"],
            "Cons": ["Quality loss", "Limited context", "Download needed"],
            "Best for": "Edge devices and laptops"
        }
    }

    for approach, details in comparisons.items():
        print(f"\n{approach}:")
        print(f"  Pros: {', '.join(details['Pros'])}")
        print(f"  Cons: {', '.join(details['Cons'])}")
        print(f"  Best for: {details['Best for']}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("\n🎯 For Production:")
    print("   → Use API-based (OpenAI, Anthropic) for best quality")
    print("   → Use Ollama for privacy-sensitive applications")

    print("\n💻 For Development:")
    print("   → Use Hugging Face for experimentation")
    print("   → Use LangChain for complex workflows")

    print("\n🚀 For Deployment:")
    print("   → Use Quantized models for edge devices")
    print("   → Use Ollama for on-premise solutions")

    print("\n📱 For Laptops/Mobile:")
    print("   → Use Quantized Q4 models")
    print("   → Use GPT4All for simplest setup")


def train():
    """Run all approaches and compare"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE LLM APPROACHES DEMONSTRATION")
    print("=" * 60)
    print("\nTesting all alternative LLM approaches...")
    print("This will take a few minutes...\n")

    results = []

    # Test each approach
    results.append(test_huggingface())
    results.append(test_ollama())
    results.append(test_langchain())
    results.append(test_quantized())

    # Print comparison
    print_comparison(results)

    # Installation guide
    print("\n" + "=" * 60)
    print("INSTALLATION GUIDE")
    print("=" * 60)

    print("\n1. Hugging Face:")
    print("   pip install transformers torch")

    print("\n2. Ollama:")
    print("   # Install from: https://ollama.ai")
    print("   ollama pull llama2")
    print("   ollama serve")

    print("\n3. LangChain:")
    print("   pip install langchain langchain-community")

    print("\n4. Quantized Models:")
    print("   pip install gpt4all")
    print("   # or")
    print("   pip install llama-cpp-python")

    print("\n" + "=" * 60)
    print("QA VALIDATION")
    print("=" * 60)

    successful = [r for r in results if r.get("status") == "success"]
    print(f"\n✓ Approaches tested: {len(results)}")
    print(f"✓ Successful: {len(successful)}")
    print(f"✓ Failed: {len(results) - len(successful)}")

    return {
        "results": results,
        "successful": len(successful),
        "total": len(results)
    }


if __name__ == "__main__":
    train()
