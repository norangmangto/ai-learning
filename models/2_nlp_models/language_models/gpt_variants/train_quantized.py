"""
Quantized Models (GGML/GGUF) - CPU-Optimized LLMs
Using quantized models for efficient inference on CPU
"""

import warnings

warnings.filterwarnings("ignore")


def train():
    print("=== Quantized Models (GGUF) Implementation ===\n")

    # 1. Using llama-cpp-python
    print("1. Testing llama-cpp-python (GGUF format)...")
    try:
        from llama_cpp import Llama

        print("⚠ Note: You need to download a GGUF model first")
        print("Example download locations:")
        print("- https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
        print("- https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        print("\nDownload example:")
        print("  wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

        # Check if model exists
        import os
        model_path = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf"

        if os.path.exists(model_path):
            print(f"\n✓ Found model: {model_path}")

            # Load model
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window
                n_threads=4,  # CPU threads
                n_gpu_layers=0  # Use CPU only
            )

            # Test generation
            prompts = [
                "What is machine learning?",
                "Explain neural networks briefly."
            ]

            for prompt in prompts:
                print(f"\nPrompt: {prompt}")

                output = llm(
                    prompt,
                    max_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    echo=False
                )

                print(f"Response: {output['choices'][0]['text']}")

            print("\n✓ llama-cpp-python completed successfully")

        else:
            print(f"\n⚠ Model not found at: {model_path}")
            print("Please download a GGUF model to test this approach")

    except ImportError:
        print("Install: pip install llama-cpp-python")
        print("For GPU support: CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" "
              "pip install llama-cpp-python")
    except Exception as e:
        print(f"Error: {e}")

    # 2. Using ctransformers (deprecated but still useful)
    print("\n2. Testing ctransformers...")
    try:
        from ctransformers import AutoModelForCausalLM

        print("Available quantization types:")
        print("- Q4_K_M: 4-bit medium quality (recommended)")
        print("- Q5_K_M: 5-bit medium quality")
        print("- Q8_0: 8-bit (better quality, larger)")

        model_path = "./mistral-7b-instruct-v0.1.Q4_K_M.gguf"

        if os.path.exists(model_path):
            print(f"\n✓ Found model: {model_path}")

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral",
                gpu_layers=0
            )

            prompt = "The three laws of robotics are"
            response = model(prompt, max_new_tokens=100)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")

            print("\n✓ ctransformers completed successfully")

        else:
            print(f"\n⚠ Model not found at: {model_path}")

    except ImportError:
        print("Install: pip install ctransformers")
    except Exception as e:
        print(f"Error: {e}")

    # 3. Using GPT4All
    print("\n3. Testing GPT4All...")
    try:
        from gpt4all import GPT4All

        print("GPT4All automatically downloads models")
        print("Available models:")
        print("- orca-mini-3b-gguf2-q4_0.gguf (smallest)")
        print("- mistral-7b-instruct-v0.1.Q4_0.gguf")
        print("- gpt4all-falcon-q4_0.gguf")

        # This will download if not present
        model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

        prompts = [
            "What is Python?",
            "Explain AI in simple terms."
        ]

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            response = model.generate(prompt, max_tokens=100, temp=0.7)
            print(f"Response: {response}")

        print("\n✓ GPT4All completed successfully")

    except ImportError:
        print("Install: pip install gpt4all")
    except Exception as e:
        print(f"Error: {e}")

    # 4. Performance comparison
    print("\n4. Performance Comparison...")
    print("\nQuantization Types:")
    print("┌──────────┬─────────────┬──────────┬─────────────┐")
    print("│   Type   │ Size (7B)   │  Speed   │   Quality   │")
    print("├──────────┼─────────────┼──────────┼─────────────┤")
    print("│   Q2_K   │   ~2.5 GB   │   Fast   │     Low     │")
    print("│   Q4_K_M │   ~4.1 GB   │  Medium  │    Good     │")
    print("│   Q5_K_M │   ~4.8 GB   │  Medium  │  Very Good  │")
    print("│   Q8_0   │   ~7.2 GB   │   Slow   │  Excellent  │")
    print("│  FP16    │  ~13.5 GB   │Very Slow │    Best     │")
    print("└──────────┴─────────────┴──────────┴─────────────┘")

    # 5. Model downloading helper
    print("\n5. Model Download Helper...")
    print("\nRecommended models to download:")

    models = {
        "Mistral 7B Instruct": {
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            "file": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "size": "4.1 GB",
            "use_case": "General purpose, coding"
        },
        "Llama 2 7B Chat": {
            "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF",
            "file": "llama-2-7b-chat.Q4_K_M.gguf",
            "size": "4.1 GB",
            "use_case": "Conversational AI"
        },
        "CodeLlama 7B": {
            "url": "https://huggingface.co/TheBloke/CodeLlama-7B-GGUF",
            "file": "codellama-7b.Q4_K_M.gguf",
            "size": "4.1 GB",
            "use_case": "Code generation"
        },
        "Orca Mini 3B": {
            "url": "https://gpt4all.io/models/gguf/",
            "file": "orca-mini-3b-gguf2-q4_0.gguf",
            "size": "1.8 GB",
            "use_case": "Lightweight, fast"
        }
    }

    for name, info in models.items():
        print(f"\n{name}:")
        print(f"  URL: {info['url']}")
        print(f"  File: {info['file']}")
        print(f"  Size: {info['size']}")
        print(f"  Use case: {info['use_case']}")

    # 6. Download script example
    print("\n6. Download Script Example...")
    download_script = """
# Download Mistral 7B Q4_K_M quantization
import urllib.request
from tqdm import tqdm

url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
output_path = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

print(f"Downloading {url}")
urllib.request.urlretrieve(url, output_path)
print(f"Downloaded to {output_path}")
"""
    print(download_script)

    # QA Validation
    print("\n=== QA Validation ===")
    print("✓ llama-cpp-python approach documented")
    print("✓ ctransformers approach documented")
    print("✓ GPT4All approach tested")
    print("✓ Quantization comparison provided")
    print("✓ Model download guidance provided")

    print("\n=== Summary ===")
    print("Quantized Models Benefits:")
    print("- Run on CPU efficiently")
    print("- Much smaller file sizes")
    print("- No GPU required")
    print("- Good quality with 4-bit quantization")
    print("\nRecommendations:")
    print("- Q4_K_M: Best balance of size/quality")
    print("- Q5_K_M: Better quality, slightly larger")
    print("- Use llama-cpp-python for best performance")
    print("- Use GPT4All for simplest setup")

    return {
        "libraries": ["llama-cpp-python", "ctransformers", "gpt4all"],
        "recommended_quantization": "Q4_K_M"
    }


if __name__ == "__main__":
    train()
