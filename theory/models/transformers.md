# Transformers

The Transformer architecture (2017) revolutionized NLP by replacing recurrence with **Attention**.

## 1. Transformer (Encoder-Decoder)

### Concept
Rely entirely on the **Self-Attention** mechanism.
*   **Self-Attention**: Allows the model to weigh the importance of different words in a sentence relative to each other, regardless of distance.
*   **Parallelization**: Unlike RNNs, Transformers process the entire sequence at once.
*   **Positional Encoding**: Since there is no recurrence, order info is injected via additonal vectors.

### Key Variants
*   **Encoder-Only (BERT)**: Good for understanding/classification.
*   **Decoder-Only (GPT)**: Good for generation.
*   **Encoder-Decoder (T5, Original)**: Good for translation.

### Pros & Cons
*   **Pros**: State-of-the-art performance. highly parallelizable. Scalable to billions of parameters.
*   **Cons**: Quadratic memory complexity with respect to sequence length ($O(N^2)$).

### Use Cases
*   LLMs (ChatGPT, Claude).
*   Translation.
*   Sentiment Analysis.

### Code
*   [PyTorch Transformer](../../models/advanced/transformer/train_pytorch.py)
