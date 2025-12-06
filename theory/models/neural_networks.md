# Neural Networks (Deep Learning)

Deep Learning models use multiple layers of neurons to learn hierarchical representations.

## 1. Multi-Layer Perceptron (MLP) / DNN

### Concept
The simplest form of Deep Learning. Consists of an Input Layer, multiple Hidden Layers, and an Output Layer. Every neuron is fully connected to all neurons in the next layer. It uses non-linear activation functions (ReLU, Sigmoid) to learn complex patterns.

### Pros & Cons
*   **Pros**: Universal function approximator.
*   **Cons**: Loses spatial info (flat inputs). Parameter heavy for large inputs like images.

### Use Cases
*   Simple classification tasks on vectors.
*   Feature mixing after a CNN/RNN backbone.

### Code
*   [PyTorch DNN](../../models/advanced/dnn/train_pytorch.py)

---

## 2. Convolutional Neural Network (CNN)

### Concept
Designed for grid-like data (Images). Uses **Convolutional Layers** that slide filters (kernels) over the input to detect local features (edges, textures). **Pooling Layers** reduce dimensionality and provide translation invariance.

### Pros & Cons
*   **Pros**: Preserves spatial structure. Very parameter efficient compared to MLP for images.
*   **Cons**: Requires large labeled datasets. Fixed input size (usually).

### Use Cases
*   Image Classification (MNIST, ImageNet).
*   Object Detection vs Segmentation.

### Code
*   [PyTorch CNN](../../models/advanced/cnn/train_pytorch.py)

---

## 3. Recurrent Neural Network (RNN/LSTM)

### Concept
Designed for sequence data (Text, Time Series). Processes inputs step-by-step, maintaining a "hidden state" (memory) that captures information from previous steps. **LSTM** (Long Short-Term Memory) fixes the "vanishing gradient" problem of vanilla RNNs using gating mechanisms.

### Pros & Cons
*   **Pros**: Handles variable length inputs. Captures temporal dependencies.
*   **Cons**: Slow sequential processing (hard to parallelize). largely replaced by Transformers for NLP.

### Use Cases
*   Time-series forecasting.
*   Simple text processing.

### Code
*   [PyTorch RNN](../../models/advanced/rnn/train_pytorch.py)
