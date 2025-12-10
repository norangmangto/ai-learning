# Keras/TensorFlow Model Training

This directory contains TensorFlow/Keras implementations of deep learning models.

## 📋 Models Implemented

Models in this directory typically have PyTorch counterparts also available.

### Running Models

```bash
# Basic execution
python train_tensorflow.py

# With specific parameters
python train_tensorflow.py --epochs 50 --batch-size 32
```

### Framework Comparison

| Aspect | PyTorch | TensorFlow |
|--------|---------|-----------|
| **Syntax** | Pythonic | Functional |
| **Debugging** | Eager execution | Graph/Eager |
| **Performance** | CPU/GPU | CPU/GPU/TPU |
| **Production** | Good | Excellent (TFLite, TFServing) |
| **Learning Curve** | Easier | Steeper |

## 📚 Structure

- Each file typically includes:
  - Model architecture
  - Data loading and preprocessing
  - Training loop with validation
  - Evaluation metrics
  - Visualization of results

## 🔗 Equivalent PyTorch Models

For PyTorch implementations, see `train_pytorch.py` in the same directory.

## 💡 Key Differences from PyTorch

1. **Functional API vs. Imperative**: TensorFlow/Keras is more declarative
2. **Graph Mode vs. Eager**: TensorFlow builds computation graphs
3. **Automatic Differentiation**: Similar but different implementation
4. **Mixed Precision**: TensorFlow has better native support

## 📊 Performance Notes

- Generally similar performance between frameworks
- TensorFlow may be faster on TPUs
- PyTorch often easier to optimize for specific use cases

---

**Status:** ✅ TensorFlow implementations available
**Last Updated:** December 2024
