# 🧠 Deep Learning Library from Scratch

This is a NumPy-based deep learning framework implemented entirely from scratch for educational and experimental purposes. It includes custom-built neural network layers, activation functions, optimizers, cost functions, and a simple model training loop. Convolutional layers and FFT-based convolutions are also supported.

## 📦 Features

- Fully modular design (layers, activations, optimizers, etc.)
- Support for dense and convolutional layers
- Gradient-based backpropagation
- Training loop for models
- Simple implementation of max/average pooling placeholders
- XOR dataset generator for testing
- Custom FFT-based 2D convolution (valid padding)

## 🧱 Project Structure

### 🔧 Core Interfaces

- **`activation_function`**: Base class for all activation functions.
- **`layer`**: Base class for all neural network layers.
- **`optimizer`**: Base class for optimization algorithms.
- **`cost_function`**: Base class for loss functions.

### 🧮 Activation Functions

- `relu`
- `sigmoid`
- `softmax` (partially implemented)

### 📉 Cost Functions

- `MSE` – Mean Squared Error
- `binary_cross_entropy`
- `categorical_cross_entropy` (not implemented)

### 🧠 Layers

- `linear`: Fully-connected dense layer
- `convolutional_2d`: 2D convolutional layer using FFT-based convolution
- `flatten`: Converts multi-dimensional tensors into vectors
- Pooling classes (currently empty skeletons): `max_pool`, `avg_pool2d`, etc.

### ⚙️ Optimizer

- `SGD`: Stochastic Gradient Descent with learning rate

### 🧪 Utility Functions

- `initialize_weights`: Initializes weights based on activation
- `get_activation_functions`, `get_cost_functions`, `get_optimizers`: Factory methods
- `convolution`, `valid_convolution_cnn2d`: FFT-based convolutions with optional padding

## 🧠 Model Training

The `model` class handles forward and backward passes, batching, cost computation, and optimization.

### 🧪 Example: XOR Problem

```python
# XOR Dataset
X = np.random.randn(500, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

# Model definition
trainable_model = model([
    linear(2, 32, "relu"),
    linear(32, 32, "relu"),
    linear(32, 1, "sigmoid")
], "binary_cross_entropy", SGD(0.001), batch_size=100, X=X, Y=Y, epochs=10000)

trainable_model.train()
```

## 📈 Output

- Cost per batch is printed (debug info)
- Final model can be saved (method `save()` is a placeholder)

## 🚧 Incomplete or Placeholder Classes

- Pooling layers (`max_pool`, `avg_pool`, etc.)
- 3D convolution (`convolutional_3d`)
- Softmax gradient
- Categorical cross-entropy implementation
- Model `save()` method

## 🧠 Requirements

- Python 3.x
- NumPy
- Matplotlib (for visualizing XOR dataset)

## 📌 Notes

- Not optimized for speed or memory.
- Intended for understanding low-level mechanics of neural networks.
- Debug prints are left for educational tracing during backpropagation.
