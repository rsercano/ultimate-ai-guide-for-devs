---
layout: default
title: Neural Network Basics
parent: 02 - Training & Learning
nav_order: 1
---

# Neural Network Basics

## TL;DR

A neural network is a function made of layers. Each layer transforms its input with weights, biases, and non-linear activations. The forward pass computes the output; training adjusts the weights to minimize error.

## What's a Neuron?

A single neuron computes:

```
output = activation(Σ(weight_i × input_i) + bias)
```

- **Inputs**: Numbers from previous layer (or raw data)
- **Weights**: Learned multipliers (how important each input is)
- **Bias**: Learned offset (baseline activation)
- **Activation**: Non-linear function (introduces non-linearity)

## What's a Layer?

Multiple neurons grouped together:

```
Input (3 values)        Hidden Layer (4 neurons)        Output (2 values)
     │                          │                            │
    [x1] ─────┬────────────→ [n1] ────┬───────────────→ [o1]
     │        │                │      │                  │
    [x2] ────┼────────────→ [n2] ────┼───────────────→ [o2]
     │        │                │      │
    [x3] ─────┴────────────→ [n3] ────┘
                               │
                             [n4]
```

Each neuron in a layer connects to ALL neurons in the previous layer (fully connected).

## The Forward Pass

Data flows through the network:

```python
# Pseudocode for one layer
def forward(x, W, b):
    z = x @ W + b          # Linear transformation
    a = activation(z)       # Non-linearity
    return a

# Full network
h1 = forward(input, W1, b1)     # Input → Hidden 1
h2 = forward(h1, W2, b2)        # Hidden 1 → Hidden 2
output = forward(h2, W3, b3)    # Hidden 2 → Output
```

## Why Non-Linearity?

Without activation functions, stacking layers is pointless:

```
Layer1: y = W1 × x + b1
Layer2: z = W2 × y + b2

Combined: z = W2 × (W1 × x + b1) + b2
            = (W2 × W1) × x + (W2 × b1 + b2)
            = W' × x + b'  ← Just another linear function!
```

Non-linear activations let networks learn complex patterns.

## Common Activations

| Function | Formula | Use |
|----------|---------|-----|
| ReLU | max(0, x) | Hidden layers (default choice) |
| GELU | x × Φ(x) | Transformers (smooth ReLU) |
| Sigmoid | 1/(1+e^-x) | Binary classification output |
| Softmax | e^xi / Σe^xj | Multi-class classification output |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | Older networks, some RNNs |

## Parameters vs Hyperparameters

| Parameters | Hyperparameters |
|------------|-----------------|
| Learned during training | Set before training |
| Weights and biases | Learning rate, batch size |
| What the model knows | How the model learns |

## Key Takeaways

- Neural networks are compositions of simple operations
- The forward pass is just matrix multiplications + non-linearities
- Weights and biases are the learned parameters
- Non-linearity is essential for learning complex patterns

## References

- [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Karpathy — micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
