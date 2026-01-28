---
layout: default
title: Loss and Backpropagation
parent: "02 - Training & Learning"
nav_order: 2
---

# Loss and Backpropagation

## TL;DR

Loss functions measure how wrong the model is. Backpropagation efficiently computes how to adjust each weight to reduce that loss. It's just the chain rule from calculus, applied automatically.

## The Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. Forward Pass    Data → Network → Prediction            │
│           │                                                  │
│           ▼                                                  │
│   2. Compute Loss    Compare prediction to target            │
│           │                                                  │
│           ▼                                                  │
│   3. Backward Pass   Compute gradients for all weights       │
│           │                                                  │
│           ▼                                                  │
│   4. Update Weights  weights -= learning_rate × gradients    │
│           │                                                  │
│           └──────────────► Repeat                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Loss Functions

Loss = how far off are we?

### For LLMs: Cross-Entropy Loss

```
Loss = -Σ target_i × log(predicted_i)
```

In practice for language modeling:
- Target: one-hot vector (1 for correct next token, 0 elsewhere)
- Prediction: probability distribution over vocabulary
- Loss: -log(probability of correct token)

**Intuition**: If model says correct token has 90% probability, loss is low. If it says 1%, loss is high.

### Other Common Losses

| Task | Loss Function | What It Measures |
|------|---------------|------------------|
| Classification | Cross-Entropy | Probability error |
| Regression | MSE (Mean Squared Error) | Squared distance |
| Ranking | Contrastive/Triplet | Relative ordering |

## Backpropagation

The key insight: compute gradients layer by layer, from output back to input.

### The Chain Rule

If `y = f(g(x))`, then:

```
dy/dx = dy/dg × dg/dx
```

Applied to neural networks:

```
dLoss/dW1 = dLoss/dOutput × dOutput/dHidden × dHidden/dW1
```

### Visual Example

```
x ──→ [×W1] ──→ [ReLU] ──→ [×W2] ──→ [Softmax] ──→ Loss
                                                     │
    ◄────────────────────────────────────────────────┘
                   Gradients flow backward
```

Each operation "remembers" what it needs to compute its local gradient.

### Automatic Differentiation

Modern frameworks (PyTorch, TensorFlow) do this automatically:

```python
# PyTorch example
output = model(input)
loss = criterion(output, target)
loss.backward()  # Computes all gradients automatically

# Gradients now stored in each parameter:
# model.layer1.weight.grad
# model.layer1.bias.grad
# etc.
```

## Gradient Descent

Once we have gradients, update weights:

```
new_weight = old_weight - learning_rate × gradient
```

**Why subtract?** Gradient points toward increasing loss. We want to decrease it.

```
           Loss
            ▲
            │      We're here
            │         ↓
      ─ ─ ─│─ ─ ✕ ─ ─ ─ ─
           │    ╲
           │     ╲  Gradient points up
           │      ╲ (loss increases)
           │       ╲
      ─────┼────────●────▶ Weight
           │     Minimum
```

## Key Takeaways

- Loss quantifies how wrong the model is
- Backprop efficiently computes all gradients in one backward pass
- The chain rule is the mathematical foundation
- Gradient descent follows the negative gradient to reduce loss

## References

- [3Blue1Brown — Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [Karpathy — micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Build backprop from scratch
