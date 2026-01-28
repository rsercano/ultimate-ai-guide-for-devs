
# Optimization and Regularization

## TL;DR

Learning rate controls step size. Adam is the default optimizer. Overfitting happens when the model memorizes training data. Regularization techniques (dropout, weight decay) force generalization.

## Learning Rate

The most important hyperparameter.

```
Too High                    Just Right                  Too Low
    │                           │                          │
Loss│    ╱╲   ╱╲               │╲                         │
    │   ╱  ╲ ╱  ╲              │ ╲                        │
    │  ╱    ╲    ╲             │  ╲                       │╲
    │ ╱           ╲            │   ╲_____                 │ ╲
    │╱             ╲           │                          │  ╲
    └───────────────▶          └───────────▶              │   ╲
        Diverges!              Converges fast             └────╲──▶
                                                          Very slow
```

### Learning Rate Schedules

Learning rate often decreases during training:

| Schedule | Description | Use Case |
|----------|-------------|----------|
| Constant | Never changes | Baseline |
| Step Decay | Divide by 10 at epochs X, Y | Classic CV |
| Cosine | Smooth decrease to 0 | Transformers |
| Warmup | Start low, increase, then decay | LLM training |

## Optimizers

Beyond vanilla gradient descent:

### SGD with Momentum

```
velocity = β × velocity + gradient
weight -= learning_rate × velocity
```

- Accumulates momentum like a ball rolling downhill
- Smooths out noisy gradients

### Adam (Adaptive Moment Estimation)

```
m = β1 × m + (1-β1) × gradient           # First moment (momentum)
v = β2 × v + (1-β2) × gradient²          # Second moment (velocity)
weight -= learning_rate × m / (√v + ε)
```

- Adapts learning rate per-parameter
- Default choice for most deep learning

### AdamW

- Adam + weight decay (properly decoupled)
- Standard for transformer training

## Overfitting

When the model memorizes training data instead of learning patterns:

```
        Error
          ▲
          │     Training Error
          │     ─────────────────────▶  (keeps decreasing)
          │
          │         Validation Error
          │     ────────╲
          │              ╲   ╱────────▶  (starts increasing)
          │               ╲ ╱
          │                ✕
          │           Overfitting starts here
          └─────────────────────────────────▶ Training Time
```

### Signs of Overfitting

- Training loss very low, validation loss high
- Model performs great on training data, poorly on new data
- Large gap between train and validation metrics

## Regularization Techniques

### 1. Dropout

Randomly zero out neurons during training:

```
Training:
[1.0] [0.0] [1.0] [0.0] [1.0]   ← 50% dropout
   │         │         │
   └────┬────┴────┬────┘
        │         │
     [output] [output]

Inference:
[1.0] [1.0] [1.0] [1.0] [1.0]   ← All neurons active (scaled)
```

Forces redundancy, prevents co-adaptation.

### 2. Weight Decay (L2 Regularization)

Penalize large weights:

```
loss = original_loss + λ × Σ(weights²)
```

Encourages smaller, more distributed weights.

### 3. Data Augmentation

Create variations of training data (more relevant for images, but text has equivalents like back-translation).

### 4. Early Stopping

Stop training when validation loss stops improving.

## Batch Size

| Batch Size | Pros | Cons |
|------------|------|------|
| Small (16-32) | Better generalization, less memory | Noisy gradients, slow |
| Large (256-1024+) | Faster, stable gradients | Worse generalization, high memory |

For LLMs, large batches are common but require careful tuning.

## Key Takeaways

- Learning rate is the most critical hyperparameter
- Adam/AdamW is the default optimizer for transformers
- Overfitting = good on training data, bad on new data
- Regularization (dropout, weight decay) forces generalization
- Always monitor validation loss, not just training loss

## References

- [StatQuest — Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [Why Momentum Really Works](https://distill.pub/2017/momentum/)
- [Decoupled Weight Decay (AdamW Paper)](https://arxiv.org/abs/1711.05101)
