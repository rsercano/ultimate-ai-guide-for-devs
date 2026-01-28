
# Transformer Architecture

## TL;DR

A transformer is a stack of identical blocks. Each block has attention (to mix information between tokens) and a feed-forward network (to process each token individually). Layer normalization and residual connections keep training stable.

## The Full Picture

```
                    Input Embeddings + Position Embeddings
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    │     ┌─────────────────────┐   │
                    │     │   TRANSFORMER BLOCK │   │
                    │     │                     │   │
                    │     │  ┌───────────────┐  │   │
                    └────►│+ │  Layer Norm    │  │   │  Residual
                          │  └───────────────┘  │   │  Connection
                          │         │           │   │
                          │  ┌───────────────┐  │   │
                          │  │  Multi-Head   │  │   │
                          │  │  Attention    │  │   │
                          │  └───────────────┘  │   │
                          │         │           │   │
                    ┌────►│+ ───────┘           │   │  Residual
                    │     │                     │   │  Connection
                    │     │  ┌───────────────┐  │   │
                    │     │  │  Layer Norm    │  │   │
                    │     │  └───────────────┘  │   │
                    │     │         │           │   │
                    │     │  ┌───────────────┐  │   │
                    │     │  │  Feed-Forward │  │   │
                    │     │  │  Network      │  │   │
                    │     │  └───────────────┘  │   │
                    │     │         │           │   │
                    └─────│+ ───────┘           │   │
                          │                     │   │
                          └─────────────────────┘   │
                                    │               │
                                    ▼               │
                          (Repeat N times)          │
                                    │               │
                                    ▼               │
                    ┌───────────────────────────────┘
                    │
                    ▼
              Final Layer Norm
                    │
                    ▼
              Output Projection (to vocabulary size)
                    │
                    ▼
                 Logits
```

## Components Explained

### Feed-Forward Network (FFN)

Applied to each token independently:

```
FFN(x) = GELU(x · W1 + b1) · W2 + b2
```

- Expands dimensionality (768 → 3072), then contracts back
- Where the model "thinks" about each token
- Most parameters live here

### Layer Normalization

```
LayerNorm(x) = γ · (x - μ) / σ + β
```

- Normalizes activations to prevent explosion/vanishing
- Applied before each sub-layer (Pre-LN) in modern architectures
- Learnable γ and β parameters

### Residual Connections

```
output = sublayer(x) + x
```

- Skip connections around every sub-layer
- Allows gradients to flow easily during training
- Enables training very deep networks (96+ layers)

## Scale Parameters

| Model | Layers | Heads | d_model | d_ff | Parameters |
|-------|--------|-------|---------|------|------------|
| GPT-2 Small | 12 | 12 | 768 | 3072 | 117M |
| GPT-2 Medium | 24 | 16 | 1024 | 4096 | 345M |
| GPT-2 Large | 36 | 20 | 1280 | 5120 | 762M |
| GPT-3 | 96 | 96 | 12288 | 49152 | 175B |

## GPT vs BERT vs T5

| Architecture | Type | Attention | Use Case |
|--------------|------|-----------|----------|
| GPT | Decoder-only | Causal (left-to-right) | Generation |
| BERT | Encoder-only | Bidirectional | Understanding |
| T5 | Encoder-Decoder | Both | Seq-to-seq tasks |

Most modern LLMs (GPT-4, Claude, Llama) are decoder-only.

## Key Takeaways

- Transformers are remarkably simple: just attention + FFN, repeated
- Residual connections and layer norm are crucial for training stability
- Scaling up (more layers, wider dimensions) = more capability
- The architecture hasn't changed much since 2017, just the scale

## References

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Minimal GPT implementation
