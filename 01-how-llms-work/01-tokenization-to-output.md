---
layout: default
title: Tokenization to Output
parent: "01 - How LLMs Work"
nav_order: 1
---

# Tokenization to Output: The Full Journey

## TL;DR

When you send "Hello, how are you?" to GPT, it gets chopped into tokens, converted to numbers, passed through many transformer layers, and produces a probability distribution over all possible next tokens. One token is sampled, appended, and the process repeats.

## The Pipeline

```
"Hello, how are you?"
        │
        ▼
┌───────────────────┐
│   TOKENIZATION    │  Text → Token IDs
│   "Hello" → 9906  │  
│   "," → 11        │
│   " how" → 703    │
│   ...             │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    EMBEDDING      │  Token ID → Vector (e.g., 768 dimensions)
│    9906 → [0.2,   │  Each token becomes a point in high-dim space
│            -0.5,  │
│            ...]   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  POSITION EMBED   │  Add position information
│  Token 1, 2, 3... │  Model needs to know word order
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  TRANSFORMER      │  Multiple layers of:
│  LAYERS (x N)     │  - Self-Attention
│                   │  - Feed-Forward Network
│                   │  - Layer Normalization
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  OUTPUT HEAD      │  Final vector → Logits for all vocab tokens
│  (Linear Layer)   │  50,000+ scores, one per possible token
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  SOFTMAX          │  Logits → Probabilities
│                   │  All scores become 0-1, sum to 1
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  SAMPLING         │  Pick next token based on probabilities
│                   │  (temperature, top-p, top-k affect this)
└───────────────────┘
        │
        ▼
    Next Token
```

## Key Concepts

### Why Tokens, Not Characters?

- Characters are too granular (slow, no semantic meaning)
- Words are too rigid (can't handle new words, typos)
- Tokens are a middle ground: subword units learned from data
- Common words = 1 token, rare words = multiple tokens

### Why Embeddings?

- Computers need numbers, not text
- Embeddings capture semantic meaning
- Similar words → similar vectors → model can generalize

### Why Position Embeddings?

- Attention has no inherent sense of order
- "Dog bites man" vs "Man bites dog" would be identical without position info
- Position embeddings add sequence awareness

### What Are Logits?

- Raw, unnormalized scores for each vocabulary token
- Higher logit = model thinks this token is more likely next
- Softmax converts these to probabilities

## Key Takeaways

- LLMs are next-token predictors, nothing more
- The entire architecture serves one goal: produce good probability distributions over next tokens
- Autoregressive generation: output becomes input, repeat until done

## References

- [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Full walkthrough
- [HuggingFace — Tokenizers](https://huggingface.co/docs/tokenizers/) — Deep dive on tokenization
