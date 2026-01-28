---
layout: default
title: "01 - How LLMs Work"
nav_order: 1
has_children: true
permalink: /01-how-llms-work/
---

# Module 01: How LLMs Work

## Objective

Answer the question: **"What happens inside GPT when I send a prompt?"**

After this module, you should be able to explain the full pipeline:
```
Input Text → Tokens → Embeddings → Attention Layers → FFN → Logits → Sampling → Output Token
```

## Prerequisites

- Basic programming knowledge
- Willingness to think about matrices (no PhD required)

## Topics

| # | Topic | Description |
|---|-------|-------------|
| 01 | [Tokenization to Output](./01-tokenization-to-output.md) | The full journey of a prompt |
| 02 | [Attention Mechanism](./02-attention-mechanism.md) | Why attention is all you need |
| 03 | [Transformer Architecture](./03-transformer-architecture.md) | Putting it all together |

## Checkpoint

After completing this module, you should be able to:

- [ ] Explain what a token is and why we use them
- [ ] Describe what an embedding represents
- [ ] Explain attention in simple terms (what attends to what, and why)
- [ ] Draw a simplified transformer block
- [ ] Explain how the model picks the next token (logits → softmax → sampling)

## Primary Resources

See [resources.md](./resources.md) for curated learning materials.
