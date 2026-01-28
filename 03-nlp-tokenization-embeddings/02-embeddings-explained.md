
# Embeddings Explained

## TL;DR

Embeddings are learned vector representations where similar meanings are nearby in vector space. They convert discrete tokens into continuous numbers that neural networks can process and learn from.

## Why Embeddings?

Neural networks need numbers, but language is symbolic:

```
"cat" → ??? → Neural Network
```

One-hot encoding doesn't work at scale:

```
Vocabulary: [cat, dog, the, is, ...]  (50,000 words)

"cat" → [1, 0, 0, 0, 0, 0, ... 0]     # 50,000-dim sparse vector
"dog" → [0, 1, 0, 0, 0, 0, ... 0]     # No relationship captured!
```

Problems:
- Huge, sparse vectors
- No semantic relationship (cat and dog look equally different as cat and the)
- Can't generalize

## Embeddings Capture Meaning

Instead, learn a dense representation:

```
"cat" → [0.2, -0.5, 0.8, 0.1, ...]    # 768 dimensions
"dog" → [0.3, -0.4, 0.7, 0.2, ...]    # Similar to cat!
"the" → [-0.1, 0.9, -0.2, 0.5, ...]   # Very different
```

Similar meanings → similar vectors → similar model behavior.

## Types of Embeddings

### 1. Token Embeddings (Static per Token)

Each token ID maps to a fixed vector:

```python
embedding_matrix = torch.nn.Embedding(vocab_size, embedding_dim)
# Shape: [50000, 768]

token_id = 42
vector = embedding_matrix[token_id]  # Shape: [768]
```

This is what's inside GPT's first layer.

### 2. Contextual Embeddings (Dynamic)

The same word gets different vectors based on context:

```
"The bank by the river"     → "bank" = [0.2, 0.5, ...]  (riverbank)
"I went to the bank"        → "bank" = [0.8, -0.3, ...] (financial)
```

Produced by running text through transformer layers (BERT, GPT).

### 3. Sentence/Document Embeddings

Entire text compressed to one vector:

```
"The quick brown fox jumps over the lazy dog"
        ↓
    [0.1, -0.3, 0.7, ..., 0.2]  (768 or 1536 dims)
```

Used for: search, similarity, clustering, RAG retrieval.

## How Embeddings Are Learned

### During LLM Training

Embeddings are learned end-to-end:

```
Token ID → Embedding Layer → Transformer → Predict Next Token
     ↑                                            │
     └────────── Gradients update embeddings ─────┘
```

The model learns embeddings that help predict the next token accurately.

### Word2Vec (Historical)

Classic approach: predict word from context (CBOW) or context from word (Skip-gram).

```
"The cat sat on the ___"
                     ↓
               predict: mat

This forces "mat", "floor", "couch" to have similar embeddings
(they appear in similar contexts)
```

### Contrastive Learning (Modern)

For sentence embeddings, learn that:
- Similar sentences → close vectors
- Different sentences → far vectors

```
Loss = -log(similarity(anchor, positive) / Σ similarity(anchor, all))
```

## Embedding Dimensions

| Model | Dimensions | Use Case |
|-------|------------|----------|
| Word2Vec | 100-300 | Historical |
| BERT | 768 | Understanding |
| GPT-3 | 12,288 | Generation |
| Ada-002 | 1,536 | OpenAI embeddings |
| Cohere | 1,024-4,096 | Search |

More dimensions = more capacity, but diminishing returns after a point.

## Visualizing Embeddings

High dimensions are hard to visualize. Use dimensionality reduction:

```
768 dimensions → PCA or t-SNE → 2 dimensions → Plot

Result: Clusters of related concepts appear!
- Programming languages cluster together
- Animals cluster together
- Etc.
```

## Key Takeaways

- Embeddings convert symbols to numbers while preserving meaning
- Similar meanings = nearby vectors
- Contextual embeddings (from transformers) capture word sense
- Embeddings are learned, not hand-crafted
- They're the foundation of all modern NLP

## References

- [Jay Alammar — The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [Embedding Projector (TensorFlow)](https://projector.tensorflow.org/) — Visualize embeddings
