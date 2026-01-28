---
layout: default
title: Semantic Similarity
parent: "03 - NLP & Embeddings"
nav_order: 3
---

# Semantic Similarity

## TL;DR

Semantic similarity measures how close two pieces of text are in meaning using their embeddings. Cosine similarity is the standard metric. This powers search, RAG, deduplication, and recommendation systems.

## Cosine Similarity

The standard way to compare embeddings:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product (sum of element-wise multiplication)
- ||A|| = magnitude (sqrt of sum of squares)
```

### Intuition

Cosine similarity measures the angle between two vectors:

```
        ▲ B
       ╱│
      ╱ │
     ╱θ │         cosine(θ) = similarity
    ╱───┼────▶ A
   
θ = 0°   → cos(θ) = 1.0   (identical direction)
θ = 90°  → cos(θ) = 0.0   (unrelated)
θ = 180° → cos(θ) = -1.0  (opposite meaning)
```

### Example

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Hypothetical embeddings
cat = np.array([0.8, 0.5, 0.1])
dog = np.array([0.7, 0.6, 0.2])
car = np.array([0.1, 0.2, 0.9])

print(cosine_similarity(cat, dog))  # ~0.98 (very similar)
print(cosine_similarity(cat, car))  # ~0.35 (not similar)
```

## Other Distance Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Cosine | 1 - cos(θ) | Normalized text (most common) |
| Euclidean | sqrt(Σ(a-b)²) | When magnitude matters |
| Dot Product | Σ(a×b) | When vectors are normalized |
| Manhattan | Σ\|a-b\| | High-dimensional sparse data |

For normalized embeddings, **cosine similarity = dot product**.

## Semantic Search Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING (Offline)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Documents → Chunk → Embed → Store in Vector DB            │
│                                                              │
│   "Machine learning is..."  →  [0.2, 0.5, ...]  →  Pinecone │
│   "Neural networks can..."  →  [0.3, 0.4, ...]  →  Weaviate │
│   "Deep learning uses..."   →  [0.2, 0.6, ...]  →  Chroma   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERYING (Online)                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Query → Embed → Find nearest neighbors → Return results   │
│                                                              │
│   "How do neural nets work?"                                │
│            ↓                                                 │
│   [0.25, 0.45, ...]                                         │
│            ↓                                                 │
│   Compare with all stored vectors                           │
│            ↓                                                 │
│   Return top-k most similar documents                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Vector Databases

Store embeddings and enable fast similarity search:

| Database | Type | Best For |
|----------|------|----------|
| Pinecone | Managed | Production, scale |
| Weaviate | Self-host/managed | Hybrid search |
| Chroma | Local/embedded | Prototyping |
| Qdrant | Self-host | High performance |
| pgvector | PostgreSQL extension | Existing Postgres |
| FAISS | Library | Custom solutions |

### Approximate Nearest Neighbors (ANN)

Exact search is O(n). ANN trades accuracy for speed:

- **HNSW** (Hierarchical Navigable Small World): Graph-based, most common
- **IVF** (Inverted File Index): Cluster-based
- **Product Quantization**: Compression-based

## Chunking Strategies

Before embedding, split documents into chunks:

```
Document (10,000 tokens)
         ↓
    ┌────┴────┐
    │ Chunking │
    └────┬────┘
         ↓
┌─────┬─────┬─────┬─────┐
│Chunk│Chunk│Chunk│Chunk│  (512 tokens each)
│  1  │  2  │  3  │  4  │
└─────┴─────┴─────┴─────┘
```

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| Fixed size | Every N tokens | Simple, may break sentences |
| Sentence | Split on sentences | Respects boundaries, variable size |
| Paragraph | Split on paragraphs | Keeps context, larger chunks |
| Semantic | Split on topic change | Best quality, complex |
| Overlap | Chunks share some content | Preserves context across splits |

## Reranking

Two-stage retrieval for better accuracy:

```
Query → Vector Search (fast, top-100) → Reranker (slow, accurate) → Top-10
```

Rerankers (cross-encoders) are more accurate but too slow for first-pass retrieval.

## Key Takeaways

- Cosine similarity is the standard metric for embedding comparison
- Vector databases enable fast approximate nearest neighbor search
- Chunking strategy significantly impacts retrieval quality
- Two-stage retrieval (retrieve then rerank) improves accuracy

## References

- [Pinecone — What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard) — Embedding model comparison
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
