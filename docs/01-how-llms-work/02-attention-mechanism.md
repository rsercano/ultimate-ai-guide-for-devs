
# Attention Mechanism

## TL;DR

Attention allows each token to "look at" all other tokens and decide which ones are relevant for its current context. It's how "it" in "The cat sat on the mat because it was tired" knows to attend to "cat" not "mat."

## The Core Idea

Every token asks: **"Which other tokens should I pay attention to?"**

```
Query (Q):  "What am I looking for?"
Key (K):    "What do I contain?"
Value (V):  "What information do I provide?"

Attention = softmax(Q · K^T / √d) · V
```

## Step by Step

### 1. Create Q, K, V

Each token's embedding is projected into three vectors:

```
Token Embedding (768 dim)
    │
    ├──→ W_Q ──→ Query  (64 dim)
    ├──→ W_K ──→ Key    (64 dim)
    └──→ W_V ──→ Value  (64 dim)
```

### 2. Compute Attention Scores

Each query is compared against all keys:

```
Score(i,j) = Query_i · Key_j

High score = Token i should pay attention to Token j
```

### 3. Scale and Softmax

```
Scaled Scores = Scores / √d_k

Attention Weights = softmax(Scaled Scores)
```

- Scaling prevents extreme values before softmax
- Softmax makes weights sum to 1 (probability distribution)

### 4. Weighted Sum of Values

```
Output_i = Σ (Attention_weight_ij × Value_j)
```

Each token's output is a weighted combination of all value vectors.

## Multi-Head Attention

Instead of one attention, run multiple in parallel:

```
┌─────────────────────────────────────────┐
│           Multi-Head Attention           │
├─────────────────────────────────────────┤
│  Head 1    Head 2    Head 3    Head 4   │
│    │         │         │         │      │
│    └─────────┴─────────┴─────────┘      │
│                   │                      │
│              Concatenate                 │
│                   │                      │
│              Linear Layer                │
└─────────────────────────────────────────┘
```

**Why multiple heads?**
- Different heads can learn different patterns
- One head: syntax relationships
- Another head: semantic relationships
- Another head: positional patterns

## Causal (Masked) Attention

In GPT-style models, tokens can only attend to previous tokens:

```
Attention Matrix (4 tokens):

         Token1  Token2  Token3  Token4
Token1   [  ✓      ✗       ✗       ✗   ]
Token2   [  ✓      ✓       ✗       ✗   ]
Token3   [  ✓      ✓       ✓       ✗   ]
Token4   [  ✓      ✓       ✓       ✓   ]

✓ = can attend
✗ = masked (set to -infinity before softmax)
```

This prevents the model from "cheating" by looking at future tokens.

## Intuition Check

Think of attention as a database query:
- **Query**: Your search term
- **Keys**: Tags/labels on all documents
- **Values**: The actual content of documents
- **Attention**: How relevant each document is to your search
- **Output**: A weighted blend of relevant documents

## Key Takeaways

- Attention is a learned, differentiable way to route information
- Multi-head attention captures different types of relationships
- Causal masking ensures autoregressive generation works
- The Q/K/V projection matrices are the learned parameters

## References

- [3Blue1Brown — Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original paper
