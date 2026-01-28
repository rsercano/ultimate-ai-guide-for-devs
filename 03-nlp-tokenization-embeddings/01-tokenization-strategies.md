---
layout: default
title: Tokenization Strategies
parent: 03 - NLP & Embeddings
nav_order: 1
---

# Tokenization Strategies

## TL;DR

Tokenization splits text into pieces the model can process. Character-level is too granular, word-level can't handle new words. Subword tokenization (BPE, WordPiece) is the sweet spot—common words stay whole, rare words split into pieces.

## The Tokenization Spectrum

```
"unhappiness"

Character:  [u] [n] [h] [a] [p] [p] [i] [n] [e] [s] [s]  → 11 tokens
Subword:    [un] [happiness]  or  [un] [happy] [ness]     → 2-3 tokens
Word:       [unhappiness]                                  → 1 token
```

## Trade-offs

| Approach | Vocabulary Size | Sequence Length | OOV Handling | Semantic Meaning |
|----------|-----------------|-----------------|--------------|------------------|
| Character | ~100 | Very long | Perfect | None |
| Subword | 30k-100k | Medium | Good | Some |
| Word | 100k+ | Short | Poor (OOV) | Best |

**OOV = Out of Vocabulary**

LLMs use subword tokenization as the optimal trade-off.

## Byte Pair Encoding (BPE)

The most common algorithm. Used by GPT models.

### Training Algorithm

```
1. Start with character vocabulary
2. Count all adjacent pairs in training data
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary size reached
```

### Example

```
Corpus: "low lower lowest"

Initial: [l][o][w][ ][l][o][w][e][r][ ][l][o][w][e][s][t]

Step 1: Most frequent pair = (l, o) → Merge to [lo]
        [lo][w][ ][lo][w][e][r][ ][lo][w][e][s][t]

Step 2: Most frequent pair = (lo, w) → Merge to [low]
        [low][ ][low][e][r][ ][low][e][s][t]

Step 3: Most frequent pair = (e, r) → Merge to [er]
        [low][ ][low][er][ ][low][e][s][t]

... continue until vocab size reached
```

### Encoding (at inference)

Apply learned merges greedily:

```
"lowest" → [low][est]  (if [low] and [est] were learned)
"xyz"    → [x][y][z]   (falls back to characters if unknown)
```

## WordPiece

Similar to BPE, used by BERT.

Difference: Instead of frequency, scores merges by likelihood improvement.

```
Score(ab) = freq(ab) / (freq(a) × freq(b))
```

Tokens start with `##` when not at word start:

```
"unhappiness" → ["un", "##happiness"]
```

## SentencePiece

Language-agnostic tokenization (no pre-tokenization on spaces).

Used by: T5, LLaMA, many multilingual models

Benefits:
- Handles any language/script
- Treats space as a regular character
- More consistent across languages

## Token Counts Matter

| Model | Context Length | Approximate Words |
|-------|---------------|-------------------|
| GPT-3.5 | 4k tokens | ~3k words |
| GPT-4 | 8k-128k tokens | ~6k-100k words |
| Claude | 100k-200k tokens | ~75k-150k words |

Rule of thumb: **1 token ≈ 0.75 words** (English)

## Tokenization Gotchas

### 1. Whitespace Sensitivity

```
"Hello World" → ["Hello", " World"]      # Space attached to World
" Hello"      → [" Hello"]               # Leading space = different token
```

### 2. Numbers

```
"2024" → ["20", "24"] or ["2", "024"]    # Not atomic
```

### 3. Code/Special Characters

```
"def foo():" → ["def", " foo", "()", ":"]  # Varies by tokenizer
```

### 4. Multilingual

```
"こんにちは" → Many tokens in English-centric tokenizers
             → Fewer tokens in multilingual tokenizers
```

## Key Takeaways

- Subword tokenization balances vocabulary size vs sequence length
- BPE is most common, learned from training data
- Token count affects cost, latency, and context limits
- Tokenization quirks can affect model behavior

## References

- [Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [HuggingFace — Tokenizers](https://huggingface.co/docs/tokenizers/)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
