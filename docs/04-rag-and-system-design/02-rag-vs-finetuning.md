
# RAG vs Fine-tuning

## TL;DR

RAG adds knowledge at inference time (dynamic). Fine-tuning bakes knowledge into weights (static). Use RAG for facts/documents, fine-tuning for style/behavior. Often, you use both.

## Quick Decision Framework

```
Do you need to...

├── Add new factual knowledge?
│   └── → RAG (facts can be updated without retraining)
│
├── Change how the model writes/responds?
│   └── → Fine-tuning (style, format, persona)
│
├── Teach domain-specific terminology?
│   └── → Fine-tuning (vocabulary, patterns)
│
├── Work with frequently updating data?
│   └── → RAG (no retraining needed)
│
├── Reduce hallucinations with citations?
│   └── → RAG (source attribution built-in)
│
├── Minimize latency?
│   └── → Fine-tuning (no retrieval step)
│
└── Maintain data privacy with citations?
    └── → RAG (data never in model weights)
```

## Detailed Comparison

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Knowledge type** | Facts, documents | Behavior, style |
| **Update frequency** | Real-time | Requires retraining |
| **Cost** | Inference + retrieval | Training + inference |
| **Latency** | Higher (retrieval step) | Lower |
| **Hallucination** | Can cite sources | May still hallucinate |
| **Data privacy** | Data stays external | Data baked into weights |
| **Complexity** | Pipeline complexity | Training complexity |

## When to Use RAG

### Good Use Cases

- **Documentation Q&A**: "How do I use feature X?"
- **Customer support**: Up-to-date product info
- **Legal/compliance**: Cite specific policies
- **Research assistants**: Search across papers
- **Codebase Q&A**: Answer questions about your code

### Advantages

- No training required
- Easy to update (just re-index)
- Provides citations/sources
- Works with any base model
- Data stays outside model

### Limitations

- Retrieval quality bottleneck
- Added latency (100-500ms typically)
- Context window limits
- Can't change model behavior

## When to Fine-tune

### Good Use Cases

- **Custom writing style**: Match brand voice
- **Structured output**: Always return JSON
- **Domain adaptation**: Medical, legal, technical
- **Task specialization**: Classification, extraction
- **Efficiency**: Smaller model, same performance

### Advantages

- Lower latency (no retrieval)
- Consistent behavior
- Can work with smaller models
- Deep domain adaptation

### Limitations

- Expensive to update
- Can forget or distort knowledge
- Training data quality critical
- Risk of overfitting

## Hybrid Approach

Often the best solution combines both:

```
┌─────────────────────────────────────────────────────────────┐
│                     HYBRID SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Base Model (GPT-4, Claude, etc.)                          │
│         │                                                    │
│         ▼                                                    │
│   Fine-tuned Model                                          │
│   • Custom output format                                    │
│   • Domain terminology                                      │
│   • Response style                                          │
│         │                                                    │
│         ▼                                                    │
│   + RAG Context                                             │
│   • Current documentation                                   │
│   • User-specific data                                      │
│   • Recent information                                      │
│         │                                                    │
│         ▼                                                    │
│   Final Response                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Example: Customer Support Bot

```
Fine-tuning provides:
- Brand voice and tone
- Response format standards
- Escalation triggers

RAG provides:
- Current product documentation
- Pricing information
- Policy updates
- Known issues
```

## Cost Comparison

### RAG Costs

```
Per query:
- Embedding: ~$0.0001 (embed query)
- Vector DB: ~$0.0001 (search)
- LLM call: ~$0.01-0.10 (with context)

Monthly (1M queries):
- ~$10,000-100,000
```

### Fine-tuning Costs

```
Training (one-time):
- GPT-4 fine-tuning: ~$8/1M tokens
- Open source: Compute costs ($100-10,000)

Per query:
- Often cheaper than base model
- No retrieval overhead

Monthly (1M queries):
- Training: $100-10,000 (amortized)
- Inference: Often 50% of base model cost
```

## Decision Matrix

| Requirement | RAG | Fine-tune | Both |
|-------------|-----|-----------|------|
| Latest documentation | ✓ | | |
| Custom JSON output | | ✓ | |
| Domain expertise + docs | | | ✓ |
| Low latency critical | | ✓ | |
| Frequently changing data | ✓ | | |
| Brand voice + knowledge | | | ✓ |
| Small model, high quality | | ✓ | |

## Key Takeaways

- RAG for dynamic knowledge, fine-tuning for behavior
- Start with RAG (faster to iterate)
- Fine-tune when you need style/format changes
- Combine both for complex production systems
- Measure and compare before committing

## References

- [OpenAI — Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [Chip Huyen — RAG vs Fine-tuning](https://huyenchip.com/2023/04/11/llm-engineering.html)
- [Anthropic — Fine-tuning Best Practices](https://docs.anthropic.com/claude/docs/fine-tuning)
