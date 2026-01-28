---
layout: default
title: Training Pipeline Overview
parent: 06 - Fine-tuning & RLHF
nav_order: 1
---

# Training Pipeline Overview

## TL;DR

Modern LLMs go through three stages: pretraining (learn language), supervised fine-tuning (learn to follow instructions), and RLHF (learn to be helpful/harmless). Each stage serves a different purpose.

## The Three Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   STAGE 1: PRETRAINING                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Internet Text (trillions of tokens)                     │   │
│   │  → Predict next token                                    │   │
│   │  → Learn language, facts, patterns                       │   │
│   │  → Produces: Base Model (GPT-4-base, Llama-base)        │   │
│   │                                                          │   │
│   │  Cost: $10M - $100M+   Time: Months                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   STAGE 2: SUPERVISED FINE-TUNING (SFT)                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Human-written examples (100k+ conversations)            │   │
│   │  → Learn instruction-following format                    │   │
│   │  → Learn helpful response patterns                       │   │
│   │  → Produces: Instruct Model (GPT-4-turbo, Llama-Instruct)│   │
│   │                                                          │   │
│   │  Cost: $100K - $1M   Time: Days-Weeks                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   STAGE 3: RLHF / ALIGNMENT                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Human preference rankings                               │   │
│   │  → Learn what humans prefer                              │   │
│   │  → Optimize for helpfulness, harmlessness                │   │
│   │  → Produces: Aligned Model (ChatGPT, Claude)            │   │
│   │                                                          │   │
│   │  Cost: $100K - $10M   Time: Weeks                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Stage 1: Pretraining

### What Happens

```
Training Data: "The quick brown fox jumps over the lazy"
Target: "dog"

Model learns: P(next_token | previous_tokens)
```

Trillions of tokens from:
- Web pages (Common Crawl)
- Books
- Code repositories
- Wikipedia
- Scientific papers

### What the Model Learns

| Learns | Doesn't Learn |
|--------|--------------|
| Grammar, syntax | How to be helpful |
| World knowledge | When to refuse |
| Reasoning patterns | Conversation format |
| Code structure | Safety guidelines |

### Base Model Behavior

```
Input: "What is the capital of France?"

Base Model Output: "
What is the capital of Germany?
What is the capital of Spain?
What is the capital of Italy?
..."

(It completes text, doesn't answer questions)
```

## Stage 2: Supervised Fine-Tuning (SFT)

### What Happens

```
Training Data:
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

Thousands of high-quality examples written by humans.

### What the Model Learns

- Follow instructions
- Respond in conversation format
- Be helpful and informative
- Basic safety behaviors

### After SFT

```
Input: "What is the capital of France?"

SFT Model Output: "The capital of France is Paris."

Much better! But still issues:
- May still produce harmful content
- Quality varies
- Doesn't reliably refuse dangerous requests
```

## Stage 3: RLHF (Reinforcement Learning from Human Feedback)

### The Process

```
1. Generate multiple responses to same prompt
   
   Prompt: "How do I pick a lock?"
   
   Response A: "Here's a step-by-step guide..."
   Response B: "I can't help with that, but..."
   Response C: "Lock picking is a skill used by..."

2. Humans rank responses: B > C > A

3. Train a Reward Model to predict human preferences

4. Use RL (PPO) to optimize the LLM to maximize reward
```

### What the Model Learns

- Which responses humans prefer
- How to be helpful AND harmless
- When to refuse vs when to help
- Nuance in tricky situations

## Comparison

| Aspect | Pretraining | SFT | RLHF |
|--------|-------------|-----|------|
| Data size | Trillions tokens | 100K+ examples | 100K+ comparisons |
| Data type | Raw text | Demonstrations | Preferences |
| Objective | Next token | Imitate humans | Maximize reward |
| Computes | Massive | Moderate | Moderate |
| Result | Knowledge | Format | Alignment |

## Why Each Stage Matters

### Without Pretraining
- No language understanding
- No world knowledge
- Must learn everything from scratch

### Without SFT
- Doesn't follow instructions
- Just completes text
- Unusable as assistant

### Without RLHF
- Inconsistent quality
- May produce harmful content
- Doesn't handle edge cases well

## What You Can Customize

| Stage | Customizable? | How |
|-------|---------------|-----|
| Pretraining | Rarely | Train from scratch (expensive) |
| SFT | Yes | Fine-tune on your data |
| RLHF | Sometimes | DPO, RLAIF on preferences |

Most fine-tuning happens at the SFT stage.

## Key Takeaways

- Pretraining creates the foundation (language + knowledge)
- SFT teaches instruction-following format
- RLHF aligns with human preferences
- Each stage builds on the previous
- You typically fine-tune at the SFT stage

## References

- [Karpathy — State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [Anthropic — Constitutional AI](https://www.anthropic.com/research/constitutional-ai)
