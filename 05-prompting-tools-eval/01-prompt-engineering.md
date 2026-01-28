---
layout: default
title: Prompt Engineering
parent: "05 - Prompting & Eval"
nav_order: 1
---

# Prompt Engineering

## TL;DR

Prompt engineering is the art of crafting inputs that reliably produce desired outputs. It's not magic—it's systematic application of patterns that work with how LLMs process text.

## Core Principles

### 1. Be Specific

```
❌ Bad:  "Write about dogs"
✓ Good: "Write a 200-word blog post about golden retrievers, 
         covering their temperament, exercise needs, and 
         suitability for families with young children."
```

### 2. Provide Context

```
❌ Bad:  "Is this good?"
✓ Good: "You are a senior code reviewer. Review this Python 
         function for correctness, performance, and readability.
         
         [code here]
         
         Provide specific feedback with line numbers."
```

### 3. Show, Don't Tell (Few-Shot)

```
Convert the following to JSON:

Input: John is 30 years old and lives in NYC
Output: {"name": "John", "age": 30, "city": "NYC"}

Input: Sarah, 25, from London
Output: {"name": "Sarah", "age": 25, "city": "London"}

Input: Mike is 40 and based in Tokyo
Output:
```

## Key Techniques

### Chain-of-Thought (CoT)

Ask the model to reason step by step:

```
Solve this problem step by step:

Q: If a train travels at 60 mph for 2.5 hours, then 80 mph 
   for 1.5 hours, what's the total distance?

Let's think through this:
1. First segment: 60 mph × 2.5 hours = 150 miles
2. Second segment: 80 mph × 1.5 hours = 120 miles
3. Total: 150 + 120 = 270 miles
```

**When to use:** Math, logic, multi-step reasoning, debugging.

### System Prompts

Set the context and persona:

```
System: You are an expert Python developer with 15 years of 
experience. You write clean, well-documented code following 
PEP 8 guidelines. When reviewing code, you focus on security, 
performance, and maintainability.
```

### Structured Output

Request specific formats:

```
Analyze the sentiment of the following review and respond 
in exactly this JSON format:

{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": 0.0-1.0,
  "key_phrases": ["phrase1", "phrase2"]
}

Review: "The food was amazing but the service was slow"
```

### Role Prompting

```
You are a [specific role] with expertise in [domain].

Examples:
- "You are a senior security engineer reviewing code for vulnerabilities"
- "You are a data scientist explaining concepts to a business audience"
- "You are a skeptical reviewer who challenges assumptions"
```

### Decomposition

Break complex tasks into steps:

```
I need to build a REST API. Let's approach this systematically:

1. First, list all the endpoints we need
2. For each endpoint, define the request/response schema
3. Identify shared models and utilities
4. Write the implementation

Let's start with step 1...
```

## Advanced Patterns

### Self-Critique

```
Write a solution, then critique it:

[Generate solution]

Now, identify potential issues with this solution:
- What edge cases might fail?
- What assumptions are we making?
- How could this be improved?

Based on your critique, provide an improved version.
```

### Constrained Generation

```
Write a function that:
- Is under 20 lines
- Has no external dependencies
- Handles edge cases explicitly
- Includes type hints
- Has a docstring
```

### Negative Examples

```
Good response:
"The error occurs because the array index exceeds bounds.
 Fix: Check array length before accessing."

Bad response (don't do this):
"There's an error in your code."

Now analyze this error...
```

## Prompt Structure Template

```markdown
# Role/Context
You are [role] with expertise in [domain].

# Task
[Clear description of what you want]

# Format
[Exact output format expected]

# Examples (if few-shot)
Input: [example]
Output: [example]

# Constraints
- [Constraint 1]
- [Constraint 2]

# Input
[Actual input to process]
```

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Too vague | Inconsistent outputs | Be specific about format, length, style |
| No examples | Model guesses format | Provide 2-3 examples |
| Contradictions | Confused model | Review prompt for conflicts |
| Too long | Buried instructions | Important stuff first and last |
| Assuming knowledge | Missing context | Provide necessary background |

## Temperature & Sampling

| Parameter | Low (0-0.3) | Medium (0.5-0.7) | High (0.8-1.0) |
|-----------|-------------|------------------|----------------|
| Use case | Facts, code, classification | General use | Creative, brainstorming |
| Output | Deterministic, focused | Balanced | Varied, creative |
| Risk | Repetitive | - | Inconsistent, off-topic |

## Key Takeaways

- Specificity beats cleverness
- Show examples when possible (few-shot)
- Use Chain-of-Thought for reasoning tasks
- Structure prompts consistently
- Iterate and test systematically

## References

- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
