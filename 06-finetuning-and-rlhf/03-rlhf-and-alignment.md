---
layout: default
title: RLHF and Alignment
parent: "06 - Fine-tuning & RLHF"
nav_order: 3
---

# RLHF and Alignment

## TL;DR

RLHF (Reinforcement Learning from Human Feedback) trains models to produce outputs humans prefer. It's how ChatGPT learned to be helpful and refuse harmful requests. Alternatives like DPO simplify the process while achieving similar results.

## The Alignment Problem

Base models and SFT models have issues:

```
Issue 1: Harmful content
User: "How do I make a bomb?"
SFT Model: "[Detailed instructions]"  ← Learned from internet

Issue 2: Unhelpful refusals  
User: "Write a story about a bank robbery"
SFT Model: "I cannot help with illegal activities"  ← Too restrictive

Issue 3: Inconsistent quality
Same prompt → Very different quality responses
```

RLHF addresses these by learning what humans actually prefer.

## RLHF Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        RLHF PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 1: COLLECT COMPARISONS                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Prompt: "Explain quantum computing"                     │   │
│   │                                                          │   │
│   │  Response A: "Quantum computing uses qubits..."          │   │
│   │  Response B: "It's complicated, look it up"              │   │
│   │  Response C: "QC leverages superposition and..."         │   │
│   │                                                          │   │
│   │  Human ranking: A > C > B                                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   Step 2: TRAIN REWARD MODEL                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Input: (prompt, response)                               │   │
│   │  Output: scalar reward score                             │   │
│   │                                                          │   │
│   │  Training: Learn to predict human preferences            │   │
│   │  Loss: -log(σ(r(preferred) - r(rejected)))              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   Step 3: OPTIMIZE POLICY WITH RL                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  LLM generates response                                  │   │
│   │  Reward model scores it                                  │   │
│   │  PPO updates LLM to maximize reward                      │   │
│   │  KL penalty prevents diverging too far from SFT model   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Collecting Human Preferences

### What Labelers Do

```
Given: Prompt + Multiple responses
Task: Rank from best to worst

Criteria (example):
- Helpfulness: Does it answer the question?
- Harmlessness: Is it safe?
- Honesty: Is it accurate?
```

### Comparison Data Format

```json
{
  "prompt": "What's the best way to learn programming?",
  "chosen": "Start with Python basics, build projects...",
  "rejected": "Just Google it, there are tutorials online"
}
```

### Scale

- OpenAI: ~100K+ comparisons for InstructGPT
- Anthropic: Millions for Claude
- Quality matters more than quantity

## Step 2: Reward Model

A model that predicts human preference:

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.model = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward

# Training objective
def reward_loss(preferred_reward, rejected_reward):
    return -torch.log(torch.sigmoid(preferred_reward - rejected_reward))
```

### What the Reward Model Learns

| High Reward | Low Reward |
|-------------|------------|
| Helpful, complete answers | Vague, unhelpful responses |
| Appropriate refusals | Harmful content |
| Accurate information | Hallucinations |
| Good formatting | Poorly structured |

## Step 3: RL Optimization (PPO)

```python
# Simplified PPO loop
for batch in prompts:
    # Generate responses from current policy
    responses = policy_model.generate(batch)
    
    # Get reward scores
    rewards = reward_model(batch, responses)
    
    # Compute KL penalty (don't diverge too far from SFT)
    kl = compute_kl(policy_model, sft_model, responses)
    
    # Adjusted reward
    adjusted_reward = rewards - β * kl
    
    # PPO update
    policy_model.update(adjusted_reward)
```

### Why KL Penalty?

Without KL penalty:
- Model might find "reward hacks"
- Exploit reward model weaknesses
- Produce degenerate outputs

```
Without KL: "HELPFUL! HELPFUL! I AM VERY HELPFUL!"
           (Reward model says this is helpful)

With KL:    Stays close to coherent SFT behavior
```

## Alternatives to RLHF

### DPO (Direct Preference Optimization)

Skip the reward model, train directly on preferences:

```python
# DPO loss
def dpo_loss(policy, reference, preferred, rejected, β):
    log_ratio_preferred = policy(preferred) - reference(preferred)
    log_ratio_rejected = policy(rejected) - reference(rejected)
    
    return -log(sigmoid(β * (log_ratio_preferred - log_ratio_rejected)))
```

Advantages:
- No reward model needed
- No RL instability
- Simpler to implement
- Similar results to RLHF

### RLAIF (RL from AI Feedback)

Use another LLM instead of humans:

```
1. Generate responses
2. Ask GPT-4/Claude to rank them
3. Train reward model on AI preferences
4. Run PPO as usual
```

Cheaper than human annotation, but introduces biases.

### Constitutional AI (Anthropic)

```
1. Generate response
2. Ask model to critique itself based on principles
3. Ask model to revise
4. Train on (original, revised) pairs

Principles:
- "Please choose the response that is most helpful"
- "Please choose the response that is least harmful"
- "Please choose the response that is most honest"
```

## Alignment Challenges

### Reward Hacking

Model finds shortcuts to maximize reward without being actually helpful:

```
Reward model likes longer responses
→ Model produces unnecessarily verbose answers

Reward model likes confidence
→ Model never says "I don't know"
```

### Specification Gaming

```
Task: "Be helpful"
Hack: Always agree with user (sycophancy)

Task: "Be safe"
Hack: Refuse everything remotely sensitive
```

### Capability vs Safety Trade-off

```
More aligned → More refusals → Less capable
More capable → Fewer refusals → More risk
```

## Key Takeaways

- RLHF teaches models what humans prefer
- Three steps: collect preferences → train reward model → optimize with RL
- DPO is a simpler alternative with similar results
- Alignment is an ongoing challenge, not a solved problem
- Balance helpfulness and harmlessness carefully

## References

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Chip Huyen — RLHF Explained](https://huyenchip.com/2023/05/02/rlhf.html)
